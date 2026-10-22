"""Colored human point cloud -> colored PLY mesh.

Run from the VS Code terminal in your existing Python environment.
See requirements.txt for dependencies and docs/使用说明.md for commands.
Input/output coordinates are millimeters by default. Geometry processing uses
meters internally. Open3D's legacy Screened Poisson implementation runs on CPU.
"""

import argparse
import copy
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from surface_detail import denoise_local_planes, fit_local_planes, select_main_point_component


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "export_00000.ply"
PRESETS = {
    "fast": {"voxel": 0.004, "depth": 8, "faces": 60000},
    "balanced": {"voxel": 0.003, "depth": 9, "faces": 100000},
    "quality": {"voxel": 0.002, "depth": 10, "faces": 200000},
    "detail": {"voxel": 0.001, "depth": 10, "faces": 0,
               "normal_radius": 0.008, "normal_max_nn": 32,
               "normal_method": "robust"},
    "scan": {"voxel": 0.002, "depth": 9, "faces": 0,
             "normal_radius": 0.012, "normal_max_nn": 64,
             "normal_method": "robust", "point_denoise_iterations": 2,
             "point_denoise_max_move_mm": 1.0, "point_components": "largest",
             "max_extrapolation_mm": 0.0, "smooth_iterations": 2},
}
UNIT_TO_METERS = {"m": 1.0, "mm": 0.001}


@contextmanager
def stage(name, timings):
    print(f"\n[{name}]", flush=True)
    started = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - started, 3)
    print(f"  {timings[name]:.2f} s", flush=True)


def clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles):
        cross = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                         vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
        mesh.remove_triangles_by_mask(np.linalg.norm(cross, axis=1) == 0)
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError("Invalid/empty mesh after cleanup.")
    return mesh


def geometry_report(mesh, check_intersections=True):
    check_timings = {}
    with stage("topology: face_orientation_and_components", check_timings):
        oriented = bool(mesh.orient_triangles())
        _, counts, areas = mesh.cluster_connected_triangles()
        counts, areas = np.asarray(counts), np.asarray(areas)
    with stage("topology: edge_counts", check_timings):
        triangles = np.asarray(mesh.triangles)
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                        triangles[:, [2, 0]]]), axis=1)
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        # Reuse the counts: a closed manifold edge has exactly two faces.
        closed_edges = bool(len(edge_counts) and np.all(edge_counts == 2))
        del edges
    with stage("topology: vertex_manifold", check_timings):
        manifold_vertices = bool(mesh.is_vertex_manifold())
    intersecting = None
    if check_intersections:
        print(f"  Self-intersection scan: {len(mesh.triangles):,} faces; "
              "high-resolution meshes may take a long time. "
              "Use --skip-self-intersections for appearance previews.", flush=True)
        with stage("topology: self_intersections", check_timings):
            intersecting = bool(mesh.is_self_intersecting())
    else:
        print("  Self-intersection check skipped; watertight/print gate remain unknown.", flush=True)
    watertight = (closed_edges and manifold_vertices and not intersecting
                  if check_intersections else None)
    signed_volume = None
    if closed_edges and manifold_vertices and oriented and len(counts) == 1:
        vertices = np.asarray(mesh.vertices)
        signed_volume = float(np.einsum(
            "ij,ij->i", vertices[triangles[:, 0]],
            np.cross(vertices[triangles[:, 1]], vertices[triangles[:, 2]])
        ).sum() / 6.0)
        if signed_volume < 0:
            mesh.triangles = o3d.utility.Vector3iVector(triangles[:, [0, 2, 1]].copy())
            signed_volume = -signed_volume
    extent = float(np.max(mesh.get_max_bound() - mesh.get_min_bound()))
    gate = (bool(watertight and oriented and len(counts) == 1
                 and signed_volume is not None and signed_volume > extent ** 3 * 1e-12)
            if check_intersections else None)
    with stage("topology: vertex_normals", check_timings):
        mesh.compute_vertex_normals()
    order = np.argsort(areas)[::-1]
    return {
        "vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
        "connected_components": len(counts),
        "largest_components": [{"triangles": int(counts[i]), "area_m2": float(areas[i])}
                               for i in order[:10]],
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "edges_with_more_than_two_faces": int(np.count_nonzero(edge_counts > 2)),
        "closed_edge_manifold": closed_edges, "vertex_manifold": manifold_vertices,
        "orientation_succeeded": oriented, "self_intersecting": intersecting,
        "self_intersections_checked": check_intersections, "watertight": watertight,
        "volume_m3_single_closed_component": signed_volume,
        "geometry_gate_passed": gate,
        "topology_check_timings_seconds": check_timings,
    }


def distance_stats(distances):
    return {"median_mm": float(np.median(distances) * 1000),
            "p95_mm": float(np.quantile(distances, 0.95) * 1000),
            "max_mm": float(np.max(distances) * 1000)}


def transfer_colors(mesh, source, origin, k, workers, batch_size=50000):
    """Batch KD-tree queries, never an N_mesh x N_cloud distance matrix.

    k=1 copies the nearest original RGB exactly before PLY's uint8 encoding.
    k>1 interpolates inverse-distance colors and can soften color boundaries.
    Far-away inferred surfaces still receive nearest colors; report their support.
    """
    if not source.has_colors():
        return {"preserved": False, "reason": "Input PLY has no RGB properties."}
    source_points = np.asarray(source.points)
    source_colors = np.asarray(source.colors)
    valid = np.isfinite(source_colors).all(axis=1)
    if not valid.any():
        raise RuntimeError("Input RGB contains no finite colors.")
    tree = cKDTree(source_points[valid] - origin)
    source_colors = np.clip(source_colors[valid], 0, 1)
    k = min(k, len(source_colors))
    vertices = np.asarray(mesh.vertices)
    colors = np.empty((len(vertices), 3), dtype=np.float64)
    nearest_distances = np.empty(len(vertices), dtype=np.float64)
    for start in range(0, len(vertices), batch_size):
        stop = min(start + batch_size, len(vertices))
        distances, indices = tree.query(vertices[start:stop], k=k, workers=workers)
        if k == 1:
            colors[start:stop] = source_colors[indices]
            nearest_distances[start:stop] = distances
        else:
            weights = 1.0 / np.maximum(distances, 1e-12)
            weights /= weights.sum(axis=1, keepdims=True)
            colors[start:stop] = np.einsum("nk,nkc->nc", weights, source_colors[indices])
            nearest_distances[start:stop] = distances[:, 0]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return {"preserved": True, "source": "original finite unique input points",
            "neighbors": k, "transfer": "nearest RGB" if k == 1 else "inverse distance RGB",
            "invalid_color_points_ignored": int(np.count_nonzero(~valid)),
            "vertex_to_original_point": distance_stats(nearest_distances),
            "vertices_more_than_10mm_from_observation": int((nearest_distances > 0.010).sum())}


def visualize(mesh, pcd):
    """Interactive same-coordinate comparison: 1 mesh, 2 points, 3 overlay, W wireframe."""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name="Human reconstruction | 1 mesh  2 points  3 both  W wire",
                             width=1280, height=800):
        print("Visualization unavailable. PLY/report were saved; try a local desktop session.")
        return False
    try:
        vis.add_geometry(mesh)
        options = vis.get_render_option()
        options.background_color = np.array([0.10, 0.10, 0.12])
        options.point_size = 2.0
        options.mesh_show_back_face = True
        # Open3D 0.19 Python enum names are Default/Color; numeric 1 is
        # SmoothShade in the C++ renderer (0 is FlatShade).
        options.mesh_shade_option = o3d.visualization.MeshShadeOption(1)
        original_mesh_colors = np.asarray(mesh.vertex_colors).copy()
        gray_mode = False

        def toggle_geometry_color(viewer):
            nonlocal gray_mode
            gray_mode = not gray_mode
            if gray_mode:
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
            else:
                mesh.vertex_colors = o3d.utility.Vector3dVector(original_mesh_colors)
            viewer.update_geometry(mesh)
            viewer.update_renderer()
            return False

        def show(geometries):
            def callback(viewer):
                viewer.clear_geometries()
                for geometry in geometries:
                    viewer.add_geometry(geometry, reset_bounding_box=False)
                viewer.update_renderer()
                return False
            return callback

        def toggle_wire(viewer):
            opt = viewer.get_render_option()
            opt.mesh_show_wireframe = not opt.mesh_show_wireframe
            viewer.update_renderer()
            return False

        def toggle_shading(viewer):
            opt = viewer.get_render_option()
            smooth = opt.mesh_shade_option == o3d.visualization.MeshShadeOption(1)
            opt.mesh_shade_option = o3d.visualization.MeshShadeOption(0 if smooth else 1)
            viewer.update_renderer()
            return False

        vis.register_key_callback(ord("1"), show([mesh]))
        vis.register_key_callback(ord("2"), show([pcd]))
        vis.register_key_callback(ord("3"), show([mesh, pcd]))
        vis.register_key_callback(ord("W"), toggle_wire)
        vis.register_key_callback(ord("C"), toggle_geometry_color)
        vis.register_key_callback(ord("N"), toggle_shading)
        print("Viewer: 1=mesh, 2=original points, 3=overlay, W=wireframe, C=gray/RGB, N=flat/smooth shading.", flush=True)
        print("Mouse: left=rotate, wheel=zoom, Ctrl+left=pan; Q/Esc=close.", flush=True)
        vis.run()
    finally:
        vis.destroy_window()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path,
                        help="Output directory; default: script folder / outputs / preset.")
    parser.add_argument("--input-unit", choices=UNIT_TO_METERS, default="mm")
    parser.add_argument("--output-unit", choices=["input", "m", "mm"], default="input")
    parser.add_argument("--preset", choices=PRESETS, default="balanced")
    parser.add_argument("--voxel", type=float, help="Meters, default 0.003 in balanced preset.")
    parser.add_argument("--depth", type=int, help="Poisson octree depth, default 9.")
    parser.add_argument("--faces", type=int, help="Target face count; 0 disables decimation.")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1),
                        help="Poisson and KD-tree workers; -1=all, default up to 8.")
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std", type=float, default=2.5)
    parser.add_argument("--normal-radius", type=float, help="Meters; balanced 12 mm, detail 8 mm.")
    parser.add_argument("--normal-max-nn", type=int)
    parser.add_argument("--normal-method", choices=["legacy", "robust"])
    parser.add_argument("--point-denoise-iterations", type=int,
                        help="0..3 robust local-plane filtering iterations; default off.")
    parser.add_argument("--point-denoise-max-move-mm", type=float,
                        help="Total point displacement cap across all iterations, default 0.5 mm.")
    parser.add_argument("--point-components", choices=["all", "largest"],
                        help="Pre-Poisson point clustering; scan=largest, other presets=all.")
    parser.add_argument("--point-cluster-radius-mm", type=float, default=12.0)
    parser.add_argument("--max-extrapolation-mm", type=float,
                        help="Drop Poisson vertices farther than this from retained observations; 0=off. Can open holes.")
    parser.add_argument("--crop-bounds", type=float, nargs=6,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                        help="Optional ROI box in input coordinate units; exports a separate local mesh.")
    parser.add_argument("--orient-k", type=int, default=30)
    parser.add_argument("--poisson-scale", type=float, default=1.1)
    parser.add_argument("--trim-quantile", type=float, default=0.0,
                        help="0 to 0.05; removing low-density vertices can open holes.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--keep-largest", dest="keep_largest", action="store_true",
                           help="Default: keep largest surface component (the human in this scan).")
    selection.add_argument("--keep-all-components", dest="keep_largest", action="store_false",
                           help="Keep disconnected anatomy and extraneous fragments for inspection.")
    parser.set_defaults(keep_largest=True)
    parser.add_argument("--smooth-iterations", type=int)
    parser.add_argument("--color-k", type=int, default=1, help="1=preserve nearest RGB, 3-4=interpolate.")
    parser.add_argument("--qa-samples", type=int, default=10000, help="0 skips distance QA.")
    parser.add_argument("--skip-self-intersections", action="store_true",
                        help="Faster inspection only; report will NOT assert watertight/print gate.")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--preview-normals", action="store_true")
    parser.add_argument("--no-vis", action="store_true", help="Batch run without interactive viewer.")
    parser.add_argument("--overwrite", action="store_true", help="Replace previous generated output.")
    args = parser.parse_args()
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name, value in {"normal_radius": 0.012, "normal_max_nn": 40,
                        "normal_method": "legacy", "point_denoise_iterations": 0,
                        "point_denoise_max_move_mm": 0.5, "point_components": "all",
                        "max_extrapolation_mm": 0.0, "smooth_iterations": 0}.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ["voxel", "outlier_std", "normal_radius", "poisson_scale"]:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not 5 <= args.depth <= 12 or args.poisson_scale <= 1:
        parser.error("depth must be 5..12 and poisson-scale > 1")
    if min(args.outlier_neighbors, args.normal_max_nn, args.orient_k) < 3:
        parser.error("Neighbor counts must be >= 3")
    if (args.faces != 0 and args.faces < 4) or args.qa_samples < 0 or not 1 <= args.color_k <= 8:
        parser.error("faces = 0 or >= 4, qa-samples >= 0, color-k = 1..8")
    if not 0 <= args.point_denoise_iterations <= 3:
        parser.error("point-denoise-iterations must be 0..3")
    if not np.isfinite(args.point_denoise_max_move_mm) or args.point_denoise_max_move_mm <= 0:
        parser.error("point-denoise-max-move-mm must be finite and positive")
    if not np.isfinite(args.point_cluster_radius_mm) or args.point_cluster_radius_mm < 4:
        parser.error("point-cluster-radius-mm must be finite and >= 4 for the 4mm proxy")
    if not np.isfinite(args.max_extrapolation_mm) or args.max_extrapolation_mm < 0:
        parser.error("max-extrapolation-mm must be finite and >= 0")
    if (args.normal_method == "robust" or args.point_denoise_iterations) and args.normal_max_nn < 6:
        parser.error("robust local-plane fitting requires normal-max-nn >= 6")
    if args.crop_bounds is not None:
        bounds = np.asarray(args.crop_bounds)
        if not np.isfinite(bounds).all() or np.any(bounds[:3] >= bounds[3:]):
            parser.error("crop-bounds must be finite and each minimum < maximum")
    if args.threads == 0 or args.threads < -1:
        parser.error("threads must be -1 or positive")
    if not 0 <= args.smooth_iterations <= 5:
        parser.error("smooth-iterations must be 0..5")
    if not np.isfinite(args.trim_quantile) or not 0 <= args.trim_quantile <= 0.05:
        parser.error("trim-quantile must be 0..0.05")
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if args.out is None:
        args.out = SCRIPT_DIR / "outputs" / args.preset
    args.input, args.out = args.input.resolve(), args.out.resolve()
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    timings = {}
    if o3d.__version__ != "0.19.0":
        print(f"Notice: Open3D {o3d.__version__}; tested version is 0.19.0.")
    print(f"Input: {args.input}\nOutput folder: {args.out}")
    print(f"CPU Poisson: depth={args.depth}, voxel={args.voxel * 1000:g} mm, threads={args.threads}")
    print(f"Open3D CUDA available: {o3d.core.cuda.is_available()} (Poisson still uses CPU)")
    output_unit = args.input_unit if args.output_unit == "input" else args.output_unit
    with stage("read_and_clean", timings):
        source = o3d.io.read_point_cloud(str(args.input))
        if not source.has_points():
            raise RuntimeError("Empty or unsupported point cloud.")
        input_count = len(source.points)
        source.remove_non_finite_points()
        source.remove_duplicated_points()
        finite_unique_count = len(source.points)
        if args.crop_bounds is not None:
            source = source.crop(o3d.geometry.AxisAlignedBoundingBox(
                np.asarray(args.crop_bounds[:3]), np.asarray(args.crop_bounds[3:])))
        if len(source.points) < 100:
            raise RuntimeError("Too few finite unique points.")
        source.scale(UNIT_TO_METERS[args.input_unit], center=np.zeros(3))
        if not source.has_colors():
            print("WARNING: Input has no RGB; output will have geometry without source colors.")
        source_extent = source.get_max_bound() - source.get_min_bound()
        if args.crop_bounds is None and not 0.2 <= np.max(source_extent) <= 5.0:
            print(f"WARNING: body extent {source_extent} m; check --input-unit or extraneous geometry.")
    with stage("downsample_and_filter", timings):
        pcd = source.voxel_down_sample(args.voxel)
        voxel_count = len(pcd.points)
        if voxel_count <= args.outlier_neighbors:
            raise RuntimeError("Too few voxel points; check input unit and voxel size.")
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.outlier_neighbors, std_ratio=args.outlier_std)
        sor_count = len(pcd.points)
        point_component_report = {"method": "disabled", "removed_points": 0}
        component_rejected = None
        if args.point_components == "largest":
            pcd, component_rejected, point_component_report = select_main_point_component(
                pcd, args.point_cluster_radius_mm * .001)
            print(f"  Point clustering: {sor_count} -> {len(pcd.points)}; "
                  f"removed {point_component_report['removed_points']} points before normals/Poisson", flush=True)
        if len(pcd.points) <= max(args.normal_max_nn, args.orient_k):
            raise RuntimeError("Too few filtered points.")
        origin = (pcd.get_min_bound() + pcd.get_max_bound()) / 2
        pcd.translate(-origin)
        extent = pcd.get_max_bound() - pcd.get_min_bound()
        rng = np.random.default_rng(0)
        indices = rng.choice(len(pcd.points), min(10000, len(pcd.points)), replace=False)
        local_points = np.asarray(pcd.points)
        spacing = float(np.median(cKDTree(local_points).query(
            local_points[indices], k=2, workers=args.threads)[0][:, 1]))
        print(f"  Points: {input_count} -> {voxel_count} -> {len(pcd.points)}; extent(m): {extent}")
    # Keep unshifted observations for QA; fitting against moved points alone can
    # hide denoising bias. Positions here already use the shared local origin.
    observations = copy.deepcopy(pcd)
    point_denoise_report = {"iterations": 0, "method": "disabled"}
    if args.point_denoise_iterations:
        with stage("robust_point_denoise", timings):
            denoised, point_denoise_report = denoise_local_planes(
                np.asarray(pcd.points), args.normal_max_nn, args.normal_radius,
                args.threads, args.point_denoise_iterations,
                args.point_denoise_max_move_mm * 0.001)
            pcd.points = o3d.utility.Vector3dVector(denoised)
            print(f"  Maximum point movement: {point_denoise_report['movement_max_mm']:.3f} mm")
    normal_report = {"method": args.normal_method}
    with stage("estimate_normals", timings):
        # Discard inconsistent normals stored in this scan; estimate from geometry.
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.normal_radius, max_nn=args.normal_max_nn))
        if args.normal_method == "robust":
            fitted, _, supported, _, support_count = fit_local_planes(
                np.asarray(pcd.points), args.normal_max_nn, args.normal_radius, args.threads)
            normals = np.asarray(pcd.normals).copy()
            normals[supported] = fitted[supported]
            pcd.normals = o3d.utility.Vector3dVector(normals)
            normal_report.update({"robust_fitted_points": int(supported.sum()),
                                  "legacy_fallback_points": int((~supported).sum()),
                                  "median_neighborhood_points": float(np.median(support_count))})
            print(f"  Robust normal fits: {supported.sum()}; fallback: {(~supported).sum()}")
        pcd.normalize_normals()
    with stage("orient_normals", timings):
        pcd.orient_normals_consistent_tangent_plane(args.orient_k)
        normals = np.asarray(pcd.normals)
        if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) < 1e-8):
            raise RuntimeError("Invalid normals after estimation/orientation.")
        if args.normal_radius < 3 * spacing:
            print("WARNING: normal radius < 3x point spacing; inspect surface stability.")
    if args.preview_normals:
        o3d.visualization.draw_geometries([pcd], window_name="Inspect oriented normals",
                                         point_show_normal=True)
    extent = pcd.get_max_bound() - pcd.get_min_bound()
    with stage("poisson_cpu", timings):
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.depth, scale=args.poisson_scale, n_threads=args.threads)
        # Densities use original Poisson vertex indices: trim before cleanup.
        if args.trim_quantile > 0:
            density = np.asarray(density)
            mesh.remove_vertices_by_mask(density < np.quantile(density, args.trim_quantile))
        extrapolation_report = {"enabled": False, "removed_vertices": 0}
        if args.max_extrapolation_mm > 0:
            distance, _ = cKDTree(np.asarray(observations.points)).query(
                np.asarray(mesh.vertices), workers=args.threads)
            unsupported = distance > args.max_extrapolation_mm * .001
            extrapolation_report = {"enabled": True, "limit_mm": args.max_extrapolation_mm,
                                    "vertices_before": len(mesh.vertices),
                                    "removed_vertices": int(unsupported.sum())}
            mesh.remove_vertices_by_mask(unsupported)
            print(f"  Removed {unsupported.sum()} unsupported Poisson vertices", flush=True)
        mesh = clean_mesh(mesh)
        del density
    with stage("postprocess_and_decimate", timings):
        labels, component_counts, areas = mesh.cluster_connected_triangles()
        components_before = len(component_counts)
        removed_faces = 0
        if args.keep_largest:
            remove = np.asarray(labels) != int(np.argmax(areas))
            removed_faces = int(remove.sum())
            mesh.remove_triangles_by_mask(remove)
            mesh = clean_mesh(mesh)
            print(f"  Surface components: {components_before}; removed {removed_faces} non-main faces")
        if args.smooth_iterations:
            mesh = clean_mesh(mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iterations))
            if mesh.has_vertex_colors():
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(mesh.vertex_colors), 0, 1))
        highres_closed_edges = bool(mesh.is_edge_manifold(allow_boundary_edges=False))
        highres_manifold_vertices = bool(mesh.is_vertex_manifold())
        highres_mesh = copy.deepcopy(mesh) if args.save_intermediates else None
        faces_before = len(mesh.triangles)
        if args.faces > 0 and faces_before > args.faces:
            mesh = clean_mesh(mesh.simplify_quadric_decimation(target_number_of_triangles=args.faces))
        print(f"  Faces: {faces_before} -> {len(mesh.triangles)} (target {args.faces})")
    with stage("topology_checks", timings):
        report = geometry_report(mesh, check_intersections=not args.skip_self_intersections)
    with stage("original_rgb_transfer", timings):
        report["color"] = transfer_colors(mesh, source, origin, args.color_k, args.threads)
    if args.qa_samples:
        with stage("distance_qa", timings):
            points = np.asarray(observations.points)
            indices = rng.choice(len(points), min(args.qa_samples, len(points)), replace=False)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            distances = scene.compute_distance(o3d.core.Tensor(
                points[indices].astype(np.float32), dtype=o3d.core.Dtype.Float32)).numpy()
            o3d.utility.random.seed(0)
            samples = mesh.sample_points_uniformly(number_of_points=args.qa_samples)
            report["processed_observations_to_mesh"] = distance_stats(distances)
            report["sampled_mesh_to_processed_observations"] = distance_stats(
                np.asarray(samples.compute_point_cloud_distance(observations)))
    args.out.mkdir(parents=True, exist_ok=True)
    basename = args.input.stem
    if args.crop_bounds is not None:
        basename += "_roi"
    mesh_path = args.out / f"{basename}_mesh.ply"
    report_path = args.out / f"{basename}_report.json"
    if not args.overwrite and (mesh_path.exists() or report_path.exists()):
        basename += "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mesh_path = args.out / f"{basename}_mesh.ply"
        report_path = args.out / f"{basename}_report.json"
    with stage("write_colored_ply", timings):
        exported = copy.deepcopy(mesh)
        exported.translate(origin)
        exported.scale(1 / UNIT_TO_METERS[output_unit], center=np.zeros(3))
        if not o3d.io.write_triangle_mesh(str(mesh_path), exported, write_ascii=False,
                                         write_vertex_normals=True, write_vertex_colors=True):
            raise OSError(f"Failed to write {mesh_path}")
        if args.save_intermediates:
            highres_mesh.translate(origin)
            # Poisson already interpolates downsampled colors; final mesh always
            # receives original RGB again after simplification.
            highres_mesh.scale(1 / UNIT_TO_METERS[output_unit], center=np.zeros(3))
            if not o3d.io.write_triangle_mesh(str(args.out / f"{basename}_highres.ply"), highres_mesh,
                                             write_ascii=False, write_vertex_colors=True):
                raise OSError("Failed to write high resolution mesh.")
            processed = copy.deepcopy(pcd).translate(origin)
            processed.scale(1 / UNIT_TO_METERS[output_unit], center=np.zeros(3))
            if not o3d.io.write_point_cloud(str(args.out / f"{basename}_processed.ply"), processed):
                raise OSError("Failed to write processed point cloud.")
            if component_rejected is not None and component_rejected.has_points():
                rejected = copy.deepcopy(component_rejected)
                rejected.scale(1 / UNIT_TO_METERS[output_unit], center=np.zeros(3))
                if not o3d.io.write_point_cloud(str(args.out / f"{basename}_rejected_components.ply"), rejected):
                    raise OSError("Failed to write rejected point components.")
    report.update({
        "open3d_version": o3d.__version__,
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "input_points": input_count, "finite_unique_points": finite_unique_count,
        "roi_source_points": len(source.points),
        "point_denoise": point_denoise_report, "normal_estimation": normal_report,
        "point_components": point_component_report, "extrapolation_trim": extrapolation_report,
        "sor_retained_points": sor_count,
        "voxel_points": voxel_count, "filtered_points": len(pcd.points),
        "input_unit": args.input_unit, "output_coordinate_unit": output_unit,
        "processing_coordinate_unit": "m", "mesh_path": str(mesh_path),
        "meter_coordinate_origin": origin.tolist(), "extent_m": extent.tolist(),
        "median_sampled_point_spacing_mm": spacing * 1000,
        "finest_poisson_cube_cell_mm": float(args.poisson_scale * np.max(extent)
                                              / 2 ** args.depth * 1000),
        "poisson_resolution_note": "Nominal finest cube cell size, not guaranteed surface accuracy; octree is adaptive.",
        "triangles_before_decimation": faces_before, "target_triangles": args.faces,
        "components_before_selection": components_before, "removed_non_main_component_faces": removed_faces,
        "highres_closed_edge_manifold": highres_closed_edges,
        "highres_vertex_manifold": highres_manifold_vertices,
        "timings_seconds": timings, "total_before_visualization_seconds": round(time.perf_counter() - started, 3),
        "qa_note": "Distances use SOR-retained observations BEFORE optional position filtering, not ground truth. When keep-largest "
                   "removes fragments, point-to-mesh maxima can include those removed fragments. "
                   "Inferred surfaces have no true observed color. PLY does not store units.",
        "manual_checks": ["Inferred holes and anatomy", "Nearby finger/limb bridges",
                          "Disconnected components", "Color on inferred surfaces",
                          "Print scale, minimum thickness, orientation and supports"],
    })
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved colored mesh ({output_unit}): {mesh_path}\nReport: {report_path}")
    print(f"Total excluding viewer: {report['total_before_visualization_seconds']:.2f} s")
    print(f"Geometry gate: {report['geometry_gate_passed']}; components: {report['connected_components']}; "
          f"boundary edges: {report['boundary_edges']}")
    print("Review inferred anatomy and physical print constraints even when the geometry gate passes.")
    if not args.no_vis:
        display_points = copy.deepcopy(source).translate(-origin)
        visualize(mesh, display_points)


if __name__ == "__main__":
    main()
