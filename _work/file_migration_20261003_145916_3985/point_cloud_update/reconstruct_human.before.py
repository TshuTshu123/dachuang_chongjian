"""Nearly complete human scan -> mesh, with a conservative printing geometry gate.

Target: Python 3.10/3.11, Open3D 0.19.0, NumPy >=1.24,<3.
Input coordinates must be meters. The algorithm does not infer missing anatomy.
Example:
    python reconstruct_human.py human.ply --out result --faces 100000
Outputs: raw/reduced PLY in original meter coordinates, report.json, and a
centered millimeter STL only if the reduced mesh passes the geometry checks.
Passing those checks does not verify anatomical accuracy, wall thickness,
printer size, support requirements, or mechanical strength.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    if len(t):
        cross = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        mesh.remove_triangles_by_mask(np.linalg.norm(cross, axis=1) == 0)
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise RuntimeError("No triangles remain after cleanup.")
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError("Reconstruction contains non-finite vertices.")
    return mesh


def components(mesh):
    labels, counts, areas = mesh.cluster_connected_triangles()
    return np.asarray(labels), np.asarray(counts), np.asarray(areas)


def write_mesh(path, mesh):
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise OSError(f"Failed to write {path}")


def write_world_ply(path, mesh, origin):
    exported = copy.deepcopy(mesh)
    exported.translate(origin)
    write_mesh(path, exported)


def distance_stats(distances_m):
    distances_m = np.asarray(distances_m)
    return {
        "median_mm": float(np.median(distances_m) * 1000),
        "p95_mm": float(np.quantile(distances_m, 0.95) * 1000),
        "max_mm": float(np.max(distances_m) * 1000),
    }


def geometry_report(mesh):
    oriented = bool(mesh.orient_triangles())
    _, counts, areas = components(mesh)
    t = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    closed_edges = bool(mesh.is_edge_manifold(allow_boundary_edges=False))
    manifold_vertices = bool(mesh.is_vertex_manifold())
    intersecting = bool(mesh.is_self_intersecting())
    # Same geometric conditions as Open3D's is_watertight(), without repeating
    # the potentially expensive intersection test.
    watertight = closed_edges and manifold_vertices and not intersecting
    gate = watertight and oriented and len(counts) == 1
    signed_volume = None
    if gate:
        v = np.asarray(mesh.vertices)
        signed_volume = float(np.einsum(
            "ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])
        ).sum() / 6.0)
        if signed_volume < 0:
            mesh.triangles = o3d.utility.Vector3iVector(t[:, [0, 2, 1]].copy())
            signed_volume = -signed_volume
        extent = float(np.max(mesh.get_max_bound() - mesh.get_min_bound()))
        gate = signed_volume > extent ** 3 * 1e-12
    mesh.compute_vertex_normals()
    order = np.argsort(areas)[::-1]
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "connected_components": len(counts),
        "largest_components": [
            {"triangles": int(counts[i]), "area_m2": float(areas[i])}
            for i in order[:10]
        ],
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "edges_with_more_than_two_faces": int(np.count_nonzero(edge_counts > 2)),
        "closed_edge_manifold": closed_edges,
        "vertex_manifold": manifold_vertices,
        "orientation_succeeded": oriented,
        "self_intersecting": intersecting,
        "watertight": watertight,
        "volume_m3_single_closed_component": signed_volume,
        "geometry_gate_passed": bool(gate),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("human_result"),
                        help="New output directory; an existing directory is refused.")
    parser.add_argument("--voxel", type=float, default=0.003, help="Meters; default 3 mm.")
    parser.add_argument("--outlier-neighbors", type=int, default=30)
    parser.add_argument("--outlier-std", type=float, default=2.5)
    parser.add_argument("--normal-radius", type=float, default=0.010, help="Meters.")
    parser.add_argument("--normal-max-nn", type=int, default=60)
    parser.add_argument("--orient-k", type=int, default=50)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--poisson-scale", type=float, default=1.1)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--faces", type=int, default=100000)
    parser.add_argument("--trim-quantile", type=float, default=0.0,
                        help="Optional density quantile to discard; may create holes.")
    parser.add_argument("--keep-largest", action="store_true",
                        help="Keep largest component by surface area; may discard anatomy.")
    parser.add_argument("--smooth-iterations", type=int, default=0)
    parser.add_argument("--qa-samples", type=int, default=50000)
    parser.add_argument("--print-scale", type=float, default=1.0,
                        help="1 = original size; 0.1 = 1:10. STL numbers are millimeters.")
    parser.add_argument("--preview-normals", action="store_true")
    args = parser.parse_args()
    for name in ("voxel", "outlier_std", "normal_radius", "poisson_scale", "print_scale"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.poisson_scale <= 1 or not 5 <= args.depth <= 12:
        parser.error("Use poisson-scale > 1 and depth between 5 and 12 (recommend 9-11).")
    if not np.isfinite(args.trim_quantile) or not 0 <= args.trim_quantile <= 0.05:
        parser.error("trim-quantile must be between 0 and 0.05; default is 0.")
    if not 0 <= args.smooth_iterations <= 5:
        parser.error("smooth-iterations must be between 0 and 5.")
    if min(args.outlier_neighbors, args.normal_max_nn, args.orient_k) < 3:
        parser.error("Neighbor counts must be at least 3.")
    if args.faces < 4 or args.qa_samples < 1 or args.threads == 0 or args.threads < -1:
        parser.error("faces >= 4, qa-samples >= 1, threads = -1 or a positive integer.")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.out.exists():
        parser.error("Output directory already exists. Choose a new directory.")
    if o3d.__version__ != "0.19.0":
        print(f"Notice: using Open3D {o3d.__version__}; this script targets 0.19.0.")

    pcd = o3d.io.read_point_cloud(str(args.input))
    if not pcd.has_points():
        raise RuntimeError("Empty/unsupported input. Use a readable PLY/PCD/XYZ point cloud.")
    input_count = len(pcd.points)
    pcd.remove_non_finite_points()
    pcd.remove_duplicated_points()
    finite_unique_count = len(pcd.points)
    pcd = pcd.voxel_down_sample(args.voxel)
    if len(pcd.points) <= args.outlier_neighbors:
        raise RuntimeError("Too few points after voxel downsampling. Check units/voxel size.")
    voxel_count = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=args.outlier_neighbors, std_ratio=args.outlier_std
    )
    if len(pcd.points) <= max(args.orient_k, args.normal_max_nn):
        raise RuntimeError("Too few points after filtering. Check parameters/data.")
    origin = (pcd.get_min_bound() + pcd.get_max_bound()) / 2.0
    pcd.translate(-origin)  # Preserve meter scale; improve Float32 QA precision.
    extent = pcd.get_max_bound() - pcd.get_min_bound()
    spacing = float(np.median(pcd.compute_nearest_neighbor_distance()))
    cube_cell = float(args.poisson_scale * np.max(extent) / 2 ** args.depth)
    print(f"Points: {input_count} -> {len(pcd.points)}; extent(m): {extent}")
    print(f"Median spacing: {spacing * 1000:.2f} mm; finest cube cell: {cube_cell * 1000:.2f} mm")
    print("Cell size is an octree bound, not guaranteed geometric accuracy.")
    if args.normal_radius < 3 * spacing:
        print("Notice: normal radius is below 3x median spacing; inspect normal stability.")
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.normal_radius, max_nn=args.normal_max_nn
    ))
    pcd.normalize_normals()
    # Consistency does not guarantee outward normals or anatomically correct
    # propagation across nearby fingers/limbs. Inspect them when necessary.
    pcd.orient_normals_consistent_tangent_plane(args.orient_k)
    normals = np.asarray(pcd.normals)
    if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) == 0):
        raise RuntimeError("Invalid estimated normals. Inspect point neighborhoods/data.")
    if args.preview_normals:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    print("Reconstructing with CPU Screened Poisson...")
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth, scale=args.poisson_scale, n_threads=args.threads
    )
    if not mesh.has_triangles():
        raise RuntimeError("Poisson produced no triangles.")
    args.out.mkdir(parents=True, exist_ok=False)
    processed_world = copy.deepcopy(pcd)
    processed_world.translate(origin)
    if not o3d.io.write_point_cloud(str(args.out / "processed_m.ply"), processed_world):
        raise OSError("Failed to write processed point cloud.")
    del processed_world
    write_world_ply(args.out / "poisson_raw_m.ply", mesh, origin)
    if args.trim_quantile > 0:
        density = np.asarray(density)
        mesh.remove_vertices_by_mask(density < np.quantile(density, args.trim_quantile))
    mesh = clean_mesh(mesh)
    if args.keep_largest:
        labels, _, areas = components(mesh)
        mesh.remove_triangles_by_mask(labels != int(np.argmax(areas)))
        mesh = clean_mesh(mesh)
    if args.smooth_iterations:
        mesh = clean_mesh(mesh.filter_smooth_taubin(
            number_of_iterations=args.smooth_iterations
        ))
    faces_before_decimation = len(mesh.triangles)
    write_world_ply(args.out / "highres_m.ply", mesh, origin)
    if faces_before_decimation > args.faces:
        mesh = clean_mesh(mesh.simplify_quadric_decimation(
            target_number_of_triangles=args.faces
        ))
    print("Checking reduced mesh topology and self intersections...")
    report = geometry_report(mesh)
    write_world_ply(args.out / "reduced_m.ply", mesh, origin)

    rng = np.random.default_rng(0)
    points = np.asarray(pcd.points)
    indices = rng.choice(len(points), min(args.qa_samples, len(points)), replace=False)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = scene.compute_distance(o3d.core.Tensor(
        points[indices].astype(np.float32), dtype=o3d.core.Dtype.Float32
    )).numpy()
    o3d.utility.random.seed(0)
    mesh_samples = mesh.sample_points_uniformly(number_of_points=args.qa_samples)
    reverse_distances = mesh_samples.compute_point_cloud_distance(pcd)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    report.update({
        "open3d_version": o3d.__version__, "settings": settings,
        "input_points": input_count, "finite_unique_points": finite_unique_count,
        "voxel_points": voxel_count, "filtered_points": len(pcd.points),
        "meter_coordinate_origin": origin.tolist(), "extent_m": extent.tolist(),
        "median_point_spacing_mm": spacing * 1000,
        "finest_poisson_cube_cell_mm": cube_cell * 1000,
        "triangles_before_decimation": faces_before_decimation,
        "target_triangles": args.faces,
        "processed_observed_points_to_mesh": distance_stats(distances),
        "sampled_mesh_to_processed_observed_points": distance_stats(reverse_distances),
        "qa_note": "Distances use processed observations, not ground truth. Reverse "
                   "distances include point sampling gaps and inferred surfaces.",
        "manual_checks": ["Missing-region shape", "Limb/finger bridges", "Print size",
                          "Minimum thickness/strength", "Print orientation/supports"],
        "stl_written": False,
    })
    if report["geometry_gate_passed"]:
        print_mesh = copy.deepcopy(mesh)
        print_mesh.scale(1000.0 * args.print_scale, center=np.zeros(3))
        write_mesh(args.out / "print_centered_mm.stl", print_mesh)
        report["stl_written"] = True
        report["stl_extent_mm"] = (print_mesh.get_max_bound() - print_mesh.get_min_bound()).tolist()
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["stl_written"]:
        print("STL withheld: inspect reduced_m.ply/report.json and repair the failed checks.")
    else:
        print("STL passed geometric checks. Review anatomy and physical print constraints.")


if __name__ == "__main__":
    main()
