"""Interactive 3D comparison by default; --2d exports fixed-view comparison images.

Python 3.9; Open3D 0.19.0; NumPy 1.23.5; Matplotlib.
No reconstruction, component deletion, color correction, or image enhancement.
Input and displayed coordinates are millimeters; processing uses meters.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_work" / "denoise_preview" / "mpl"))
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import open3d as o3d

from output_paths import latest_report


INPUT = ROOT / "export_00000.ply"
REPORT = latest_report(ROOT)
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")


def export_2d():
    settings = {"voxel": 0.003, "outlier_neighbors": 20, "outlier_std": 2.5}
    if REPORT is not None:
        settings.update(json.loads(REPORT.read_text(encoding="utf-8")).get("settings", {}))
        print(f"Parameters from report: {REPORT}", flush=True)
    voxel = settings["voxel"]
    neighbors = settings["outlier_neighbors"]
    std = settings["outlier_std"]
    started = time.perf_counter()
    raw = o3d.io.read_point_cloud(str(INPUT))
    raw_count = len(raw.points)
    raw.remove_non_finite_points()
    raw.remove_duplicated_points()
    raw.scale(0.001, center=np.zeros(3))
    down = raw.voxel_down_sample(voxel)
    kept, indices = down.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std)
    mask = np.ones(len(down.points), dtype=bool)
    mask[np.asarray(indices, dtype=np.int64)] = False
    removed = down.select_by_index(indices, invert=True)
    filter_time = time.perf_counter() - started
    stats = {"input": str(INPUT), "open3d": o3d.__version__,
             "input_unit": "mm", "display_unit": "mm", "voxel_mm": voxel * 1000,
             "sor_neighbors": neighbors, "sor_std_ratio": std,
             "original_points": raw_count, "finite_unique_points": len(raw.points),
             "voxel_points": len(down.points), "retained_points": len(kept.points),
             "sor_removed_points": len(removed.points),
             "sor_removed_fraction_of_voxel_points": len(removed.points) / len(down.points),
             "preprocessing_seconds": round(filter_time, 3),
             "display": "All points, fixed orthographic view along Z; X vertical, Y horizontal.",
             "note": "Voxel reduction is resampling, not a count of removed noise. SOR flags are not noise ground truth. Dense background fragments may remain."}
    arrays = [np.asarray(p.points) * 1000 for p in [raw, down, kept]]
    all_xyz = arrays[0]
    center = (all_xyz.min(axis=0) + all_xyz.max(axis=0)) / 2
    # The scan's X axis follows the body length. This changes the view only.
    xlim = (all_xyz[:, 1].min() - center[1] - 50, all_xyz[:, 1].max() - center[1] + 50)
    ylim = (all_xyz[:, 0].min() - center[0] - 60, all_xyz[:, 0].max() - center[0] + 60)

    def setup(ax):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#f4f5f7")
        ax.set_axis_off()

    def draw_rgb(ax, cloud, size=0.2):
        xyz = np.asarray(cloud.points) * 1000 - center
        colors = np.asarray(cloud.colors)
        order = np.argsort(xyz[:, 2])[::-1]
        ax.scatter(xyz[order, 1], xyz[order, 0], s=size, c=colors[order],
                   edgecolors="none", linewidths=0, rasterized=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 8.5), facecolor="white")
    titles = [("原始彩色点云", len(raw.points)),
              (f"{voxel * 1000:g} mm 体素降采样", len(down.points)),
              ("统计离群点去噪后", len(kept.points))]
    for ax, cloud, (title, count) in zip(axes, [raw, down, kept], titles):
        setup(ax)
        draw_rgb(ax, cloud)
        ax.set_title(f"{title}\n{count:,} 点", fontproperties=FONT, fontsize=16, pad=10)
    fig.suptitle("点云处理前后对比 · 同视角、同尺度、保留原始颜色", fontproperties=FONT,
                 fontsize=20, y=0.975)
    removed_pct = len(removed.points) / len(down.points) * 100
    fig.text(0.5, 0.045, f"SOR：邻居数 {neighbors}，std_ratio={std:g}；剔除 {len(removed.points):,} 点（降采样后点数的 {removed_pct:.2f}%）",
             ha="center", fontproperties=FONT, fontsize=13, color="#263143")
    fig.text(0.5, 0.015, "降采样主要减少点密度；不能把全部减少的点数解释为被删除的噪声。",
             ha="center", fontproperties=FONT, fontsize=12, color="#667080")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.09, wspace=0.08)
    fig.savefig(ROOT / "denoise_comparison.png", dpi=160, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), facecolor="white")
    down_xyz = arrays[1] - center
    removed_xyz = np.asarray(removed.points) * 1000 - center
    for ax in axes:
        setup(ax)
    axes[0].scatter(down_xyz[:, 1], down_xyz[:, 0], s=0.18, c="#bac1ca",
                    edgecolors="none", linewidths=0, rasterized=True)
    axes[0].scatter(removed_xyz[:, 1], removed_xyz[:, 0], s=6, c="#e33535",
                    edgecolors="none", linewidths=0, rasterized=True)
    axes[0].set_title("位置对照：灰色为降采样点云\n红色为被 SOR 剔除的点", fontproperties=FONT, fontsize=16)
    axes[1].scatter(removed_xyz[:, 1], removed_xyz[:, 0], s=6, c="#e33535",
                    edgecolors="none", linewidths=0, rasterized=True)
    axes[1].set_title(f"单独显示被剔除的 {len(removed.points):,} 个点", fontproperties=FONT, fontsize=16)
    fig.suptitle("统计离群点的空间分布", fontproperties=FONT, fontsize=21, y=0.97)
    fig.text(0.5, 0.035, "红点是统计离群判定，不等于噪声真值；密集浮片不会自动被 SOR 全部删除。",
             ha="center", fontproperties=FONT, fontsize=12, color="#667080")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.08, wspace=0.08)
    fig.savefig(ROOT / "denoise_removed_points.png", dpi=160, facecolor="white")
    plt.close(fig)
    (ROOT / "denoise_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Comparison:", ROOT / "denoise_comparison.png")
    print("Removed points:", ROOT / "denoise_removed_points.png")


def main():
    import sys
    if "--2d" in sys.argv[1:]:
        if len(sys.argv) != 2:
            raise SystemExit("Use --2d alone to export using the reconstruction report settings.")
        export_2d()
    else:
        from denoise_compare_3d import main as run_3d
        run_3d()


if __name__ == "__main__":
    main()
