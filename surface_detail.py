"""Traditional robust local-plane fitting. All distances are in meters.

No point insertion or learned prior. Queries and covariance matrices are batched
so memory scales with batch_size * neighbors, not the square of cloud size.
"""

import numpy as np
from scipy.spatial import cKDTree


def fit_local_planes(points, neighbors, radius, workers, batch_size=8192):
    """Return robust PCA normals, signed plane offsets, support and flatness.

    Spatial Gaussian weights are followed by one Cauchy residual reweighting.
    A radius cap avoids expanding the neighborhood across large empty gaps.
    At least six samples and a non-collinear neighborhood are required. Normal
    signs are arbitrary here; orient them consistently before Poisson.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
        raise ValueError("Expected at least six finite 3D points")
    if not np.isfinite(points).all():
        raise ValueError("Local-plane input must be finite")
    if neighbors < 6 or radius <= 0 or not np.isfinite(radius):
        raise ValueError("neighbors >= 6 and finite radius > 0 required")
    tree = cKDTree(points)
    normals = np.empty_like(points)
    offsets = np.empty(len(points))
    valid = np.zeros(len(points), dtype=bool)
    flat = np.zeros(len(points), dtype=bool)
    supports = np.empty(len(points), dtype=np.int32)
    k = min(neighbors, len(points))

    def pca(delta, weights):
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
        center = np.einsum("nk,nkj->nj", weights, delta)
        centered = delta - center[:, None, :]
        cov = np.einsum("nk,nki,nkj->nij", weights, centered, centered)
        values, vectors = np.linalg.eigh(cov)
        return center, values, vectors[:, :, 0]

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        query = points[start:stop]
        distances, indices = tree.query(query, k=k, workers=workers)
        delta = points[indices] - query[:, None, :]
        inside = distances <= radius
        count = inside.sum(axis=1)
        bandwidth = np.maximum(np.max(np.where(inside, distances, 0), axis=1), 1e-6)
        spatial = np.exp(-0.5 * (distances / bandwidth[:, None]) ** 2) * inside
        center, _, normal = pca(delta, spatial)
        residual = np.einsum("nkj,nj->nk", delta - center[:, None, :], normal)
        # nanmedian ignores samples beyond the physical radius cap.
        masked = np.where(inside, residual, np.nan)
        median = np.nanmedian(masked, axis=1)
        mad = np.nanmedian(np.abs(masked - median[:, None]), axis=1)
        sigma = np.maximum(1.4826 * mad, 0.00015)
        robust = 1.0 / (1.0 + (residual / (2 * sigma[:, None])) ** 2)
        center, values, normal = pca(delta, spatial * robust)
        supported = (count >= 6) & (values[:, 1] > 1e-12)
        normals[start:stop] = normal
        offsets[start:stop] = np.einsum("nj,nj->n", center, normal)
        valid[start:stop] = supported
        # Skip noisy/non-planar neighborhoods for position filtering. This is a
        # conservative feature guard, not a semantic nose/eye/edge detector.
        flat[start:stop] = supported & (values[:, 0] / np.maximum(values[:, 1], 1e-30) < 0.1)
        supports[start:stop] = count
    return normals, offsets, valid, flat, supports


def denoise_local_planes(points, neighbors, radius, workers, iterations,
                         max_move=0.0005):
    """Damped projection along local normals, with a TOTAL displacement cap.

    Freeze one set of points per iteration, rebuild neighbors between iterations,
    leave unsupported/non-planar points untouched, and cap accumulated movement
    from the original observation. RGB and point count are unchanged.
    """
    original = np.asarray(points, dtype=np.float64).copy()
    current = original.copy()
    per_iteration = []
    for _ in range(iterations):
        normal, offset, valid, flat, _ = fit_local_planes(
            current, neighbors, radius, workers)
        step = 0.5 * np.clip(offset, -max_move, max_move)
        step[~flat] = 0
        candidate = current + step[:, None] * normal
        displacement = candidate - original
        length = np.linalg.norm(displacement, axis=1)
        scale = np.minimum(1.0, max_move / np.maximum(length, 1e-30))
        current = original + displacement * scale[:, None]
        per_iteration.append({"supported_points": int(valid.sum()),
                              "eligible_flat_points": int(flat.sum())})
    movement = np.linalg.norm(current - original, axis=1) * 1000
    return current, {"method": "robust local-plane damped projection",
                     "iterations": iterations, "total_displacement_limit_mm": max_move * 1000,
                     "moved_points": int((movement > 1e-9).sum()),
                     "movement_median_mm": float(np.median(movement)),
                     "movement_p95_mm": float(np.quantile(movement, 0.95)),
                     "movement_max_mm": float(np.max(movement)),
                     "per_iteration": per_iteration}
