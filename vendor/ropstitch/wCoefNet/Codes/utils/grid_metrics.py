import numpy as np
import torch


def compute_mesh_triangles(mesh):
    """
    Given mesh of shape (H+1, W+1, 2) in numpy or torch,
    decompose each quad into 2 triangles.
    Returns:
        v0, v1, v2: arrays of shape (H, W, 2, 2) where:
            tri 0: (v00, v10, v11)
            tri 1: (v00, v11, v01)
    """
    if isinstance(mesh, torch.Tensor):
        mesh_np = mesh.detach().cpu().numpy()
    else:
        mesh_np = np.asarray(mesh)

    H_cells = mesh_np.shape[0] - 1
    W_cells = mesh_np.shape[1] - 1

    v00 = mesh_np[:-1, :-1]  # (H, W, 2)
    v10 = mesh_np[1:, :-1]   # (H, W, 2)
    v11 = mesh_np[1:, 1:]    # (H, W, 2)
    v01 = mesh_np[:-1, 1:]   # (H, W, 2)

    # Tri 1: v00 -> v10 -> v11
    # Tri 2: v00 -> v11 -> v01
    tri1_v0, tri1_v1, tri1_v2 = v00, v10, v11
    tri2_v0, tri2_v1, tri2_v2 = v00, v11, v01

    return (tri1_v0, tri1_v1, tri1_v2), (tri2_v0, tri2_v1, tri2_v2)


def compute_triangle_signed_area(v0, v1, v2):
    """
    Compute signed area: 0.5 * ((x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0))
    """
    e1 = v1 - v0
    e2 = v2 - v0
    signed_area = 0.5 * (e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0])
    return signed_area


def compute_grid_metrics(mesh, rigid_mesh=None, eps=1e-6):
    """
    Compute geometric quality metrics on a 2D mesh grid of shape (H+1, W+1, 2).

    Metrics:
    - folding_ratio: fraction of triangles whose signed area <= 0 (or opposite to rigid orientation)
    - degenerate_ratio: fraction of quad cells whose area is below threshold (e.g. < 1e-4 of rigid area)
    - avg_stretch: average distortion ratio of triangle edges relative to rigid
    - max_stretch: maximum distortion ratio
    - scale_variation: std deviation of log-scale (local area change relative to rigid)
    - bending_degree: normalized second-order discrete curvature along grid lines
    """
    if isinstance(mesh, torch.Tensor):
        if mesh.dim() == 4:
            mesh = mesh[0]  # take first batch
        mesh = mesh.detach().cpu().numpy()

    if rigid_mesh is not None:
        if isinstance(rigid_mesh, torch.Tensor):
            if rigid_mesh.dim() == 4:
                rigid_mesh = rigid_mesh[0]
            rigid_mesh = rigid_mesh.detach().cpu().numpy()
    else:
        H_cells = mesh.shape[0] - 1
        W_cells = mesh.shape[1] - 1
        y = np.linspace(0.0, float(H_cells), H_cells + 1)
        x = np.linspace(0.0, float(W_cells), W_cells + 1)
        xx, yy = np.meshgrid(x, y)
        rigid_mesh = np.stack([xx, yy], axis=-1)

    t1, t2 = compute_mesh_triangles(mesh)
    r1, r2 = compute_mesh_triangles(rigid_mesh)

    area_t1 = compute_triangle_signed_area(*t1)
    area_t2 = compute_triangle_signed_area(*t2)
    area_r1 = compute_triangle_signed_area(*r1)
    area_r2 = compute_triangle_signed_area(*r2)

    # 1. Folding ratio
    # Orientation check: if area has opposite sign to rigid or is <= 0
    sign_r1 = np.sign(area_r1 + eps)
    sign_r2 = np.sign(area_r2 + eps)
    flipped_t1 = (area_t1 * sign_r1) <= eps
    flipped_t2 = (area_t2 * sign_r2) <= eps
    total_triangles = area_t1.size + area_t2.size
    folding_count = int(np.sum(flipped_t1) + np.sum(flipped_t2))
    folding_ratio = float(folding_count / total_triangles)

    # 2. Quad cell areas & Nearly degenerate cells ratio
    quad_area = np.abs(area_t1) + np.abs(area_t2)
    quad_rigid_area = np.abs(area_r1) + np.abs(area_r2)
    degenerate_mask = (quad_area / (quad_rigid_area + eps)) < 1e-3
    degenerate_ratio = float(np.mean(degenerate_mask))

    # 3. Stretch (distortion of edge lengths compared to rigid)
    # Compare triangle edge lengths:
    def tri_edges(v0, v1, v2):
        l0 = np.linalg.norm(v1 - v0, axis=-1)
        l1 = np.linalg.norm(v2 - v1, axis=-1)
        l2 = np.linalg.norm(v0 - v2, axis=-1)
        return l0, l1, l2

    l1_0, l1_1, l1_2 = tri_edges(*t1)
    l2_0, l2_1, l2_2 = tri_edges(*t2)
    r1_0, r1_1, r1_2 = tri_edges(*r1)
    r2_0, r2_1, r2_2 = tri_edges(*r2)

    s1 = np.concatenate([l1_0 / (r1_0 + eps), l1_1 / (r1_1 + eps), l1_2 / (r1_2 + eps)], axis=None)
    s2 = np.concatenate([l2_0 / (r2_0 + eps), l2_1 / (r2_1 + eps), l2_2 / (r2_2 + eps)], axis=None)
    all_edge_scales = np.concatenate([s1, s2])

    # Normalized distortion: ratio between max and min scale or deviation from mean scale
    avg_scale = np.mean(all_edge_scales) + eps
    normalized_stretch = np.maximum(all_edge_scales / avg_scale, avg_scale / (all_edge_scales + eps))
    avg_stretch = float(np.mean(normalized_stretch) - 1.0)  # 0 for rigid/isotropic
    max_stretch = float(np.max(normalized_stretch) - 1.0)

    # 4. Local scale variation (std of log scale)
    relative_quad_scale = quad_area / (quad_rigid_area + eps)
    log_scale = np.log(np.maximum(relative_quad_scale, eps))
    scale_variation = float(np.std(log_scale))

    # 5. Bending degree (second-order difference / discrete Laplacian along grid lines)
    # Horizontal lines: mesh[i, j+1] - 2*mesh[i, j] + mesh[i, j-1]
    d2_h = mesh[:, 2:] - 2 * mesh[:, 1:-1] + mesh[:, :-2]
    # Vertical lines: mesh[i+1, j] - 2*mesh[i, j] + mesh[i-1, j]
    d2_v = mesh[2:, :] - 2 * mesh[1:-1, :] + mesh[:-2, :]

    # Normalize by average local grid step to be scale-invariant
    d1_h = np.linalg.norm(mesh[:, 1:] - mesh[:, :-1], axis=-1)
    d1_v = np.linalg.norm(mesh[1:, :] - mesh[:-1, :], axis=-1)
    mean_step = 0.5 * (np.mean(d1_h) + np.mean(d1_v)) + eps

    bend_h = np.linalg.norm(d2_h, axis=-1) / mean_step
    bend_v = np.linalg.norm(d2_v, axis=-1) / mean_step
    bending_degree = float(0.5 * (np.mean(bend_h) + np.mean(bend_v)))

    return {
        "folding_ratio": folding_ratio,
        "degenerate_ratio": degenerate_ratio,
        "avg_stretch": avg_stretch,
        "max_stretch": max_stretch,
        "scale_variation": scale_variation,
        "bending_degree": bending_degree,
    }
