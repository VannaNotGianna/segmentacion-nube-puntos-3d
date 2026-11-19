"""Coloreado, exportación y figuras de la segmentación."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Paleta por clase (RGB en [0, 1]); -1 = sin clasificar.
CLASS_COLORS = {
    -1: (0.75, 0.75, 0.75),  # sin clasificar: gris
    0: (0.10, 0.20, 0.85),   # piso: azul
    1: (0.55, 0.80, 0.95),   # techo: celeste
    2: (0.95, 0.85, 0.10),   # plataforma: amarillo
    3: (0.95, 0.55, 0.15),   # horizontal_otro: naranja
}


def colors_from_classes(classes: np.ndarray, palette: dict = CLASS_COLORS) -> np.ndarray:
    default = palette.get(-1, (0.5, 0.5, 0.5))
    return np.array([palette.get(int(c), default) for c in classes])


def colors_from_segments(seg_labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Color aleatorio por segmento (útil para revisar DBSCAN); ruido en gris."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seg_labels)
    table = {s: rng.uniform(0.15, 0.95, 3) for s in uniq if s >= 0}
    return np.array([table.get(s, (0.75, 0.75, 0.75)) for s in seg_labels])


def save_colored_ply(path, points: np.ndarray, colors: np.ndarray, normals=None) -> None:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if not o3d.io.write_point_cloud(str(path), pcd):
        raise IOError(f"No se pudo escribir {path}")


def plot_projections(points: np.ndarray, colors: np.ndarray, path=None,
                     title: str = "", max_points: int = 200_000, seed: int = 0):
    """Vista superior (XY) y laterales (XZ, YZ) con colores por punto."""
    if len(points) > max_points:
        idx = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]

    views = [(0, 1, "Planta (X-Y)"), (0, 2, "Lateral (X-Z)"), (1, 2, "Lateral (Y-Z)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (i, j, name) in zip(axes, views):
        # Vista superior: dibujar de abajo hacia arriba; laterales: por profundidad.
        depth = points[:, 2] if (i, j) == (0, 1) else points[:, 3 - i - j]
        order = np.argsort(depth)
        ax.scatter(points[order, i], points[order, j], c=colors[order], s=0.3, linewidths=0)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("XYZ"[i])
        ax.set_ylabel("XYZ"[j])
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig
