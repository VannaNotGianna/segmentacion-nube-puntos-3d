"""Coloreado, exportación y figuras de la segmentación."""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Patch

# Paleta por clase (RGB en [0, 1]); -1 = sin clasificar.
CLASS_COLORS = {
    -1: (0.75, 0.75, 0.75),  # sin clasificar: gris
    0: (0.10, 0.20, 0.85),   # piso: azul
    1: (0.55, 0.80, 0.95),   # techo: celeste
    2: (0.95, 0.85, 0.10),   # plataforma: amarillo
    3: (0.95, 0.55, 0.15),   # horizontal_otro: naranja
    4: (0.10, 0.65, 0.20),   # columna: verde
    5: (0.85, 0.10, 0.10),   # tubería: rojo
    6: (0.55, 0.20, 0.75),   # viga / estructura metálica: morado
    7: (0.95, 0.55, 0.15),   # maquinaria: naranja
    8: (0.75, 0.75, 0.75),   # otros: gris
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


# --------------------------------------------------------------------------- #
# Vistas 3D (capturas para el reporte)
# --------------------------------------------------------------------------- #
# Cámaras predefinidas: dirección "front" (desde la escena hacia la cámara),
# desplazamiento del punto mirado respecto al centro de la nube y zoom.
VIEWS = {
    "aerea": {"front": (-0.5, -0.75, 0.45), "offset": (0, 0, 0), "zoom": 0.55},
    "aerea_opuesta": {"front": (0.55, 0.7, 0.45), "offset": (0, 0, 0), "zoom": 0.55},
    "lateral": {"front": (0.0, -1.0, 0.15), "offset": (0, 0, 0), "zoom": 0.6},
    "interior_nave": {"front": (-0.3, -0.95, 0.1), "offset": (0, 4, -2), "zoom": 0.25},
    "planta": {"front": (0.0, 0.0, 1.0), "offset": (0, 0, 0), "zoom": 0.8, "up": (0, 1, 0)},
}


def render_view(points: np.ndarray, colors: np.ndarray, path, view: str = "aerea",
                width: int = 1600, height: int = 900, point_size: float = 2.0,
                hide_above: float | None = None) -> None:
    """Guarda una captura 3D de la nube con colores por punto.

    Usa una ventana oculta de Open3D (el renderizador offscreen devuelve una
    imagen negra en algunos equipos Windows). `hide_above` permite quitar los
    puntos por encima de cierta Z (p. ej. el techo) para ver el interior.
    """
    if hide_above is not None:
        keep = points[:, 2] <= hide_above
        points, colors = points[keep], colors[keep]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    cfg = VIEWS[view]
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    try:
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.ones(3)
        opt.light_on = False
        ctrl = vis.get_view_control()
        ctrl.set_front(cfg["front"])
        ctrl.set_up(cfg.get("up", (0, 0, 1)))
        ctrl.set_lookat(pcd.get_center() + np.asarray(cfg["offset"], dtype=float))
        ctrl.set_zoom(cfg["zoom"])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(path), do_render=True)
    finally:
        vis.destroy_window()


def plot_legend(class_names: dict, path=None, palette: dict = CLASS_COLORS, exclude=(-1, 3)):
    """Leyenda de colores por clase (para acompañar las capturas)."""
    handles = [Patch(color=palette[c], label=name.replace("_", " "))
               for c, name in class_names.items() if c not in exclude]
    fig, ax = plt.subplots(figsize=(2.6, 0.38 * len(handles) + 0.3))
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=11)
    ax.axis("off")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
