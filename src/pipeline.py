"""Ejecución del pipeline de segmentación.

Uso (desde la raíz del proyecto):
    python -m src.pipeline
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .clustering import (CLASS_NAMES, MATERIAL_NAMES, OTHER, stage1_structure,
                         stage2_objects, transfer_labels)
from .preprocessing import load_point_cloud, remove_outliers, to_arrays, voxel_downsample
from .visualization import colors_from_classes, colors_from_segments, plot_projections, save_colored_ply

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data/raw/FARO_Scan_Full_Sub1_down3_with_normals.ply"


def _display_path(path) -> str:
    """Ruta relativa al proyecto si es posible (no expone rutas personales)."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def prepare_data(path=DEFAULT_INPUT, voxel: float = 0.08, verbose: bool = True,
                 return_full: bool = False):
    """Carga, submuestrea y filtra la nube.

    Devuelve PointCloudData submuestreado; con `return_full=True` devuelve
    además la nube original completa: (submuestreada, original).
    """
    if verbose:
        print("Cargando", _display_path(path))
    pcd = load_point_cloud(path)
    full = to_arrays(pcd) if return_full else None
    n_orig = len(pcd.points)
    pcd = voxel_downsample(pcd, voxel)
    pcd, _ = remove_outliers(pcd)
    data = to_arrays(pcd)
    if verbose:
        print(f"  puntos originales: {n_orig:,}")
        print(f"  tras vóxel {voxel} m y filtro de outliers: {len(data):,}")
    return (data, full) if return_full else data


def main():
    matplotlib.use("Agg")  # sin ventanas: solo se guardan figuras
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # tildes en la consola de Windows
    parser = argparse.ArgumentParser(description="Segmentación de nube de puntos 3D")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--voxel", type=float, default=0.08, help="tamaño de vóxel (m)")
    parser.add_argument("--surface-eps", type=float, default=0.15, help="eps DBSCAN superficies (m)")
    args = parser.parse_args()

    fig_dir = ROOT / "results/figures"
    pc_dir = ROOT / "results/point_clouds"
    label_dir = ROOT / "results/labels"
    metric_dir = ROOT / "results/metricas"
    for d in (fig_dir, pc_dir, label_dir, metric_dir):
        d.mkdir(parents=True, exist_ok=True)

    data, full = prepare_data(args.input, args.voxel, return_full=True)

    print("Etapa 1: estructura")
    s1 = stage1_structure(data, surface_eps=args.surface_eps)
    print("  centroides de orientación |n| [H, VX, VY]:\n", s1["centers"].round(2))
    for c, n in zip(*np.unique(s1["classes"], return_counts=True)):
        print(f"  {CLASS_NAMES[c]:<16} {n:>8,}")

    colors = colors_from_classes(s1["classes"])
    save_colored_ply(pc_dir / "etapa1_estructura.ply", data.points, colors, data.normals)
    np.save(label_dir / "etapa1_clases.npy", s1["classes"])
    plot_projections(data.points, data.colors, fig_dir / "00_original_rgb.png", "Nube original (RGB)")
    orient_rgb = np.array([(0.9, 0.2, 0.2), (0.2, 0.7, 0.2), (0.2, 0.3, 0.9)])
    plot_projections(data.points, orient_rgb[s1["orientation"]], fig_dir / "01_orientacion_normales.png",
                     "Etapa 1a: orientación (rojo=horizontal, verde=vertical X, azul=vertical Y)")
    plot_projections(data.points, colors, fig_dir / "02_estructura.png",
                     "Etapa 1: piso (azul), techo (celeste), plataforma (amarillo), "
                     "horizontal otro (naranja), sin clasificar (gris)")

    print("Etapa 2: objetos")
    s2 = stage2_objects(data, s1)
    print(f"  piso de referencia z = {s2['floor_z']:.2f} m")
    print("  clusters de color (RGB medio -> material):")
    for rgb, mat in s2["colors"]:
        print(f"    {np.round(rgb, 2)} -> {MATERIAL_NAMES[mat]}")
    print(f"  objetos DBSCAN: {s2['objects'].max() + 1}")
    for c, n in zip(*np.unique(s2["classes"], return_counts=True)):
        print(f"  {CLASS_NAMES[c]:<16} {n:>8,}")

    final_colors = colors_from_classes(s2["classes"])
    save_colored_ply(pc_dir / "segmentacion_final.ply", data.points, final_colors, data.normals)
    np.save(label_dir / "clases_final.npy", s2["classes"])
    mat_rgb = np.array([(0.85, 0.85, 0.85), (0.2, 0.7, 0.2), (0.7, 0.3, 0.1), (0.15, 0.15, 0.4)])
    mat_colors = np.where(s2["materials"][:, None] >= 0, mat_rgb[np.maximum(s2["materials"], 0)], 1.0)
    plot_projections(data.points, mat_colors, fig_dir / "03_materiales_color.png",
                     "Etapa 2b: materiales por color (claro, verde, óxido, oscuro)")
    plot_projections(data.points, colors_from_segments(s2["objects"]), fig_dir / "04_objetos_dbscan.png",
                     "Etapa 2c: objetos DBSCAN por material")
    plot_projections(data.points, final_colors, fig_dir / "05_segmentacion_final.png",
                     "Final: piso (azul), techo (celeste), plataforma (amarillo), columna (verde), "
                     "tubería (rojo), viga/estructura (morado), maquinaria (naranja), otros (gris)")

    print("Paso 4: transferencia a la nube original")
    max_dist = 2 * args.voxel
    full_classes, dist = transfer_labels(full.points, data.points, s2["classes"], max_dist=max_dist)
    n_far = int((dist > max_dist).sum())
    print(f"  {len(full):,} puntos; {n_far:,} a más de {max_dist:.2f} m -> {CLASS_NAMES[OTHER]}")
    rows = []
    for c, n in zip(*np.unique(full_classes, return_counts=True)):
        print(f"  {CLASS_NAMES[c]:<16} {n:>9,}  ({n / len(full):6.1%})")
        rows.append({"clase": CLASS_NAMES[c], "puntos": int(n), "porcentaje": n / len(full)})
    pd.DataFrame(rows).to_csv(metric_dir / "conteo_clases_nube_completa.csv", index=False)

    out_ply = pc_dir / "segmentacion_final_completa.ply"
    save_colored_ply(out_ply, full.points, colors_from_classes(full_classes), full.normals)
    np.save(label_dir / "clases_final_completa.npy", full_classes)
    print(f"  PLY entregable: {_display_path(out_ply)}")
    print("Resultados guardados en results/")


if __name__ == "__main__":
    main()
