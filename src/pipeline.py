"""Ejecución del pipeline de segmentación.

Uso (desde la raíz del proyecto):
    python -m src.pipeline
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .clustering import CLASS_NAMES, stage1_structure
from .preprocessing import load_point_cloud, remove_outliers, to_arrays, voxel_downsample
from .visualization import colors_from_classes, plot_projections, save_colored_ply

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data/raw/FARO_Scan_Full_Sub1_down3_with_normals.ply"


def main():
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
    for d in (fig_dir, pc_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Cargando", args.input)
    pcd = load_point_cloud(args.input)
    print(f"  puntos originales: {len(pcd.points):,}")

    pcd = voxel_downsample(pcd, args.voxel)
    pcd, _ = remove_outliers(pcd)
    data = to_arrays(pcd)
    print(f"  tras vóxel {args.voxel} m y filtro de outliers: {len(data):,}")

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
    print("Resultados guardados en results/")


if __name__ == "__main__":
    main()
