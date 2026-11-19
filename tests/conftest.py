"""Escena sintética con respuesta conocida para probar el pipeline.

Contiene piso, techo, dos columnas, una tubería (clara), una viga elevada
(verde) y una máquina (oscura) sobre el piso. Cada punto lleva su clase
verdadera para comparar con la salida de la segmentación.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import clustering as C  # noqa: E402
from src.preprocessing import PointCloudData  # noqa: E402

STEP = 0.08  # separación entre puntos (igual al vóxel del pipeline)
GRAY = (0.45, 0.45, 0.45)
LIGHT_RGB = (0.80, 0.82, 0.85)
GREEN_RGB = (0.38, 0.55, 0.30)
DARK_RGB = (0.18, 0.17, 0.25)


def _grid(a0, a1, b0, b1, step=STEP):
    a = np.arange(a0, a1 + 1e-9, step)
    b = np.arange(b0, b1 + 1e-9, step)
    A, B = np.meshgrid(a, b, indexing="ij")
    return A.ravel(), B.ravel()


def plane_z(x0, x1, y0, y1, z):
    x, y = _grid(x0, x1, y0, y1)
    pts = np.c_[x, y, np.full_like(x, z)]
    return pts, np.tile([0.0, 0.0, 1.0], (len(pts), 1))


def box(x0, x1, y0, y1, z0, z1, top=True):
    """Caras laterales (y opcionalmente la tapa) de una caja alineada a los ejes."""
    parts = []
    for x in (x0, x1):
        y, z = _grid(y0, y1, z0, z1)
        parts.append((np.c_[np.full_like(y, x), y, z], [1.0, 0.0, 0.0]))
    for y in (y0, y1):
        x, z = _grid(x0, x1, z0, z1)
        parts.append((np.c_[x, np.full_like(x, y), z], [0.0, 1.0, 0.0]))
    if top:
        x, y = _grid(x0, x1, y0, y1)
        parts.append((np.c_[x, y, np.full_like(x, z1)], [0.0, 0.0, 1.0]))
    pts = np.vstack([p for p, _ in parts])
    nrm = np.vstack([np.tile(n, (len(p), 1)) for p, n in parts])
    return pts, nrm


def cylinder_x(x0, x1, y, z, radius):
    """Tubería horizontal a lo largo de X."""
    n_around = max(8, int(2 * np.pi * radius / STEP))
    ang = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    xs = np.arange(x0, x1 + 1e-9, STEP)
    X, A = np.meshgrid(xs, ang, indexing="ij")
    X, A = X.ravel(), A.ravel()
    nrm = np.c_[np.zeros_like(A), np.cos(A), np.sin(A)]
    pts = np.c_[X, y + radius * np.cos(A), z + radius * np.sin(A)]
    return pts, nrm


def build_scene(seed=0):
    rng = np.random.default_rng(seed)
    parts = []  # (puntos, normales, rgb, clase verdadera)

    def add(geom, rgb, cls):
        pts, nrm = geom
        parts.append((pts, nrm, rgb, cls))

    add(plane_z(0, 12, 0, 12, 0.0), GRAY, C.FLOOR)
    add(plane_z(0, 12, 0, 12, 8.0), GRAY, C.CEILING)
    add(box(2.0, 2.6, 2.0, 2.6, 0.1, 7.9, top=False), GRAY, C.COLUMN)
    add(box(9.0, 9.6, 2.0, 2.6, 0.1, 7.9, top=False), GRAY, C.COLUMN)
    add(cylinder_x(3.5, 8.0, 7.0, 4.0, 0.15), LIGHT_RGB, C.PIPE)
    add(box(3.5, 8.0, 9.5, 9.8, 5.5, 5.9), GREEN_RGB, C.BEAM)
    add(box(5.0, 6.0, 4.0, 5.0, 0.1, 1.2), DARK_RGB, C.MACHINERY)

    points = np.vstack([p[0] for p in parts])
    normals = np.vstack([p[1] for p in parts])
    colors = np.vstack([np.tile(p[2], (len(p[0]), 1)) for p in parts])
    truth = np.concatenate([np.full(len(p[0]), p[3]) for p in parts])

    # Ruido realista: posición (~1 cm) y color (~0.03)
    points = points + rng.normal(0, 0.01, points.shape)
    colors = np.clip(colors + rng.normal(0, 0.03, colors.shape), 0, 1)
    return PointCloudData(points, colors, normals), truth


@pytest.fixture(scope="session")
def scene():
    return build_scene()
