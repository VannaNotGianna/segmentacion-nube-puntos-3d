"""Segmentación multietapa de la nube de puntos con K-Means y DBSCAN.

Etapa 1 (estructura):
  1a. K-Means sobre |normal| -> orientación de cada punto
      (horizontal, vertical en X, vertical en Y).
  1b. DBSCAN sobre XYZ de los puntos horizontales -> superficies conexas.
  1c. Cada superficie se etiqueta por altura y tamaño:
      piso, techo, plataforma u otra superficie horizontal.
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from .preprocessing import PointCloudData

# Etiquetas de clase (comunes a todas las etapas)
UNLABELED = -1
FLOOR = 0
CEILING = 1
PLATFORM = 2
HORIZONTAL_OTHER = 3

CLASS_NAMES = {
    UNLABELED: "sin_clasificar",
    FLOOR: "piso",
    CEILING: "techo",
    PLATFORM: "plataforma",
    HORIZONTAL_OTHER: "horizontal_otro",
}

# Orientaciones (salida de la etapa 1a)
HORIZONTAL = 0
VERTICAL_X = 1
VERTICAL_Y = 2


# --------------------------------------------------------------------------- #
# Etapa 1a: orientación por normales
# --------------------------------------------------------------------------- #
def cluster_orientation(normals: np.ndarray, random_state: int = 0):
    """K-Means (k=3) sobre el valor absoluto de las normales.

    Se usa |n| porque el signo de la normal es arbitrario. Los centroides se
    reasignan a HORIZONTAL / VERTICAL_X / VERTICAL_Y según su componente
    dominante, de modo que el resultado no depende del orden de K-Means.

    Devuelve (orientación por punto, centroides ordenados [H, VX, VY]).
    """
    km = KMeans(n_clusters=3, n_init=10, random_state=random_state)
    raw = km.fit_predict(np.abs(normals))
    centers = km.cluster_centers_

    dominant = np.argmax(centers, axis=1)  # 0=x, 1=y, 2=z
    if sorted(dominant) != [0, 1, 2]:
        raise RuntimeError(f"Centroides de orientación ambiguos: {centers.round(2)}")
    axis_to_orientation = {2: HORIZONTAL, 0: VERTICAL_X, 1: VERTICAL_Y}
    mapping = np.array([axis_to_orientation[a] for a in dominant])

    ordered = np.empty_like(centers)
    ordered[mapping] = centers
    return mapping[raw], ordered


# --------------------------------------------------------------------------- #
# Etapa 1b: superficies horizontales conexas
# --------------------------------------------------------------------------- #
def cluster_surfaces(points: np.ndarray, eps: float = 0.15, min_samples: int = 8) -> np.ndarray:
    """DBSCAN sobre XYZ. Devuelve etiquetas de superficie (-1 = ruido)."""
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)


def describe_segments(points: np.ndarray, seg_labels: np.ndarray) -> list[dict]:
    """Estadísticas simples por segmento (ordenadas de mayor a menor)."""
    stats = []
    for s in np.unique(seg_labels):
        if s < 0:
            continue
        q = points[seg_labels == s]
        lo, hi = q.min(axis=0), q.max(axis=0)
        stats.append({
            "segment": int(s),
            "n": len(q),
            "z_median": float(np.median(q[:, 2])),
            "extent": hi - lo,
            "area_xy": float((hi[0] - lo[0]) * (hi[1] - lo[1])),
        })
    return sorted(stats, key=lambda d: -d["n"])


# --------------------------------------------------------------------------- #
# Etapa 1c: etiquetado de superficies por altura y tamaño
# --------------------------------------------------------------------------- #
def label_horizontal_surfaces(
    points: np.ndarray,
    seg_labels: np.ndarray,
    ceiling_band: float = 1.5,
    floor_band: float = 0.6,
    min_platform_area: float = 10.0,
    min_ceiling_points: int = 200,
) -> np.ndarray:
    """Asigna clase a cada punto horizontal según la superficie a la que pertenece.

    - Piso: la superficie más grande; además, toda superficie cuya mediana de Z
      quede a menos de `floor_band` m de la del piso.
    - Techo: superficies con mediana de Z dentro de los `ceiling_band` m
      superiores de la nube y al menos `min_ceiling_points` puntos.
    - Plataforma: superficies intermedias con área XY >= `min_platform_area` m².
    - Resto: horizontal_otro (mesas, tapas de máquinas, etc.).
    """
    stats = describe_segments(points, seg_labels)
    classes = np.full(len(points), HORIZONTAL_OTHER)
    if not stats:
        return classes

    floor_z = stats[0]["z_median"]
    top_z = np.percentile(points[:, 2], 99.5)

    for s in stats:
        mask = seg_labels == s["segment"]
        z = s["z_median"]
        if abs(z - floor_z) < floor_band:
            classes[mask] = FLOOR
        elif z > top_z - ceiling_band and s["n"] >= min_ceiling_points:
            classes[mask] = CEILING
        elif s["area_xy"] >= min_platform_area:
            classes[mask] = PLATFORM
    # Ruido de DBSCAN: se deja como horizontal_otro.
    return classes


def stage1_structure(
    data: PointCloudData,
    surface_eps: float = 0.15,
    surface_min_samples: int = 8,
    random_state: int = 0,
):
    """Ejecuta la etapa 1 completa.

    Devuelve un dict con:
      orientation: orientación por punto (HORIZONTAL / VERTICAL_X / VERTICAL_Y)
      classes:     clase estructural por punto (UNLABELED para no horizontales)
      surfaces:    id de superficie DBSCAN por punto (-1 si no es horizontal o ruido)
      centers:     centroides de orientación
    """
    orientation, centers = cluster_orientation(data.normals, random_state)
    horizontal = orientation == HORIZONTAL

    surfaces = np.full(len(data), -1)
    surfaces[horizontal] = cluster_surfaces(
        data.points[horizontal], eps=surface_eps, min_samples=surface_min_samples
    )

    classes = np.full(len(data), UNLABELED)
    classes[horizontal] = label_horizontal_surfaces(
        data.points[horizontal], surfaces[horizontal]
    )
    return {"orientation": orientation, "classes": classes,
            "surfaces": surfaces, "centers": centers}
