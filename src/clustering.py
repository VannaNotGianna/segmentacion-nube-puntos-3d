"""Segmentación multietapa de la nube de puntos con K-Means y DBSCAN.

Etapa 1 (estructura):
  1a. K-Means sobre |normal| -> orientación de cada punto
      (horizontal, vertical en X, vertical en Y).
  1b. DBSCAN sobre XYZ de los puntos horizontales -> superficies conexas.
  1c. Cada superficie se etiqueta por altura y tamaño:
      piso, techo, plataforma u otra superficie horizontal.

Etapa 2 (objetos), sobre los puntos no asignados en la etapa 1:
  2a. Columnas: mapa de ocupación vertical en planta + DBSCAN 2D (XY).
  2b. Materiales: K-Means sobre el color (tono, saturación, brillo).
  2c. Objetos: DBSCAN sobre XYZ dentro de cada grupo de material.
  2d. Clase por objeto según material y altura sobre el piso.
  2e. Suavizado: voto mayoritario entre vecinos cercanos.
"""

import warnings

import numpy as np
from matplotlib.colors import rgb_to_hsv
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from .preprocessing import PointCloudData

# Etiquetas de clase (comunes a todas las etapas)
UNLABELED = -1
FLOOR = 0
CEILING = 1
PLATFORM = 2
HORIZONTAL_OTHER = 3  # solo intermedio: la etapa 2 lo reclasifica
COLUMN = 4
PIPE = 5
BEAM = 6
MACHINERY = 7
OTHER = 8

CLASS_NAMES = {
    UNLABELED: "sin_clasificar",
    FLOOR: "piso",
    CEILING: "techo",
    PLATFORM: "plataforma",
    HORIZONTAL_OTHER: "horizontal_otro",
    COLUMN: "columna",
    PIPE: "tuberia",
    BEAM: "viga_estructura",
    MACHINERY: "maquinaria",
    OTHER: "otros",
}

# Materiales (etapa 2b)
LIGHT, GREEN, RUST, DARK = 0, 1, 2, 3
MATERIAL_NAMES = {LIGHT: "claro", GREEN: "verde", RUST: "oxido", DARK: "oscuro"}

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
def cluster_surfaces(points: np.ndarray, eps: float = 0.15, min_samples: int = 8,
                     max_noise: float = 0.5) -> np.ndarray:
    """DBSCAN sobre XYZ. Devuelve etiquetas de superficie (-1 = ruido).

    Si más de `max_noise` de los puntos queda como ruido, `eps` es demasiado
    pequeño para la densidad de la nube (p. ej. eps ≈ tamaño de vóxel) y las
    reglas posteriores (el piso es la superficie más grande) dejan de valer.
    """
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)
    noise = (labels < 0).mean()
    if noise > max_noise:
        warnings.warn(
            f"DBSCAN de superficies: {noise:.0%} de los puntos es ruido con eps={eps}. "
            "Use un eps de al menos ~2 veces el tamaño de vóxel.", RuntimeWarning)
    return labels


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


# --------------------------------------------------------------------------- #
# Etapa 2a: columnas
# --------------------------------------------------------------------------- #
def vertical_coverage(points: np.ndarray, floor_z: float,
                      cell: float = 0.25, z_bin: float = 0.5) -> np.ndarray:
    """Altura ocupada (m) en la celda XY de cada punto.

    Se divide la planta en celdas de `cell` m y la altura en franjas de
    `z_bin` m; la cobertura de una celda es el número de franjas con al menos
    un punto, multiplicado por `z_bin`. Las columnas ocupan casi toda la altura.
    """
    ij = np.floor(points[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    k = np.floor((points[:, 2] - floor_z) / z_bin).astype(np.int64)
    pairs = np.unique(np.c_[key, k], axis=0)
    cell_keys, n_bins = np.unique(pairs[:, 0], return_counts=True)
    return n_bins[np.searchsorted(cell_keys, key)] * z_bin


def detect_columns(points: np.ndarray, floor_z: float, min_height: float = 5.0,
                   eps: float = 0.35, min_samples: int = 20,
                   max_footprint: float = 2.5) -> np.ndarray:
    """Máscara de puntos pertenecientes a columnas.

    1. Se conservan los puntos cuya celda XY cubre al menos `min_height` m.
    2. DBSCAN en 2D (solo XY) agrupa esas celdas en candidatos.
    3. Un candidato es columna si su huella XY es menor a `max_footprint` m
       por lado y su altura supera `min_height` m.
    """
    tall = vertical_coverage(points, floor_z) >= min_height
    idx = np.flatnonzero(tall)
    is_column = np.zeros(len(points), dtype=bool)
    if len(idx) == 0:
        return is_column
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points[idx, :2])
    for s in np.unique(labels[labels >= 0]):
        members = idx[labels == s]
        ext = np.ptp(points[members], axis=0)
        if ext[0] < max_footprint and ext[1] < max_footprint and ext[2] > min_height:
            is_column[members] = True
    return is_column


# --------------------------------------------------------------------------- #
# Etapa 2b: materiales por color
# --------------------------------------------------------------------------- #
def color_features(colors: np.ndarray) -> np.ndarray:
    """Tono en el círculo cromático (ponderado por saturación) + brillo.

    El tono es un ángulo, por eso se representa con (cos, sin); se multiplica
    por la saturación para que los grises no tengan un tono arbitrario.
    """
    hsv = rgb_to_hsv(np.clip(colors, 0, 1))
    angle = hsv[:, 0] * 2 * np.pi
    return np.c_[np.cos(angle) * hsv[:, 1], np.sin(angle) * hsv[:, 1], hsv[:, 2]]


def material_of_rgb(rgb) -> int:
    """Asigna un material a un color RGB medio (centroide de K-Means)."""
    r, g, b = rgb
    if max(rgb) > 0.6:
        return LIGHT
    if g > r + 0.05 and g > b + 0.05:
        return GREEN
    if r > b + 0.1:
        return RUST
    return DARK


def cluster_materials(colors: np.ndarray, k: int = 5, random_state: int = 0, n_init: int = 30):
    """K-Means sobre el color; cada cluster se mapea a un material.

    Se usan varias inicializaciones (`n_init`) porque los grupos de color están
    poco separados: con 10 inicializaciones algunas semillas convergían a un
    mínimo local distinto (ver src/evaluation.py, estabilidad por semilla).

    Devuelve (material por punto, lista de (rgb medio, material) por cluster).
    """
    labels = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit_predict(
        color_features(colors))
    summary, mapping = [], np.empty(k, dtype=int)
    for c in range(k):
        mean_rgb = colors[labels == c].mean(axis=0)
        mapping[c] = material_of_rgb(mean_rgb)
        summary.append((mean_rgb, int(mapping[c])))
    return mapping[labels], summary


# --------------------------------------------------------------------------- #
# Etapa 2c-2d: objetos y su clase
# --------------------------------------------------------------------------- #
def classify_object(material: int, z_low: float, z_median: float,
                    elevated_min: float = 3.0, low_median_max: float = 2.5) -> int:
    """Clase de un objeto (segmento DBSCAN) según material y altura sobre el piso.

    Se usa la mediana de altura (y no el máximo) para que unos pocos puntos
    altos no impidan reconocer una máquina.
    """
    if material == LIGHT:
        return PIPE                  # tuberías: superficies claras / metálicas
    if z_low >= elevated_min:
        return BEAM                  # elementos elevados: vigas, cerchas, rieles
    if material == GREEN:
        return BEAM                  # estructura metálica verde (pórticos, arriostres)
    if z_median <= low_median_max:
        return MACHINERY             # objetos oscuros cercanos al piso
    return OTHER


def segment_objects(data: PointCloudData, materials: np.ndarray, floor_z: float,
                    eps: float = 0.2, min_samples: int = 10):
    """DBSCAN por material y clasificación de cada segmento.

    Los puntos que DBSCAN marca como ruido toman la clase del punto
    clasificado más cercano.

    Devuelve (clase por punto, id de objeto por punto; -1 = ruido).
    """
    classes = np.full(len(data), UNLABELED)
    objects = np.full(len(data), -1)
    next_id = 0
    for m in np.unique(materials):
        idx = np.flatnonzero(materials == m)
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(
            data.points[idx])
        for s in np.unique(labels[labels >= 0]):
            members = idx[labels == s]
            z = data.points[members, 2] - floor_z
            classes[members] = classify_object(m, z.min(), float(np.median(z)))
            objects[members] = next_id
            next_id += 1

    noise = classes == UNLABELED
    if noise.any() and (~noise).any():
        nn = NearestNeighbors(n_neighbors=1).fit(data.points[~noise])
        nearest = nn.kneighbors(data.points[noise], return_distance=False)[:, 0]
        classes[noise] = classes[~noise][nearest]
    elif noise.any():
        classes[noise] = OTHER
    return classes, objects


def smooth_labels(points: np.ndarray, classes: np.ndarray, k: int = 10,
                  fixed: np.ndarray | None = None) -> np.ndarray:
    """Voto mayoritario de la clase entre los k vecinos más cercanos.

    Corrige puntos aislados mal clasificados (p. ej. ruido de color).
    Los puntos con `fixed=True` participan en el voto pero no cambian.
    """
    idx = NearestNeighbors(n_neighbors=k).fit(points).kneighbors(points, return_distance=False)
    shifted = classes[idx] + 1  # -1 -> 0 para poder contar
    counts = np.zeros((len(points), shifted.max() + 1), dtype=np.int32)
    rows = np.arange(len(points))
    for j in range(k):
        np.add.at(counts, (rows, shifted[:, j]), 1)
    voted = counts.argmax(axis=1) - 1
    if fixed is not None:
        voted[fixed] = classes[fixed]
    return voted


def stage2_objects(data: PointCloudData, stage1: dict, n_colors: int = 5,
                   object_eps: float = 0.2, smooth_k: int = 10, random_state: int = 0):
    """Ejecuta la etapa 2 sobre los puntos no estructurales de la etapa 1.

    Devuelve un dict con:
      classes:   clase final por punto (toda la nube)
      objects:   id de objeto por punto (-1 fuera de la etapa 2 o ruido)
      materials: material por punto (-1 fuera de la etapa 2 o columnas)
      colors:    resumen de clusters de color [(rgb medio, material)]
      floor_z:   altura de referencia del piso
    """
    classes = stage1["classes"].copy()
    floor_z = float(np.median(data.points[classes == FLOOR, 2]))
    todo = np.flatnonzero(np.isin(classes, [UNLABELED, HORIZONTAL_OTHER]))
    rest = data.subset(todo)

    # 2a: columnas
    is_column = detect_columns(rest.points, floor_z)
    classes[todo[is_column]] = COLUMN

    # 2b-2d: material, objetos y clase sobre lo que no es columna
    others = todo[~is_column]
    sub = data.subset(others)
    materials, color_summary = cluster_materials(sub.colors, n_colors, random_state)
    obj_classes, obj_ids = segment_objects(sub, materials, floor_z, eps=object_eps)
    classes[others] = obj_classes

    # 2e: suavizado sobre los puntos de la etapa 2 (columnas fijas)
    classes[todo] = smooth_labels(rest.points, classes[todo], k=smooth_k, fixed=is_column)

    objects = np.full(len(data), -1)
    objects[others] = obj_ids
    mat_full = np.full(len(data), -1)
    mat_full[others] = materials
    return {"classes": classes, "objects": objects, "materials": mat_full,
            "colors": color_summary, "floor_z": floor_z}
