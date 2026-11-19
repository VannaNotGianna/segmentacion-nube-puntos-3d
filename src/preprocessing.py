"""Carga y preparación de la nube de puntos."""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from sklearn.preprocessing import StandardScaler


@dataclass
class PointCloudData:
    """Arreglos NumPy de una nube de puntos (N filas cada uno)."""

    points: np.ndarray   # (N, 3) XYZ en metros, Z hacia arriba
    colors: np.ndarray   # (N, 3) RGB en [0, 1]
    normals: np.ndarray  # (N, 3) normales unitarias

    def __len__(self) -> int:
        return len(self.points)

    def subset(self, mask: np.ndarray) -> "PointCloudData":
        return PointCloudData(self.points[mask], self.colors[mask], self.normals[mask])


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"No se pudo leer la nube de puntos: {path}")
    if not pcd.has_normals():
        raise ValueError("La nube no contiene normales")
    return pcd


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """Reduce la densidad promediando posición, color y normal en cada vóxel."""
    down = pcd.voxel_down_sample(voxel_size)
    # El promedio de normales deja de ser unitario; se renormaliza.
    down.normalize_normals()
    return down


def remove_outliers(pcd, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """Elimina puntos aislados (filtro estadístico). Devuelve (nube, índices conservados)."""
    return pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)


def to_arrays(pcd: o3d.geometry.PointCloud) -> PointCloudData:
    return PointCloudData(
        points=np.asarray(pcd.points).copy(),
        colors=np.asarray(pcd.colors).copy(),
        normals=np.asarray(pcd.normals).copy(),
    )


def build_features(
    data: PointCloudData,
    use_position: bool = True,
    use_color: bool = True,
    use_normals: bool = True,
    weights: dict | None = None,
) -> np.ndarray:
    """Concatena atributos estandarizados en una matriz de características.

    Las normales se usan en valor absoluto: el signo de la normal es arbitrario
    (n y -n describen la misma superficie). `weights` permite dar más peso a un
    grupo de atributos, p. ej. {"normals": 2.0}.
    """
    weights = weights or {}
    blocks = []
    if use_position:
        blocks.append(("position", data.points))
    if use_color:
        blocks.append(("color", data.colors))
    if use_normals:
        blocks.append(("normals", np.abs(data.normals)))
    if not blocks:
        raise ValueError("Debe seleccionarse al menos un grupo de atributos")

    feats = []
    for name, block in blocks:
        scaled = StandardScaler().fit_transform(block)
        feats.append(scaled * weights.get(name, 1.0))
    return np.hstack(feats)
