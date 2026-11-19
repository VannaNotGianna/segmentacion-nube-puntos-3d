import numpy as np
import open3d as o3d

from src.preprocessing import PointCloudData, voxel_downsample


def _random_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return PointCloudData(rng.uniform(0, 10, (n, 3)), rng.uniform(0, 1, (n, 3)), normals)


def test_voxel_downsample_keeps_unit_normals():
    data = _random_data(2000)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.points))
    pcd.colors = o3d.utility.Vector3dVector(data.colors)
    pcd.normals = o3d.utility.Vector3dVector(data.normals)
    down = voxel_downsample(pcd, 1.0)
    assert 0 < len(down.points) < 2000
    norms = np.linalg.norm(np.asarray(down.normals), axis=1)
    np.testing.assert_allclose(norms, 1, atol=1e-6)


def test_subset():
    data = _random_data(10)
    mask = np.arange(10) % 2 == 0
    sub = data.subset(mask)
    assert len(sub) == 5
    np.testing.assert_array_equal(sub.points, data.points[mask])
