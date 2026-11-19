import numpy as np
import open3d as o3d
import pytest

from src.preprocessing import PointCloudData, build_features, voxel_downsample


def _random_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return PointCloudData(rng.uniform(0, 10, (n, 3)), rng.uniform(0, 1, (n, 3)), normals)


def test_build_features_shape_and_scaling():
    data = _random_data()
    X = build_features(data)
    assert X.shape == (len(data), 9)
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-6)


def test_build_features_ignores_normal_sign():
    data = _random_data()
    flipped = PointCloudData(data.points, data.colors, -data.normals)
    np.testing.assert_allclose(build_features(data), build_features(flipped))


def test_build_features_weights_and_selection():
    data = _random_data()
    X = build_features(data, use_position=False, use_color=False, weights={"normals": 2.0})
    assert X.shape == (len(data), 3)
    np.testing.assert_allclose(X.std(axis=0), 2, atol=1e-6)
    with pytest.raises(ValueError):
        build_features(data, use_position=False, use_color=False, use_normals=False)


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
