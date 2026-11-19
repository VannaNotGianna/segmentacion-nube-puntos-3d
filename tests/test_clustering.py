"""Pruebas de cada etapa de clustering sobre datos con respuesta conocida."""

import numpy as np
import pytest

from src import clustering as C


# --------------------------------------------------------------------------- #
# Etapa 1
# --------------------------------------------------------------------------- #
def test_orientation_maps_axes_regardless_of_kmeans_order():
    rng = np.random.default_rng(1)
    base = np.repeat(np.eye(3), 300, axis=0) + rng.normal(0, 0.05, (900, 3))
    base *= rng.choice([-1, 1], size=(900, 1))  # signo arbitrario
    orientation, centers = C.cluster_orientation(base)
    expected = np.repeat([C.VERTICAL_X, C.VERTICAL_Y, C.HORIZONTAL], 300)
    assert (orientation == expected).mean() > 0.99
    assert centers[C.HORIZONTAL].argmax() == 2


def test_label_horizontal_surfaces():
    rng = np.random.default_rng(2)
    floor = np.c_[rng.uniform(0, 20, (3000, 2)), rng.normal(0, 0.02, 3000)]
    ceiling = np.c_[rng.uniform(0, 20, (1000, 2)), 8 + rng.normal(0, 0.02, 1000)]
    platform = np.c_[rng.uniform(0, 5, (500, 2)), 3 + rng.normal(0, 0.02, 500)]   # 25 m²
    table = np.c_[rng.uniform(0, 1, (300, 2)), 1 + rng.normal(0, 0.02, 300)]      # 1 m²
    pts = np.vstack([floor, ceiling, platform, table])
    segs = np.repeat([0, 1, 2, 3], [3000, 1000, 500, 300])
    classes = C.label_horizontal_surfaces(pts, segs)
    expected = np.repeat([C.FLOOR, C.CEILING, C.PLATFORM, C.HORIZONTAL_OTHER],
                         [3000, 1000, 500, 300])
    np.testing.assert_array_equal(classes, expected)


def test_cluster_surfaces_warns_when_eps_too_small():
    x, y = np.meshgrid(np.arange(0, 5, 0.08), np.arange(0, 5, 0.08))
    plane = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    with pytest.warns(RuntimeWarning, match="ruido"):
        C.cluster_surfaces(plane, eps=0.05)


def test_floor_is_lowest_large_surface_even_if_ceiling_is_bigger():
    rng = np.random.default_rng(4)
    floor = np.c_[rng.uniform(0, 10, (2000, 2)), rng.normal(0, 0.02, 2000)]
    ceiling = np.c_[rng.uniform(0, 10, (2500, 2)), 8 + rng.normal(0, 0.02, 2500)]
    debris = np.c_[rng.uniform(0, 0.3, (50, 2)), -1 + rng.normal(0, 0.02, 50)]  # pequeño y bajo
    pts = np.vstack([floor, ceiling, debris])
    segs = np.repeat([0, 1, 2], [2000, 2500, 50])
    classes = C.label_horizontal_surfaces(pts, segs)
    assert (classes[:2000] == C.FLOOR).all()
    assert (classes[2000:4500] == C.CEILING).all()
    assert not (classes[4500:] == C.FLOOR).any()


def test_stage1_on_scene(scene):
    data, truth = scene
    s1 = C.stage1_structure(data)
    for cls in (C.FLOOR, C.CEILING):
        assert (s1["classes"][truth == cls] == cls).mean() > 0.95, C.CLASS_NAMES[cls]
    # Ningún punto de columna, tubería, viga o máquina debe quedar como piso/techo
    objects = np.isin(truth, [C.COLUMN, C.PIPE, C.BEAM, C.MACHINERY])
    assert not np.isin(s1["classes"][objects], [C.FLOOR, C.CEILING]).any()


# --------------------------------------------------------------------------- #
# Etapa 2
# --------------------------------------------------------------------------- #
def test_vertical_coverage():
    z = np.arange(0, 8, 0.1)
    tall = np.c_[np.zeros_like(z), np.zeros_like(z), z]
    short = np.c_[np.full(5, 5.0), np.full(5, 5.0), np.linspace(0, 0.4, 5)]
    cov = C.vertical_coverage(np.vstack([tall, short]), floor_z=0.0)
    assert np.all(cov[: len(z)] == 8.0)
    assert np.all(cov[len(z):] == 0.5)


def test_detect_columns_ignores_wide_and_short_objects():
    rng = np.random.default_rng(3)
    column = np.c_[rng.uniform(0, 0.5, (2000, 2)), rng.uniform(0, 8, 2000)]
    wall = np.c_[rng.uniform(5, 15, 6000), rng.uniform(5, 5.3, 6000), rng.uniform(0, 8, 6000)]
    box = np.c_[rng.uniform(20, 21, (1000, 2)), rng.uniform(0, 1, 1000)]
    mask = C.detect_columns(np.vstack([column, wall, box]), floor_z=0.0)
    assert mask[:2000].mean() > 0.95
    assert not mask[2000:].any()


@pytest.mark.parametrize("rgb, material", [
    ((0.80, 0.82, 0.85), C.LIGHT),
    ((0.39, 0.53, 0.32), C.GREEN),
    ((0.42, 0.23, 0.17), C.RUST),
    ((0.19, 0.18, 0.26), C.DARK),
    ((0.29, 0.25, 0.25), C.DARK),
])
def test_material_of_rgb(rgb, material):
    assert C.material_of_rgb(rgb) == material


def test_color_features_gray_has_no_hue():
    feats = C.color_features(np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(feats[0, :2], 0, atol=1e-9)  # gris: sin tono
    assert np.hypot(*feats[1, :2]) == pytest.approx(1.0)    # rojo puro: saturación 1


@pytest.mark.parametrize("material, z_low, z_median, expected", [
    (C.LIGHT, 0.0, 1.0, C.PIPE),
    (C.DARK, 4.0, 5.0, C.BEAM),
    (C.GREEN, 0.0, 2.0, C.BEAM),
    (C.DARK, 0.0, 1.0, C.MACHINERY),
    (C.RUST, 0.5, 1.5, C.MACHINERY),
    (C.DARK, 0.0, 3.0, C.OTHER),
])
def test_classify_object(material, z_low, z_median, expected):
    assert C.classify_object(material, z_low, z_median) == expected


def test_smooth_labels_fixes_isolated_point_and_respects_fixed():
    pts = np.c_[np.arange(30) * 0.1, np.zeros(30), np.zeros(30)]
    labels = np.full(30, C.PIPE)
    labels[15] = C.MACHINERY
    out = C.smooth_labels(pts, labels, k=5)
    assert out[15] == C.PIPE

    fixed = np.zeros(30, dtype=bool)
    fixed[15] = True
    out = C.smooth_labels(pts, labels, k=5, fixed=fixed)
    assert out[15] == C.MACHINERY


def test_transfer_labels_nearest_and_far_points():
    sparse = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    sparse_labels = np.array([C.PIPE, C.COLUMN])
    full = np.array([[0.1, 0, 0], [0.9, 0, 0], [0.45, 0, 0], [5.0, 0, 0]])
    labels, dist = C.transfer_labels(full, sparse, sparse_labels, max_dist=0.5)
    np.testing.assert_array_equal(labels, [C.PIPE, C.COLUMN, C.PIPE, C.OTHER])
    assert dist[3] == pytest.approx(4.0)


def test_transfer_labels_on_scene(scene):
    """Submuestrear la escena, segmentar y volver a la resolución original."""
    import open3d as o3d

    from src.preprocessing import to_arrays, voxel_downsample

    data, truth = scene
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.points))
    pcd.colors = o3d.utility.Vector3dVector(data.colors)
    pcd.normals = o3d.utility.Vector3dVector(data.normals)
    down = to_arrays(voxel_downsample(pcd, 0.16))
    assert len(down) < 0.5 * len(data)

    s2 = C.stage2_objects(down, C.stage1_structure(down, surface_eps=0.3), object_eps=0.4)
    full_labels, _ = C.transfer_labels(data.points, down.points, s2["classes"], max_dist=0.32)
    assert len(full_labels) == len(data)
    assert (full_labels == truth).mean() > 0.95


# --------------------------------------------------------------------------- #
# Pipeline completo sobre la escena sintética
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def segmented(scene):
    data, truth = scene
    s1 = C.stage1_structure(data)
    s2 = C.stage2_objects(data, s1)
    return s2["classes"], truth


@pytest.mark.parametrize("cls", [C.FLOOR, C.CEILING, C.COLUMN, C.PIPE, C.BEAM, C.MACHINERY])
def test_full_pipeline_recovers_each_class(segmented, cls):
    pred, truth = segmented
    recall = (pred[truth == cls] == cls).mean()
    assert recall > 0.9, f"{C.CLASS_NAMES[cls]}: solo {recall:.0%} correcto"


def test_full_pipeline_overall_accuracy(segmented):
    pred, truth = segmented
    assert (pred == truth).mean() > 0.95
    assert not np.isin(pred, [C.UNLABELED, C.HORIZONTAL_OTHER]).any()
