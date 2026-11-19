"""Evaluación de la segmentación sin etiquetas de referencia.

1. Selección de k para las etapas K-Means (orientación y color):
   inercia (codo), silhouette y Davies-Bouldin para varios k.
2. Estabilidad: concordancia de las clases finales al cambiar la semilla
   aleatoria y los parámetros principales (eps de DBSCAN, k de color).

Uso (desde la raíz del proyecto):
    python -m src.evaluation
"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from . import clustering as C
from .pipeline import ROOT, _display_path, prepare_data


# --------------------------------------------------------------------------- #
# Métricas internas de K-Means
# --------------------------------------------------------------------------- #
def kmeans_sweep(X: np.ndarray, ks, sample_size: int = 10_000, random_state: int = 0) -> pd.DataFrame:
    """Ajusta K-Means para cada k y calcula métricas internas.

    - inercia: suma de distancias al centroide (menor = clusters más compactos;
      siempre baja con k, se busca el "codo").
    - silhouette: en [-1, 1], mayor = clusters mejor separados.
    - davies_bouldin: >= 0, menor = mejor.

    Silhouette es O(n²), por eso se calcula sobre una muestra aleatoria.
    """
    rng = np.random.default_rng(random_state)
    sample = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        labels = km.labels_[sample]
        rows.append({
            "k": k,
            "inercia": km.inertia_,
            "silhouette": silhouette_score(X[sample], labels),
            "davies_bouldin": davies_bouldin_score(X[sample], labels),
        })
    return pd.DataFrame(rows)


def color_feature_set(data, stage1) -> np.ndarray:
    """Características de color de los puntos que entran a la etapa 2b."""
    classes = stage1["classes"]
    floor_z = float(np.median(data.points[classes == C.FLOOR, 2]))
    todo = np.flatnonzero(np.isin(classes, [C.UNLABELED, C.HORIZONTAL_OTHER]))
    is_column = C.detect_columns(data.points[todo], floor_z)
    return C.color_features(data.colors[todo[~is_column]])


def plot_sweeps(sweeps: dict, chosen: dict, path) -> None:
    metrics = [("inercia", "Inercia (codo)"), ("silhouette", "Silhouette (↑ mejor)"),
               ("davies_bouldin", "Davies-Bouldin (↓ mejor)")]
    fig, axes = plt.subplots(len(sweeps), 3, figsize=(15, 4 * len(sweeps)), squeeze=False)
    for row, (name, df) in enumerate(sweeps.items()):
        for col, (m, title) in enumerate(metrics):
            ax = axes[row, col]
            ax.plot(df["k"], df[m], "o-")
            ax.axvline(chosen[name], color="tab:red", ls="--", label=f"k elegido = {chosen[name]}")
            ax.set_title(f"{name}: {title}")
            ax.set_xlabel("k")
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Estabilidad
# --------------------------------------------------------------------------- #
def run_segmentation(data, random_state=0, surface_eps=0.15, object_eps=0.2, n_colors=5):
    s1 = C.stage1_structure(data, surface_eps=surface_eps, random_state=random_state)
    s2 = C.stage2_objects(data, s1, n_colors=n_colors, object_eps=object_eps,
                          random_state=random_state)
    return s2["classes"]


def compare(reference: np.ndarray, other: np.ndarray) -> dict:
    """Concordancia global, ARI y concordancia por clase respecto a la referencia."""
    out = {"concordancia": (reference == other).mean(),
           "ARI": adjusted_rand_score(reference, other)}
    for c in np.unique(reference):
        out[C.CLASS_NAMES[c]] = (other[reference == c] == c).mean()
    return out


def stability(data, baseline: np.ndarray, variants: dict) -> pd.DataFrame:
    rows = []
    for name, params in variants.items():
        print(f"  variante: {name}")
        labels = run_segmentation(data, **params)
        rows.append({"variante": name, **compare(baseline, labels)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    matplotlib.use("Agg")  # sin ventanas: solo se guardan figuras
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out_dir = ROOT / "results/metricas"
    fig_dir = ROOT / "results/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)

    data = prepare_data()

    print("\n1) Selección de k")
    s1 = C.stage1_structure(data)
    sweeps = {
        "orientacion (|normal|)": kmeans_sweep(np.abs(data.normals), range(2, 7)),
        "color (etapa 2b)": kmeans_sweep(color_feature_set(data, s1), range(2, 9)),
    }
    chosen = {"orientacion (|normal|)": 3, "color (etapa 2b)": 5}
    for name, df in sweeps.items():
        print(f"\n  {name}\n{df.to_string(index=False)}")
        df.to_csv(out_dir / f"kmeans_{name.split()[0]}.csv", index=False)
    plot_sweeps(sweeps, chosen, fig_dir / "06_seleccion_k.png")

    print("\n2) Estabilidad (referencia: parámetros por defecto, semilla 0)")
    baseline = run_segmentation(data)
    variants = {f"semilla {s}": {"random_state": s} for s in (1, 2, 3, 4)}
    variants.update({
        "eps superficies 0.10": {"surface_eps": 0.10},
        "eps superficies 0.20": {"surface_eps": 0.20},
        "eps objetos 0.15": {"object_eps": 0.15},
        "eps objetos 0.25": {"object_eps": 0.25},
        "k color 4": {"n_colors": 4},
        "k color 6": {"n_colors": 6},
    })
    stab = stability(data, baseline, variants)
    print(stab.to_string(index=False))
    stab.to_csv(out_dir / "estabilidad.csv", index=False)
    print(f"\nResultados en {_display_path(out_dir)} y {_display_path(fig_dir / '06_seleccion_k.png')}")


if __name__ == "__main__":
    main()
