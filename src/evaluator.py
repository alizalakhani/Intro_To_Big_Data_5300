"""
evaluator.py – compute and report clustering quality metrics.

Internal metrics (always available):
  - Silhouette score (squaredEuclidean)
  - Inertia / WSSSE (within-cluster sum of squared errors) for KMeans models

External metrics (only if a ground-truth label column exists):
  - Normalised Mutual Information (NMI)
  - Adjusted Rand Index (ARI)
  - Purity / cluster accuracy

Note: Spark MLlib doesn't ship NMI/ARI natively, so we collect a sample
to the driver and compute via scikit-learn.  For very large datasets this
sample is capped at 500k rows.
"""

import json
import time
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.clustering import KMeansModel

DRIVER_SAMPLE_LIMIT = 500_000


def _try_sklearn_metrics(true_labels, pred_labels) -> dict:
    try:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        return {"nmi": round(nmi, 4), "ari": round(ari, 4)}
    except ImportError:
        print("[eval] scikit-learn not installed – skipping NMI/ARI")
        return {}


def evaluate(
    df: DataFrame,
    method_name: str,
    prediction_col: str = "prediction",
    features_col: str = "features",
    label_col: str | None = None,
    kmeans_model: KMeansModel | None = None,
    runtime: float = 0.0,
) -> dict:
    """
    Compute all available metrics and return a result dict.
    """
    from clustering import compute_silhouette

    result = {
        "method": method_name,
        "runtime_s": round(runtime, 2),
    }

    # cluster count
    result["n_clusters"] = df.select(prediction_col).distinct().count()

    # silhouette
    sil = compute_silhouette(df, features_col=features_col, prediction_col=prediction_col)
    result["silhouette"] = round(sil, 4)

    # inertia (KMeans only)
    if kmeans_model is not None:
        result["inertia"] = round(kmeans_model.summary.trainingCost, 2)

    # cluster size distribution
    sizes = (
        df.groupBy(prediction_col)
        .count()
        .orderBy(prediction_col)
        .rdd.map(lambda r: (r[0], r[1]))
        .collectAsMap()
    )
    result["cluster_sizes"] = sizes

    # external metrics
    if label_col and label_col in df.columns:
        n = min(df.count(), DRIVER_SAMPLE_LIMIT)
        sample = df.select(label_col, prediction_col).limit(n).toPandas()
        ext = _try_sklearn_metrics(sample[label_col].tolist(), sample[prediction_col].tolist())
        result.update(ext)

    return result


def print_summary(results: list[dict]) -> None:
    header = f"\n{'Method':<25} {'K':>4} {'Silhouette':>12} {'Inertia':>14} {'Runtime(s)':>11}"
    print(header)
    print("─" * len(header))
    for r in results:
        inertia_str = f"{r.get('inertia', 'N/A'):>14}" if isinstance(r.get("inertia"), float) else f"{'N/A':>14}"
        print(
            f"{r['method']:<25} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
            f"{inertia_str} {r['runtime_s']:>11.1f}"
        )
    print()


def save_metrics(results: list[dict], output_dir: str) -> None:
    path = Path(output_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] metrics saved → {path}")
