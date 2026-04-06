"""
clustering.py – three clustering methods:
  1. Baseline KMeans (fixed K)
  2. Improved KMeans with silhouette-based K selection (elbow/silhouette sweep)
  3. CDPC-KNN approximation: density-peak clustering via KMeans density proxy

Paper approximation note
────────────────────────
The paper referenced (CDPC-KNN, "Clustering by Fast Search and Find of Density Peaks
with K-Nearest Neighbours") requires computing a full pairwise distance matrix to
estimate local density ρ and the distance-to-higher-density δ for every point.
This is O(n²) and cannot run natively in Spark's distributed MLlib as-is.

Approximation strategy implemented here:
  • Run a coarse KMeans (large K) to partition the space.
  • Use each point's distance to its cluster centre as a proxy for local density
    (small distance → high density).
  • Identify "density peaks" as the K_final most isolated high-density points.
  • Assign remaining points to the nearest peak (Voronoi-style), equivalent to
    a second-stage KMeans seeded at the detected peaks.
  • This faithfully reproduces the spirit of CDPC-KNN (density estimation →
    peak detection → assignment) while being fully distributed.
"""

import time
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pyspark.ml.linalg as la


# ── 1. Baseline KMeans ────────────────────────────────────────────────────────

def run_kmeans_baseline(
    df: DataFrame,
    k: int = 5,
    features_col: str = "features",
    prediction_col: str = "prediction",
    seed: int = 42,
) -> tuple[DataFrame, KMeansModel, float]:
    """
    Standard KMeans with a fixed K.
    Returns (labelled_df, model, runtime_seconds).
    """
    t0 = time.time()
    km = KMeans(
        k=k,
        featuresCol=features_col,
        predictionCol=prediction_col,
        seed=seed,
        maxIter=50,
    )
    model = km.fit(df)
    labelled = model.transform(df)
    runtime = time.time() - t0
    print(f"[kmeans-baseline] k={k}, runtime={runtime:.1f}s")
    return labelled, model, runtime


# ── 2. Improved KMeans with silhouette-based K selection ─────────────────────

def run_kmeans_best_k(
    df: DataFrame,
    k_range: range = range(2, 11),
    features_col: str = "features",
    prediction_col: str = "prediction",
    seed: int = 42,
) -> tuple[DataFrame, KMeansModel, int, float, float]:
    """
    Sweep K over k_range, evaluate silhouette score, pick best K.
    Returns (labelled_df, best_model, best_k, best_silhouette, runtime_seconds).
    """
    evaluator = ClusteringEvaluator(
        featuresCol=features_col,
        predictionCol=prediction_col,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    best_k, best_score, best_model = None, -1.0, None
    t0 = time.time()

    for k in k_range:
        km = KMeans(
            k=k,
            featuresCol=features_col,
            predictionCol=prediction_col,
            seed=seed,
            maxIter=50,
        )
        model = km.fit(df)
        labelled = model.transform(df)
        score = evaluator.evaluate(labelled)
        print(f"  [k-sweep] k={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    labelled = best_model.transform(df)
    runtime = time.time() - t0
    print(f"[kmeans-best-k] best_k={best_k}, silhouette={best_score:.4f}, runtime={runtime:.1f}s")
    return labelled, best_model, best_k, best_score, runtime


# ── 3. CDPC-KNN Approximation ─────────────────────────────────────────────────

def _distance_to_centre(features, centre):
    """Squared Euclidean distance between a dense vector and a centre array."""
    import numpy as np
    v = np.array(features.toArray())
    c = np.array(centre)
    return float(np.dot(v - c, v - c))


def run_cdpc_knn_approx(
    df: DataFrame,
    k_final: int = 5,
    k_coarse: int = 50,
    features_col: str = "features",
    prediction_col: str = "prediction",
    seed: int = 42,
) -> tuple[DataFrame, float]:
    """
    CDPC-KNN approximation via two-stage KMeans density proxy.

    Stage 1 – coarse KMeans (k_coarse) partitions space finely.
    Stage 2 – pick k_final 'density peaks' from coarse centres
              (centres with the lowest average intra-cluster distance
               AND the farthest inter-peak separation).
    Stage 3 – re-assign all points to nearest peak (KMeans seeded at peaks).

    Returns (labelled_df, runtime_seconds).
    """
    import numpy as np
    from pyspark.ml.linalg import Vectors

    t0 = time.time()

    # ── Stage 1: coarse partition ─────────────────────────────────────────────
    k_coarse = min(k_coarse, df.count() - 1)  # guard for small datasets
    km_coarse = KMeans(
        k=k_coarse,
        featuresCol=features_col,
        predictionCol="_coarse_cluster",
        seed=seed,
        maxIter=30,
    )
    coarse_model = km_coarse.fit(df)
    coarse_df = coarse_model.transform(df)

    # ── Stage 2: density proxy per coarse cluster ─────────────────────────────
    # clusterCenters() returns numpy arrays in PySpark >= 3.x; handle both cases
    raw_centres = coarse_model.clusterCenters()
    centres = np.array([c.toArray() if hasattr(c, "toArray") else np.array(c) for c in raw_centres])

    # broadcast centres for UDF
    centres_bc = df.sparkSession.sparkContext.broadcast(centres)

    @F.udf(DoubleType())
    def dist_udf(features, cluster_id):
        import numpy as _np
        c = centres_bc.value[cluster_id]
        v = features.toArray() if hasattr(features, "toArray") else _np.array(features)
        return float(_np.dot(v - c, v - c))

    coarse_df = coarse_df.withColumn(
        "_dist_to_centre",
        dist_udf(F.col(features_col), F.col("_coarse_cluster")),
    )

    # average intra-cluster distance = density proxy (lower → denser)
    cluster_density = (
        coarse_df.groupBy("_coarse_cluster")
        .agg(F.avg("_dist_to_centre").alias("avg_dist"))
        .orderBy("avg_dist")  # ascending: most dense first
        .collect()
    )

    # ── pick k_final peaks using greedy max-spacing from dense candidates ─────
    # Take top-50% densest candidates, then greedily select k_final
    # with maximum minimum inter-peak distance (diversity criterion).
    n_candidates = max(k_final, len(cluster_density) // 2)
    candidates = [row["_coarse_cluster"] for row in cluster_density[:n_candidates]]
    candidate_centres = centres[candidates]

    selected_indices = [0]  # start with densest
    for _ in range(k_final - 1):
        remaining = [i for i in range(len(candidates)) if i not in selected_indices]
        if not remaining:
            break
        # pick the candidate farthest from all already-selected peaks
        best_i, best_dist = None, -1.0
        for i in remaining:
            min_dist = min(
                np.sum((candidate_centres[i] - candidate_centres[j]) ** 2)
                for j in selected_indices
            )
            if min_dist > best_dist:
                best_dist = min_dist
                best_i = i
        selected_indices.append(best_i)

    peak_centres = candidate_centres[selected_indices]
    print(f"[cdpc-approx] selected {len(peak_centres)} density peaks from {k_coarse} coarse clusters")

    # ── Stage 3: re-assign using KMeans seeded at peaks ──────────────────────
    # Spark KMeans doesn't support custom init centres directly via fit(),
    # so we implement nearest-peak assignment with a UDF.
    peak_bc = df.sparkSession.sparkContext.broadcast(peak_centres)

    @F.udf("int")
    def assign_to_peak(features):
        import numpy as _np
        v = features.toArray() if hasattr(features, "toArray") else _np.array(features)
        dists = [float(_np.sum((v - c) ** 2)) for c in peak_bc.value]
        return int(_np.argmin(dists))

    labelled = df.withColumn(prediction_col, assign_to_peak(F.col(features_col)))

    runtime = time.time() - t0
    print(f"[cdpc-approx] k_final={k_final}, runtime={runtime:.1f}s")
    return labelled, runtime


# ── shared utility ────────────────────────────────────────────────────────────

def compute_silhouette(df: DataFrame, features_col: str = "features", prediction_col: str = "prediction") -> float:
    """Compute silhouette score. Returns -1 if fewer than 2 clusters."""
    n_clusters = df.select(prediction_col).distinct().count()
    if n_clusters < 2:
        print("[eval] fewer than 2 clusters – silhouette not defined")
        return -1.0
    evaluator = ClusteringEvaluator(
        featuresCol=features_col,
        predictionCol=prediction_col,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    return evaluator.evaluate(df)