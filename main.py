"""
main.py – entry point for the CERT Insider Threat clustering pipeline.

Usage
─────
  python main.py [--data-dir r1] [--output-dir output] [--k 5] [--k-min 2] [--k-max 10]

The script will:
  1. Auto-detect dataset files in --data-dir
  2. Build per-user behavioural features
  3. Preprocess (impute / encode / scale)
  4. Run three clustering methods
  5. Evaluate and compare results
  6. Save labelled data and metrics to --output-dir
"""

import argparse
import sys
import os
import time

# ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pyspark.sql import SparkSession

from loader import build_user_features
from preprocessor import preprocess
from clustering import run_kmeans_baseline, run_kmeans_best_k, run_cdpc_knn_approx
from evaluator import evaluate, print_summary, save_metrics


# ── Spark session ─────────────────────────────────────────────────────────────

def get_spark(app_name: str = "CERT-Clustering") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: str,
    output_dir: str,
    k_baseline: int,
    k_min: int,
    k_max: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    # 1. load ──────────────────────────────────────────────────────────────────
    print("\n═══ STEP 1: DATA LOADING ═══")
    raw_df = build_user_features(spark, data_dir)
    raw_df.cache()

    # 2. preprocess ────────────────────────────────────────────────────────────
    print("\n═══ STEP 2: PREPROCESSING ═══")
    processed_df, num_cols, cat_cols = preprocess(raw_df, id_col="user")
    processed_df.cache()

    # detect optional label column
    label_col = next((c for c in ["label", "insider", "malicious"] if c in processed_df.columns), None)
    if label_col:
        print(f"[main] ground-truth label column detected: '{label_col}'")

    results = []

    # 3a. baseline KMeans ──────────────────────────────────────────────────────
    print("\n═══ STEP 3a: BASELINE KMEANS ═══")
    bl_df, bl_model, bl_runtime = run_kmeans_baseline(
        processed_df, k=k_baseline, prediction_col="prediction_baseline"
    )
    results.append(
        evaluate(
            bl_df,
            method_name="KMeans-Baseline",
            prediction_col="prediction_baseline",
            label_col=label_col,
            kmeans_model=bl_model,
            runtime=bl_runtime,
        )
    )

    # 3b. improved KMeans (best K) ─────────────────────────────────────────────
    print(f"\n═══ STEP 3b: IMPROVED KMEANS (k sweep {k_min}–{k_max}) ═══")
    bk_df, bk_model, best_k, best_sil, bk_runtime = run_kmeans_best_k(
        processed_df, k_range=range(k_min, k_max + 1), prediction_col="prediction_bestk"
    )
    results.append(
        evaluate(
            bk_df,
            method_name=f"KMeans-BestK(k={best_k})",
            prediction_col="prediction_bestk",
            label_col=label_col,
            kmeans_model=bk_model,
            runtime=bk_runtime,
        )
    )

    # 3c. CDPC-KNN approximation ───────────────────────────────────────────────
    print(f"\n═══ STEP 3c: CDPC-KNN APPROXIMATION ═══")
    cdpc_df, cdpc_runtime = run_cdpc_knn_approx(
        processed_df, k_final=best_k, k_coarse=min(50, max(best_k * 5, 20)),
        prediction_col="prediction_cdpc"
    )
    results.append(
        evaluate(
            cdpc_df,
            method_name="CDPC-KNN-Approx",
            prediction_col="prediction_cdpc",
            label_col=label_col,
            runtime=cdpc_runtime,
        )
    )

    # 4. print summary ─────────────────────────────────────────────────────────
    print("\n═══ RESULTS SUMMARY ═══")
    print_summary(results)

    # 5. save outputs ──────────────────────────────────────────────────────────
    print("═══ STEP 5: SAVING OUTPUTS ═══")

    # merge all predictions onto the raw user table and save
    final_df = (
        raw_df
        .join(bl_df.select("user", "prediction_baseline"), on="user", how="left")
        .join(bk_df.select("user", "prediction_bestk"), on="user", how="left")
        .join(cdpc_df.select("user", "prediction_cdpc"), on="user", how="left")
    )
    out_path = os.path.join(output_dir, "clustered_users")
    final_df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_path)
    print(f"[main] clustered data saved → {out_path}/")

    save_metrics(results, output_dir)

    print("\n✓ Pipeline complete.\n")
    spark.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CERT Insider Threat Clustering Pipeline")
    p.add_argument("--data-dir",   default="r1",     help="Directory containing the dataset files")
    p.add_argument("--output-dir", default="output", help="Directory for output files")
    p.add_argument("--k",          type=int, default=5,  help="K for baseline KMeans")
    p.add_argument("--k-min",      type=int, default=2,  help="Min K for sweep")
    p.add_argument("--k-max",      type=int, default=10, help="Max K for sweep")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        k_baseline=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
    )
