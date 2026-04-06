"""
loader.py – auto-detect and load CERT Insider Threat r1 dataset files,
then join them into a single per-user feature table.
"""

import os
import glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


# ── known schemas (CERT r1 format) ───────────────────────────────────────────
# The CERT r1 CSVs have NO header row – columns are positional.

LOGON_SCHEMA = StructType([
    StructField("id",       StringType(), True),
    StructField("date",     StringType(), True),
    StructField("user",     StringType(), True),
    StructField("pc",       StringType(), True),
    StructField("activity", StringType(), True),
])

DEVICE_SCHEMA = StructType([
    StructField("id",       StringType(), True),
    StructField("date",     StringType(), True),
    StructField("user",     StringType(), True),
    StructField("pc",       StringType(), True),
    StructField("activity", StringType(), True),
])

HTTP_SCHEMA = StructType([
    StructField("id",   StringType(), True),
    StructField("date", StringType(), True),
    StructField("user", StringType(), True),
    StructField("pc",   StringType(), True),
    StructField("url",  StringType(), True),
])

LDAP_SCHEMA = StructType([
    StructField("employee_name", StringType(), True),
    StructField("user_id",       StringType(), True),
    StructField("email",         StringType(), True),
    StructField("domain",        StringType(), True),
    StructField("role",          StringType(), True),
])

# Multiple timestamp formats seen in CERT datasets
TIMESTAMP_FORMATS = [
    "MM/dd/yyyy HH:mm:ss",
    "yyyy-MM-dd HH:mm:ss",
    "M/d/yyyy H:mm",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_files(data_dir: str, pattern: str) -> list[str]:
    """Return sorted list of files matching a glob pattern."""
    return sorted(glob.glob(os.path.join(data_dir, pattern)))


def _has_header(spark: SparkSession, path: str, expected_col: str) -> bool:
    """Peek at the first row to determine if the file has a header."""
    peek = spark.read.option("header", False).csv(path)
    first_row = peek.first()
    if first_row is None:
        return False
    # If any value in the first row matches the expected column name it's a header
    return any(
        str(v).strip().lower() == expected_col.lower()
        for v in first_row
        if v is not None
    )


def _load_csv(
    spark: SparkSession,
    paths: list[str],
    schema: StructType,
    header_col: str,
) -> "pyspark.sql.DataFrame | None":
    """
    Load CSV files with the given schema.
    Auto-detects whether a header row is present using the first file.
    """
    if not paths:
        return None

    has_hdr = _has_header(spark, paths[0], header_col)
    print(f"[loader] {'header detected' if has_hdr else 'no header – using fixed schema'}: {os.path.basename(paths[0])}")

    return (
        spark.read
        .option("header", str(has_hdr).lower())
        .schema(schema)
        .csv(paths)
    )


def _parse_timestamp(df, col_name: str) -> "pyspark.sql.DataFrame":
    """
    Try multiple timestamp formats until one works.
    Falls back to coerce_to_date if all formats fail.
    """
    # Try each format; coalesce returns the first non-null result
    parsed = F.coalesce(*[
        F.to_timestamp(F.col(col_name), fmt)
        for fmt in TIMESTAMP_FORMATS
    ])
    return df.withColumn("date_ts", parsed)


# ── public API ────────────────────────────────────────────────────────────────

def detect_dataset(data_dir: str) -> dict:
    """Scan data_dir and return a dict mapping file-role → list[path]."""
    mapping = {
        "logon":  _find_files(data_dir, "logon*.csv"),
        "device": _find_files(data_dir, "device*.csv"),
        "http":   _find_files(data_dir, "http*.csv") or _find_files(data_dir, "HTTP*.csv"),
        "ldap":   _find_files(data_dir, "LDAP*.csv") or _find_files(data_dir, "ldap*.csv"),
    }
    # Also check for generic parquet fallback
    parquet = _find_files(data_dir, "*.parquet")
    if parquet:
        mapping["parquet"] = parquet
    found = {k: v for k, v in mapping.items() if v}
    print(f"[loader] detected files: { {k: len(v) for k, v in found.items()} }")
    return found


def build_user_features(spark: SparkSession, data_dir: str) -> "pyspark.sql.DataFrame":
    """
    Load all source files and aggregate into one row per user with
    behavioural features suitable for clustering.
    """
    files = detect_dataset(data_dir)

    # ── logon features ────────────────────────────────────────────────────────
    logon_df = None
    if "logon" in files:
        raw = _load_csv(spark, files["logon"], LOGON_SCHEMA, header_col="id")
        raw = _parse_timestamp(raw, "date").withColumn("hour", F.hour("date_ts"))

        logon_df = raw.groupBy("user").agg(
            F.count("*").alias("logon_count"),
            F.countDistinct("pc").alias("logon_distinct_pcs"),
            F.avg("hour").alias("logon_avg_hour"),
            F.stddev("hour").alias("logon_hour_std"),
            F.sum(
                F.when((F.col("hour") < 8) | (F.col("hour") >= 18), 1).otherwise(0)
            ).alias("afterhours_logons"),
            F.sum(F.when(F.col("activity") == "Logon", 1).otherwise(0)).alias("logon_events"),
            F.sum(F.when(F.col("activity") == "Logoff", 1).otherwise(0)).alias("logoff_events"),
        )

    # ── device (thumb drive) features ────────────────────────────────────────
    device_df = None
    if "device" in files:
        raw = _load_csv(spark, files["device"], DEVICE_SCHEMA, header_col="id")
        raw = _parse_timestamp(raw, "date").withColumn("hour", F.hour("date_ts"))

        device_df = raw.groupBy("user").agg(
            F.count("*").alias("device_events"),
            F.countDistinct("pc").alias("device_distinct_pcs"),
            F.sum(
                F.when((F.col("hour") < 8) | (F.col("hour") >= 18), 1).otherwise(0)
            ).alias("afterhours_device"),
        )

    # ── HTTP features ─────────────────────────────────────────────────────────
    http_df = None
    if "http" in files:
        raw = _load_csv(spark, files["http"], HTTP_SCHEMA, header_col="id")
        raw = _parse_timestamp(raw, "date").withColumn("hour", F.hour("date_ts"))

        http_df = raw.groupBy("user").agg(
            F.count("*").alias("http_requests"),
            F.countDistinct("url").alias("http_distinct_urls"),
            F.sum(
                F.when((F.col("hour") < 8) | (F.col("hour") >= 18), 1).otherwise(0)
            ).alias("afterhours_http"),
        )

    # ── LDAP features (role / department metadata) ────────────────────────────
    ldap_df = None
    if "ldap" in files:
        # Use the last LDAP snapshot (most recent active employee list)
        raw = _load_csv(spark, [files["ldap"][-1]], LDAP_SCHEMA, header_col="employee_name")

        ldap_df = raw.select(
            F.col("user_id").alias("user"),
            F.col("role"),
        ).dropDuplicates(["user"])

    # ── parquet fallback (generic) ────────────────────────────────────────────
    if "parquet" in files and logon_df is None:
        print("[loader] falling back to parquet file(s)")
        return spark.read.parquet(*files["parquet"])

    # ── join all feature tables ───────────────────────────────────────────────
    tables = [t for t in [logon_df, device_df, http_df] if t is not None]
    if not tables:
        raise RuntimeError("No recognisable data files found in the data directory.")

    combined = tables[0]
    for tbl in tables[1:]:
        combined = combined.join(tbl, on="user", how="outer")

    if ldap_df is not None:
        combined = combined.join(ldap_df, on="user", how="left")

    # fill missing counts with 0
    num_cols = [c for c, t in combined.dtypes if t in ("int", "bigint", "double", "float") and c != "user"]
    combined = combined.fillna(0, subset=num_cols)

    print(f"[loader] user feature table: {combined.count()} rows × {len(combined.columns)} cols")
    return combined