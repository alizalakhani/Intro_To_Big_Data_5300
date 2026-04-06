"""
preprocessor.py – identify column types, handle missing values,
encode categoricals, scale numericals, assemble feature vector.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    StandardScaler,
    VectorAssembler,
    Imputer,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, NumericType


# ── column detection ──────────────────────────────────────────────────────────

def detect_column_types(df: DataFrame, id_col: str = "user") -> tuple[list, list]:
    """
    Return (numerical_cols, categorical_cols) excluding the id column.
    A column is categorical if its Spark type is StringType and it has
    fewer than 50 distinct values (avoid treating free-text as categorical).
    """
    numerical, categorical = [], []
    for field in df.schema.fields:
        if field.name == id_col:
            continue
        if isinstance(field.dataType, NumericType):
            numerical.append(field.name)
        elif isinstance(field.dataType, StringType):
            n_distinct = df.select(field.name).distinct().count()
            if n_distinct <= 50:
                categorical.append(field.name)
            # else: drop high-cardinality string col (e.g. free-text)
    print(f"[preproc] numerical={numerical}")
    print(f"[preproc] categorical={categorical}")
    return numerical, categorical


# ── pipeline builder ──────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    numerical_cols: list,
    categorical_cols: list,
    output_col: str = "features",
) -> Pipeline:
    """
    Build a Spark ML Pipeline that:
    1. Imputes missing numericals with median (approximated via mean in Spark Imputer).
    2. StringIndexes + OneHotEncodes each categorical.
    3. Assembles everything into one vector.
    4. Standard-scales the final vector.
    """
    stages = []

    # ── 1. impute numericals ──────────────────────────────────────────────────
    imputed_num = [f"{c}_imp" for c in numerical_cols]
    if numerical_cols:
        imputer = Imputer(
            inputCols=numerical_cols,
            outputCols=imputed_num,
            strategy="mean",          # Spark Imputer supports mean/median
        )
        stages.append(imputer)
    else:
        imputed_num = []

    # ── 2. encode categoricals ────────────────────────────────────────────────
    cat_ohe_cols = []
    for col in categorical_cols:
        idx_col = f"{col}_idx"
        ohe_col = f"{col}_ohe"
        indexer = StringIndexer(
            inputCol=col,
            outputCol=idx_col,
            handleInvalid="keep",     # unseen labels → extra index
        )
        encoder = OneHotEncoder(
            inputCols=[idx_col],
            outputCols=[ohe_col],
            handleInvalid="keep",
        )
        stages += [indexer, encoder]
        cat_ohe_cols.append(ohe_col)

    # ── 3. assemble ───────────────────────────────────────────────────────────
    assembler_inputs = imputed_num + cat_ohe_cols
    raw_vec_col = "raw_features"
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol=raw_vec_col,
        handleInvalid="keep",
    )
    stages.append(assembler)

    # ── 4. scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler(
        inputCol=raw_vec_col,
        outputCol=output_col,
        withMean=True,
        withStd=True,
    )
    stages.append(scaler)

    return Pipeline(stages=stages)


# ── convenience wrapper ───────────────────────────────────────────────────────

def preprocess(df: DataFrame, id_col: str = "user") -> tuple[DataFrame, list, list]:
    """
    Detect column types, fit the pipeline, and return
    (transformed_df, numerical_cols, categorical_cols).
    """
    numerical_cols, categorical_cols = detect_column_types(df, id_col=id_col)

    pipeline = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    model = pipeline.fit(df)
    transformed = model.transform(df)

    return transformed, numerical_cols, categorical_cols
