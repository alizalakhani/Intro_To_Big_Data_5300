import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Page config
st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom KMeans implementation
class CustomKMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42, init="kmeans++"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _initialize_centroids(self, X):
        rng = np.random.default_rng(self.random_state)

        if self.init == "random":
            idx = rng.choice(len(X), size=self.n_clusters, replace=False)
            return X[idx].copy()

        centroids = []
        first_idx = rng.integers(0, len(X))
        centroids.append(X[first_idx])

        for _ in range(1, self.n_clusters):
            dists = np.min(
                np.linalg.norm(X[:, None] - np.array(centroids)[None, :], axis=2) ** 2,
                axis=1
            )
            probs = dists / dists.sum()
            next_idx = rng.choice(len(X), p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = []
        rng = np.random.default_rng(self.random_state)
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                new_centroids.append(X[rng.integers(0, len(X))])
            else:
                new_centroids.append(cluster_points.mean(axis=0))
        return np.array(new_centroids)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            self.n_iter_ = i + 1
            if shift < self.tol:
                break

        self.centroids = centroids
        self.labels_ = self._assign_clusters(X, centroids)
        distances = np.linalg.norm(X - centroids[self.labels_], axis=1) ** 2
        self.inertia_ = distances.sum()
        return self

    def predict(self, X):
        return self._assign_clusters(np.asarray(X, dtype=float), self.centroids)

    def anomaly_score(self, X):
        X = np.asarray(X, dtype=float)
        labels = self.predict(X)
        return np.linalg.norm(X - self.centroids[labels], axis=1)


# Initialize Spark session
@st.cache_resource
def get_spark_session():
    return (
        SparkSession.builder
        .appName("CERT-Insider-Threat-Clustering")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "100")
        .getOrCreate()
    )


# Data preprocessing
def preprocess_data(sdf):
    sdf = sdf.withColumn(
        "date_ts",
        F.to_timestamp("date", "MM/dd/yyyy HH:mm:ss")
    ).dropna(subset=["date_ts", "user"])

    for c in ["to", "cc", "bcc", "from", "content"]:
        if c in sdf.columns:
            sdf = sdf.fillna({c: ""})

    for c in ["size", "attachments"]:
        if c in sdf.columns:
            sdf = sdf.fillna({c: 0})

    sdf = (
        sdf.withColumn("day", F.to_date("date_ts"))
           .withColumn("hour", F.hour("date_ts"))
           .withColumn("weekday", F.dayofweek("date_ts"))
           .withColumn("off_hours", F.when((F.col("hour") < 8) | (F.col("hour") > 18), 1).otherwise(0))
           .withColumn("weekend", F.when((F.col("weekday").isin([1, 7])), 1).otherwise(0))
    )

    def extract_mailto(col_name):
        return F.lower(F.regexp_extract(F.col(col_name), r"mailto:([^)]+)", 1))

    sdf = (
        sdf.withColumn("to_email", extract_mailto("to"))
           .withColumn("cc_email", extract_mailto("cc"))
           .withColumn("bcc_email", extract_mailto("bcc"))
           .withColumn(
               "recipient_count",
               (F.when(F.length(F.col("to_email")) > 0, 1).otherwise(0) +
                F.when(F.length(F.col("cc_email")) > 0, 1).otherwise(0) +
                F.when(F.length(F.col("bcc_email")) > 0, 1).otherwise(0))
           )
           .withColumn(
               "external_recipient_count",
               (F.when((F.length(F.col("to_email")) > 0) & (~F.col("to_email").endswith("@dtaa.com")), 1).otherwise(0) +
                F.when((F.length(F.col("cc_email")) > 0) & (~F.col("cc_email").endswith("@dtaa.com")), 1).otherwise(0) +
                F.when((F.length(F.col("bcc_email")) > 0) & (~F.col("bcc_email").endswith("@dtaa.com")), 1).otherwise(0))
           )
    )

    user_day = (
        sdf.groupBy("user", "day")
           .agg(
               F.count("*").alias("total_emails"),
               F.sum("recipient_count").alias("total_recipients"),
               F.avg("recipient_count").alias("mean_recipients"),
               F.sum("external_recipients").alias("external_recipients"),
               F.sum("off_hours").alias("off_hours_emails"),
               F.sum("weekend").alias("weekend_emails"),
               F.avg("hour").alias("avg_hour"),
               F.avg("size").alias("avg_email_size"),
               F.countDistinct("pc").alias("unique_pcs")
           )
           .withColumn("external_ratio", F.col("external_recipients") / F.when(F.col("total_recipients") == 0, 1).otherwise(F.col("total_recipients")))
           .withColumn("off_hours_ratio", F.col("off_hours_emails") / F.when(F.col("total_emails") == 0, 1).otherwise(F.col("total_emails")))
           .withColumn("weekend_ratio", F.col("weekend_emails") / F.when(F.col("total_emails") == 0, 1).otherwise(F.col("total_emails")))
           .fillna(0)
    )

    return user_day


# Feature engineering
def create_features(user_day):
    feature_cols = [
        "total_emails", "total_recipients", "mean_recipients",
        "external_recipients", "off_hours_emails", "weekend_emails",
        "avg_hour", "avg_email_size", "unique_pcs",
        "external_ratio", "off_hours_ratio", "weekend_ratio"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    assembled = assembler.transform(user_day)

    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)
    scaler_model = scaler.fit(assembled)
    spark_data = scaler_model.transform(assembled).select("user", "day", *feature_cols, "features")

    return spark_data, feature_cols


# Main app
def main():
    st.title("🔍 Insider Threat Detection System")
    st.markdown("""
    **Detect anomalous email behavior patterns using clustering algorithms.**

    Upload your email dataset and identify potential insider threats based on:
    - Email volume patterns
    - External recipient ratios
    - Off-hours and weekend activity
    - Device usage patterns
    """)

    # Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    st.sidebar.header("Clustering Settings")
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["Spark KMeans", "Custom KMeans", "DBSCAN", "HDBSCAN"],
        help="Choose clustering algorithm"
    )

    if algorithm in ["Spark KMeans", "Custom KMeans"]:
        n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 5)
    else:
        n_clusters = None

    if algorithm == "DBSCAN":
        eps = st.sidebar.slider("EPS (DBSCAN)", 0.5, 3.0, 1.2, 0.1)
        min_samples = st.sidebar.slider("Min Samples (DBSCAN)", 2, 20, 8)

    if algorithm == "HDBSCAN":
        min_cluster_size = st.sidebar.slider("Min Cluster Size (HDBSCAN)", 5, 50, 15)

    # Main content
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to begin analysis.")
        st.markdown("""
        ### Expected CSV Format
        Your CSV should contain the following columns:
        - `id`: Email identifier
        - `date`: Timestamp (MM/dd/yyyy HH:mm:ss)
        - `user`: User identifier
        - `pc`: Computer identifier
        - `to`, `cc`, `bcc`, `from`: Recipient/sender info
        - `size`: Email size
        - `attachments`: Number of attachments
        - `content`: Email content
        """)
        return

    # Load data
    with st.spinner("Loading data..."):
        spark = get_spark_session()
        sdf = spark.read.csv(uploaded_file, header=True, inferSchema=True)
        raw_count = sdf.count()
        st.success(f"Loaded {raw_count:,} email records")

    # Preprocess
    with st.spinner("Preprocessing data..."):
        user_day = preprocess_data(sdf)
        processed_count = user_day.count()
        st.success(f"Aggregated to {processed_count:,} user-day records")

    # Create features
    with st.spinner("Engineering features..."):
        spark_data, feature_cols = create_features(user_day)
        spark_data = spark_data.cache()

    # Convert to pandas for Python algorithms
    pdf = spark_data.select("user", "day", *feature_cols, "features").toPandas()
    X_scaled = np.vstack(pdf["features"].apply(lambda v: np.array(v.toArray())).values)

    # Run clustering
    st.header("🎯 Clustering Results")
    with st.spinner(f"Running {algorithm}..."):
        start_time = time.time()

        if algorithm == "Spark KMeans":
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            assembled = assembler.transform(user_day)
            scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=False)
            scaler_model = scaler.fit(assembled)
            scaled_data = scaler_model.transform(assembled).select("user", "day", *feature_cols, "features_scaled")

            kmeans = SparkKMeans(featuresCol="features_scaled", predictionCol="prediction", k=n_clusters, seed=42)
            model = kmeans.fit(scaled_data)
            preds = model.transform(scaled_data)
            result_pdf = preds.toPandas()
            labels = result_pdf["prediction"].values
            runtime = time.time() - start_time

            # Calculate metrics
            silhouette = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)

        elif algorithm == "Custom KMeans":
            custom_model = CustomKMeans(n_clusters=n_clusters, random_state=42)
            custom_model.fit(X_scaled)
            labels = custom_model.labels_
            runtime = time.time() - start_time

            silhouette = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)

        elif algorithm == "DBSCAN":
            db_model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db_model.fit_predict(X_scaled)
            runtime = time.time() - start_time

            valid = labels != -1
            if valid.sum() > 10 and len(set(labels[valid])) > 1:
                silhouette = silhouette_score(X_scaled[valid], labels[valid])
                db_score = davies_bouldin_score(X_scaled[valid], labels[valid])
                ch_score = calinski_harabasz_score(X_scaled[valid], labels[valid])
            else:
                silhouette, db_score, ch_score = np.nan, np.nan, np.nan

        elif algorithm == "HDBSCAN":
            hdb_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = hdb_model.fit_predict(X_scaled)
            runtime = time.time() - start_time

            valid = labels != -1
            if valid.sum() > 10 and len(set(labels[valid])) > 1:
                silhouette = silhouette_score(X_scaled[valid], labels[valid])
                db_score = davies_bouldin_score(X_scaled[valid], labels[valid])
                ch_score = calinski_harabasz_score(X_scaled[valid], labels[valid])
            else:
                silhouette, db_score, ch_score = np.nan, np.nan, np.nan

    st.success(f"Completed in {runtime:.2f} seconds")

    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Silhouette Score", f"{silhouette:.3f}" if not np.isnan(silhouette) else "N/A")
    with col2:
        st.metric("Davies-Bouldin", f"{db_score:.3f}" if not np.isnan(db_score) else "N/A")
    with col3:
        st.metric("Calinski-Harabasz", f"{ch_score:.1f}" if not np.isnan(ch_score) else "N/A")
    with col4:
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        st.metric("Clusters Found", n_clusters_found)

    # Add results to dataframe
    pdf["cluster"] = labels

    # Anomaly scores for KMeans
    if algorithm in ["Spark KMeans", "Custom KMeans"]:
        if algorithm == "Custom KMeans":
            pdf["anomaly_score"] = custom_model.anomaly_score(X_scaled)
        else:
            # Calculate distance to assigned centroid
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            assembled = assembler.transform(user_day)
            scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withStd=True, withMean=False)
            scaler_model = scaler.fit(assembled)
            scaled_data = scaler_model.transform(assembled).select("features_scaled")
            X_np = np.array([np.array(v.toArray()) for v in scaled_data.select("features_scaled").collect()])

            centroids = model.clusterCenters()
            cluster_labels = pdf["prediction"].values
            anomaly_scores = []
            for i, cluster in enumerate(cluster_labels):
                anomaly_scores.append(np.linalg.norm(X_np[i] - centroids[cluster]))
            pdf["anomaly_score"] = anomaly_scores

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Visualization", "📈 Cluster Profiles", "⚠️ Anomalies", "📋 Raw Data"])

    with tab1:
        st.subheader("PCA Projection of Clusters")

        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)

        plot_df = pd.DataFrame({
            "PC1": X_2d[:, 0],
            "PC2": X_2d[:, 1],
            "Cluster": labels.astype(str)
        })

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=plot_df,
            x="PC1",
            y="PC2",
            hue="Cluster",
            palette="tab10",
            s=50,
            ax=ax,
            alpha=0.7
        )
        ax.set_title(f"{algorithm} Clusters (PCA Projection)", fontsize=14)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.legend(title="Cluster")
        st.pyplot(fig)

        st.info(f"PCA explains {pca.explained_variance_ratio_.sum():.1%} of total variance with 2 components")

    with tab2:
        st.subheader("Cluster Profile Heatmap")

        cluster_profile = pdf.groupby("cluster")[feature_cols].mean()

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            cluster_profile.T,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Mean Value"}
        )
        plt.title("Cluster Profiles (Mean Feature Values)", fontsize=14)
        plt.xlabel("Cluster")
        plt.ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Feature Distribution by Cluster")
        selected_feature = st.selectbox("Select Feature", feature_cols)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=pdf, x="cluster", y=selected_feature, ax=ax, palette="Set2")
        ax.set_title(f"{selected_feature} Distribution by Cluster", fontsize=14)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(selected_feature)
        st.pyplot(fig)

    with tab3:
        st.subheader("Top Anomalous Records")

        if "anomaly_score" in pdf.columns:
            top_n = st.slider("Number of top anomalies to display", 5, 50, 15)
            anomaly_df = pdf.sort_values("anomaly_score", ascending=False).head(top_n)

            display_cols = ["user", "day", "anomaly_score", "total_emails", "external_ratio", "off_hours_ratio", "weekend_ratio", "cluster"]

            # Bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(
                data=anomaly_df,
                y="user",
                x="anomaly_score",
                hue="cluster",
                dodge=False,
                ax=ax,
                palette="Reds_r",
                legend=False
            )
            ax.set_title(f"Top {top_n} Anomalous User-Day Records", fontsize=14)
            ax.set_xlabel("Anomaly Score (Distance to Cluster Center)")
            ax.set_ylabel("User")
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(anomaly_df[display_cols], hide_index=True)
        else:
            # For DBSCAN/HDBSCAN, show noise points
            noise_points = pdf[pdf["cluster"] == -1]
            if len(noise_points) > 0:
                st.warning(f"Found {len(noise_points)} noise points (unclassified)")
                st.dataframe(noise_points[["user", "day", "total_emails", "external_ratio", "off_hours_ratio"]], hide_index=True)
            else:
                st.info("No noise points found")

    with tab4:
        st.subheader("User-Day Aggregated Data")

        display_df = pdf.copy()
        display_df["day"] = display_df["day"].astype(str)

        st.dataframe(display_df, hide_index=True)

        # Download button
        csv = pdf.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Summary
    st.header("📝 Analysis Summary")
    st.markdown(f"""
    - **Algorithm**: {algorithm}
    - **Records Analyzed**: {processed_count:,} user-day aggregations
    - **Clusters Identified**: {n_clusters_found}
    - **Silhouette Score**: {silhouette:.3f} (higher is better, max 1.0)
    - **Processing Time**: {runtime:.2f} seconds

    **Interpretation**:
    - Silhouette > 0.7: Strong cluster structure
    - Silhouette 0.5-0.7: Reasonable cluster structure
    - Silhouette < 0.5: Weak or overlapping clusters
    """)


if __name__ == "__main__":
    main()
