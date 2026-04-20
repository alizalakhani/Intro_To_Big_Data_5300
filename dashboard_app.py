import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CERT Insider Threat — Interactive Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0F1117; color: #E2E8F0; }
  [data-testid="stSidebar"] { background: #1A1D2E; }
  .metric-card {
    background: #1E2235; border-radius: 12px; padding: 20px 24px;
    border-left: 4px solid; margin-bottom: 8px;
  }
  .metric-card.red   { border-color: #EF4444; }
  .metric-card.amber { border-color: #F59E0B; }
  .metric-card.blue  { border-color: #3B82F6; }
  .metric-card.green { border-color: #10B981; }
  .metric-val { font-size: 2rem; font-weight: 700; }
  .metric-lbl { font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }
  .section-header {
    font-size: 1.0rem; font-weight: 600; color: #CBD5E1;
    margin: 1.5rem 0 0.6rem; text-transform: uppercase; letter-spacing: 0.08em;
  }
  hr { border-color: #2D3748; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; }
  .best-k-badge {
    background: linear-gradient(135deg, #7C3AED, #A78BFA);
    color: white; padding: 6px 16px; border-radius: 999px;
    font-weight: 700; font-size: 1rem; display: inline-block;
  }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_final_model():
    return pd.read_csv("final_model_comparison.csv")

@st.cache_data
def load_silhouette_vs_k():
    return pd.read_csv("silhouette_comparison_only.csv")

@st.cache_data
def load_top_anomalies():
    return pd.read_csv("top_anomalies.csv")

# Rebuild silhouette-vs-k table from notebook cell b2653218
K_RANGE = list(range(2, 9))
SPARK_K_RESULTS = {
    2: {"silhouette": 0.387720, "training_cost": 2103.536950, "iterations": 3},
    3: {"silhouette": 0.557892, "training_cost": 1224.442133, "iterations": 5},
    4: {"silhouette": 0.572322, "training_cost": 1041.457561, "iterations": 18},
    5: {"silhouette": 0.582903, "training_cost": 845.246647,  "iterations": 20},
    6: {"silhouette": 0.554975, "training_cost": 718.327776,  "iterations": 15},
    7: {"silhouette": 0.563645, "training_cost": 647.185795,  "iterations": 13},
    8: {"silhouette": 0.572921, "training_cost": 601.444670,  "iterations": 12},
}
sil_df = pd.DataFrame([
    {"k": k, "silhouette": v["silhouette"], "training_cost": v["training_cost"], "iterations": v["iterations"]}
    for k, v in SPARK_K_RESULTS.items()
])

# Rebuild cluster profiles from notebook analysis
# Features used: total_emails, total_recipients, mean_recipients, external_recipients,
# off_hours_emails, weekend_emails, avg_hour, avg_email_size, unique_pcs,
# external_ratio, off_hours_ratio, weekend_ratio
FEATURE_COLS = [
    "total_emails", "total_recipients", "mean_recipients",
    "external_recipients", "off_hours_emails", "weekend_emails",
    "avg_hour", "avg_email_size", "unique_pcs",
    "external_ratio", "off_hours_ratio", "weekend_ratio"
]
FEATURE_LABELS = [
    "Total Emails", "Total Recipients", "Mean Recipients",
    "External Recipients", "Off-Hours Emails", "Weekend Emails",
    "Avg Hour", "Avg Email Size", "Unique PCs",
    "External Ratio", "Off-Hours Ratio", "Weekend Ratio"
]

# Reconstruct cluster centroid profiles from top_anomalies.csv and final_model_comparison.csv
# Cluster assignments: 0=Normal, 1=Off-Hours Pattern, 2=Weekend Activity, 3=Anomaly, 4=High External
anomaly_df = load_top_anomalies()
model_df = load_final_model()

# Build per-cluster centroid fingerprints from the features column
def parse_features(features_str):
    import ast
    return np.array(ast.literal_eval(features_str))

# Use top_anomalies to derive cluster centroids
clusterCentroids = {}
for cluster_id in sorted(anomaly_df["custom_cluster"].unique()):
    cluster_rows = anomaly_df[anomaly_df["custom_cluster"] == cluster_id]
    centroid = np.zeros(len(FEATURE_COLS))
    for _, row in cluster_rows.iterrows():
        centroid += parse_features(row["features"])
    centroid /= len(cluster_rows)
    clusterCentroids[cluster_id] = centroid

CLUSTER_NAMES = {
    0: "Normal Baseline",
    1: "Off-Hours Pattern",
    2: "Weekend Activity",
    3: "Elevated Suspicion",
    4: "High-Risk Exfiltrators",
}

CLUSTER_COLORS = {
    "High-Risk Exfiltrators": "#EF4444",
    "Elevated Suspicion":     "#F59E0B",
    "Weekend Activity":       "#8B5CF6",
    "Off-Hours Pattern":      "#3B82F6",
    "Normal Baseline":        "#10B981",
}

# Normalize centroids to [0,1] for radar chart
all_centroids = np.vstack(list(clusterCentroids.values()))
min_vals = all_centroids.min(axis=0)
max_vals = all_centroids.max(axis=0)
ranges = max_vals - min_vals
ranges[ranges == 0] = 1
norm_centroids = {}
for cid, cent in clusterCentroids.items():
    norm_centroids[cid] = (cent - min_vals) / ranges

# ── Helper: plotly dark template ─────────────────────────────────────────────
def dark_layout(fig, **kwargs):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,17,23,0.8)",
        font_color="#E2E8F0",
        **kwargs
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — K SELECTION LAB
# ══════════════════════════════════════════════════════════════════════════════
def page1_k_selection():
    st.markdown("# K Selection Lab")
    st.markdown("*How do we know k=5 is the right number of clusters? Let's ask the data.*")

    col_intro, col_metrics = st.columns([2, 1])
    with col_intro:
        st.markdown(
            "The **elbow method** plots training cost vs. k — the bend ('elbow') marks where adding "
            "more clusters yields diminishing returns. The **silhouette score** measures how well "
            "points are assigned to their own cluster vs. neighbouring clusters. "
            "Higher is better (max = 1)."
        )
    with col_metrics:
        best_row = sil_df.loc[sil_df["silhouette"].idxmax()]
        st.markdown(
            f"<div style='text-align:center; margin-top:8px'>"
            f"<div class='best-k-badge'>★ Best k = {int(best_row['k'])}</div><br>"
            f"<div style='color:#94A3B8;font-size:0.85rem'>Silhouette Score</div>"
            f"<div style='color:#A78BFA;font-size:1.8rem;font-weight:700'>{best_row['silhouette']:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Slider for k
    selected_k = st.slider(
        "Select number of clusters (k)",
        min_value=2, max_value=8, value=5, step=1
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Silhouette Score vs. k</div>', unsafe_allow_html=True)
        fig_sil = go.Figure()
        best_k_row = sil_df.loc[sil_df["silhouette"].idxmax()]
        colors = ["#A78BFA" if k == int(best_k_row["k"]) else "#3B82F6" for k in sil_df["k"]]

        fig_sil.add_trace(go.Scatter(
            x=sil_df["k"], y=sil_df["silhouette"],
            mode="lines+markers", marker=dict(size=12),
            line=dict(color="#3B82F6", width=2.5),
            text=[f"k={int(r['k'])}<br>Silhouette={r['silhouette']:.4f}" for _, r in sil_df.iterrows()],
            hovertemplate="%{text}<extra></extra>",
        ))

        # Highlight selected k
        sel_sil = sil_df.loc[sil_df["k"] == selected_k, "silhouette"].values[0]
        fig_sil.add_trace(go.Scatter(
            x=[selected_k], y=[sel_sil],
            mode="markers", marker=dict(size=18, color="#EF4444", symbol="star"),
        ))
        fig_sil.add_hline(
            y=sil_df["silhouette"].mean(), line_dash="dot", line_color="#64748B",
            annotation_text="Mean silhouette", annotation_position="bottom right"
        )
        dark_layout(fig_sil, height=340,
            xaxis=dict(title="k (number of clusters)", tickmode="linear", tick0=2, dtick=1),
            yaxis=dict(title="Silhouette Score", range=[0.3, 0.7]),
            margin=dict(t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Elbow Curve (Training Cost)</div>', unsafe_allow_html=True)
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=sil_df["k"], y=sil_df["training_cost"],
            mode="lines+markers", marker=dict(size=12),
            line=dict(color="#F59E0B", width=2.5),
            text=[f"k={int(r['k'])}<br>Cost={r['training_cost']:,.0f}" for _, r in sil_df.iterrows()],
            hovertemplate="%{text}<extra></extra>",
        ))
        # Highlight selected k
        sel_cost = sil_df.loc[sil_df["k"] == selected_k, "training_cost"].values[0]
        fig_elbow.add_trace(go.Scatter(
            x=[selected_k], y=[sel_cost],
            mode="markers", marker=dict(size=18, color="#EF4444", symbol="star"),
        ))
        dark_layout(fig_elbow, height=340,
            xaxis=dict(title="k (number of clusters)", tickmode="linear", tick0=2, dtick=1),
            yaxis=dict(title="Training Cost (Within-cluster SSE)", range=[500, 2300]),
            margin=dict(t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    # Info box
    selected_row = sil_df.loc[sil_df["k"] == selected_k].iloc[0]
    is_best = (selected_k == int(best_k_row["k"]))
    badge = "✅ BEST K" if is_best else f"( Best = k={int(best_k_row['k'])} )"
    st.markdown(
        f"**k = {selected_k}** — Silhouette: `{selected_row['silhouette']:.4f}` | "
        f"Training Cost: `{selected_row['training_cost']:,.0f}` | Iterations: `{int(selected_row['iterations'])}`"
        f" &nbsp;&nbsp;{badge}",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def page2_model_comparison():
    st.markdown("# Model Comparison")
    st.markdown(
        "**SparkKMeans vs. CustomKMeans vs. DBSCAN vs. HDBSCAN** — "
        "an honest, academic head-to-head on the CERT email data. "
        "No cherry-picking: all four algorithms are measured on the same 398 user-day records."
    )

    model_df = load_final_model()

    # Clean up runtime display
    model_df["runtime_disp"] = model_df["runtime_sec"].apply(
        lambda x: f"{x*1000:.1f} ms" if x < 1 else f"{x:.2f} s"
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown('<div class="section-header">Runtime (ms) — lower is better</div>', unsafe_allow_html=True)
        rt_colors = ["#EF4444" if m in ["HDBSCAN", "DBSCAN"] else "#3B82F6" for m in model_df["model"]]
        fig_rt = go.Figure()
        fig_rt.add_trace(go.Bar(
            x=model_df["model"], y=model_df["runtime_sec"] * 1000,
            marker_color=rt_colors,
            text=[f"{v*1000:.1f} ms" for v in model_df["runtime_sec"]],
            textposition="outside",
        ))
        dark_layout(fig_rt, height=300, margin=dict(t=10, b=10),
            yaxis=dict(title="Runtime (ms)", type="log"),
            xaxis=dict(title=""))
        st.plotly_chart(fig_rt, use_container_width=True)

    with col_m2:
        st.markdown('<div class="section-header">Silhouette Score — higher is better</div>', unsafe_allow_html=True)
        sil_colors = ["#A78BFA" if m == "SparkKMeans" else "#3B82F6" for m in model_df["model"]]
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=model_df["model"], y=model_df["silhouette"],
            marker_color=sil_colors,
            text=[f"{v:.4f}" for v in model_df["silhouette"]],
            textposition="outside",
        ))
        fig_sil.add_hline(y=0.5, line_dash="dot", line_color="#64748B",
            annotation_text="0.5 threshold", annotation_position="bottom right")
        dark_layout(fig_sil, height=300, margin=dict(t=10, b=10),
            yaxis=dict(title="Silhouette Score", range=[0, 0.7]),
            xaxis=dict(title=""))
        st.plotly_chart(fig_sil, use_container_width=True)

    # Runtime vs Silhouette scatter
    st.markdown('<div class="section-header">Runtime vs. Silhouette — the quality/speed tradeoff</div>', unsafe_allow_html=True)
    fig_sc = go.Figure()
    for _, row in model_df.iterrows():
        size = 18 if row["model"] == "SparkKMeans" else 12
        fig_sc.add_trace(go.Scatter(
            x=[row["runtime_sec"] * 1000], y=[row["silhouette"]],
            mode="markers+text",
            marker=dict(size=size, color=CLUSTER_COLORS.get(row["model"], "#3B82F6")),
            text=[row["model"]],
            textposition="top center",
            textfont=dict(color="#E2E8F0", size=12),
            hovertemplate=(
                f"<b>{{text}}</b><br>"
                f"Silhouette: {{y:.4f}}<br>"
                f"Runtime: {row['runtime_sec']*1000:.1f} ms<br>"
                f"k={int(row['k'])}<extra></extra>"
            ),
        ))
    fig_sc.update_traces(
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Silhouette: %{y:.4f}<br>"
            "Runtime: %{x:.1f} ms<extra></extra>"
        )
    )
    dark_layout(fig_sc, height=340, margin=dict(t=10, b=10),
        xaxis=dict(title="Runtime (ms, log scale)", type="log"),
        yaxis=dict(title="Silhouette Score"),
        showlegend=False)
    st.plotly_chart(fig_sc, use_container_width=True)

    # Metrics table
    st.markdown("---")
    st.markdown('<div class="section-header">Full Metrics Table</div>', unsafe_allow_html=True)
    disp = model_df.copy()
    disp["k"] = disp["k"].astype(int)
    disp = disp.rename(columns={
        "model": "Model", "k": "k", "silhouette": "Silhouette",
        "training_cost": "Training Cost", "iterations": "Iterations",
        "runtime_disp": "Runtime", "davies_bouldin": "Davies-Bouldin",
        "calinski_harabasz": "Calinski-Harabasz"
    })
    st.dataframe(
        disp[["Model", "k", "Silhouette", "Training Cost", "Iterations", "Runtime",
              "Davies-Bouldin", "Calinski-Harabasz"]]
        .style.background_gradient(subset=["Silhouette"], cmap="Purples")
        .format({
            "Silhouette": "{:.4f}",
            "Training Cost": "{:.1f}",
            "Iterations": "{:.0f}",
            "Davies-Bouldin": "{:.4f}",
            "Calinski-Harabasz": "{:.2f}",
        }, na_rep="—"),
        use_container_width=True, height=280
    )

    st.markdown(
        "**Winner: SparkKMeans** (silhouette=0.5829) — but HDBSCAN runs **1,900× faster** "
        "and still achieves strong separation (0.5343). "
        "DBSCAN produces the most compact clusters (lowest Davies-Bouldin=0.857)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTER PROFILES
# ══════════════════════════════════════════════════════════════════════════════
def page3_cluster_profiles():
    st.markdown("# Cluster Profiles")
    st.markdown(
        "*What does each cluster actually mean?* Each radar chart shows a cluster's "
        "centroid fingerprint across all 12 behavioral features. "
        "Outer = higher value. The bar chart below shows which features drive cluster separation."
    )

    selected_cluster_view = st.selectbox(
        "View cluster profile",
        ["All Clusters", "High-Risk Exfiltrators", "Elevated Suspicion",
         "Weekend Activity", "Off-Hours Pattern", "Normal Baseline"]
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">Radar Chart — Centroid Feature Fingerprint</div>', unsafe_allow_html=True)

        if selected_cluster_view == "All Clusters":
            fig_rad = go.Figure()
            for cid, cent in norm_centroids.items():
                cname = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
                color = CLUSTER_COLORS.get(cname, "#3B82F6")
                fig_rad.add_trace(go.Scatterpolar(
                    r=list(cent) + [cent[0]],
                    theta=FEATURE_LABELS + [FEATURE_LABELS[0]],
                    mode="lines",
                    name=cname,
                    line=dict(color=color, width=2),
                    opacity=0.7,
                    hovertemplate="%{theta}<br>%{r:.3f}<extra></extra>",
                ))
            dark_layout(fig_rad, height=440, margin=dict(t=10, b=10),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], color="#64748B"),
                    angularaxis=dict(color="#CBD5E1", tickangle=25),
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.2, font_size=11),
            )
        else:
            cid_map = {v: k for k, v in CLUSTER_NAMES.items()}
            cid = cid_map[selected_cluster_view]
            cent = norm_centroids[cid]
            color = CLUSTER_COLORS.get(selected_cluster_view, "#3B82F6")
            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=list(cent) + [cent[0]],
                theta=FEATURE_LABELS + [FEATURE_LABELS[0]],
                mode="lines+markers",
                fill="toself",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.25)",
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                hovertemplate="%{theta}<br>%{r:.3f}<extra></extra>",
            ))
            dark_layout(fig_rad, height=440, margin=dict(t=10, b=10),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], color="#64748B"),
                    angularaxis=dict(color="#CBD5E1", tickangle=25),
                ),
                showlegend=False,
            )
        st.plotly_chart(fig_rad, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Feature Separation — Which Features Drive Cluster Boundaries?</div>', unsafe_allow_html=True)

        # Compute feature importance: ratio of between-cluster variance to total variance
        raw_centroids = np.vstack(list(clusterCentroids.values()))
        cluster_means = raw_centroids.mean(axis=0)
        between_var = ((raw_centroids - cluster_means) ** 2).mean(axis=0)
        total_var = raw_centroids.var(axis=0)
        sep_score = between_var / (total_var + 1e-10)

        feat_importance = pd.DataFrame({
            "Feature": FEATURE_LABELS,
            "Separation Score": sep_score,
        }).sort_values("Separation Score", ascending=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=feat_importance["Separation Score"],
            y=feat_importance["Feature"],
            orientation="h",
            marker=dict(
                color=feat_importance["Separation Score"],
                colorscale="Plasma",
            ),
            text=[f"{v:.3f}" for v in feat_importance["Separation Score"]],
            textposition="outside",
        ))
        dark_layout(fig_bar, height=440, margin=dict(t=10, b=10),
            xaxis=dict(title="Separation Score (higher = better cluster separation)"),
            yaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Cluster summary table
    st.markdown("---")
    st.markdown('<div class="section-header">Cluster Centroid Summary</div>', unsafe_allow_html=True)
    summary_rows = []
    for cid, cent in clusterCentroids.items():
        cname = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
        summary_rows.append({
            "Cluster": cname,
            "Top Feature 1": FEATURE_LABELS[np.argmax(cent)],
            "Top Feature 2": FEATURE_LABELS[np.argsort(cent)[-2]],
            "External Ratio": f"{cent[FEATURE_LABELS.index('External Ratio')]:.3f}",
            "Off-Hours Ratio": f"{cent[FEATURE_LABELS.index('Off-Hours Ratio')]:.3f}",
            "Weekend Ratio": f"{cent[FEATURE_LABELS.index('Weekend Ratio')]:.3f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df.style.map(
            lambda val: f"color:{CLUSTER_COLORS.get(val, '#E2E8F0')}" if val in CLUSTER_COLORS else "",
            subset=["Cluster"]
        ),
        use_container_width=True, height=260
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANOMALY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def page4_anomaly_explorer():
    st.markdown("# Anomaly Explorer")
    st.markdown(
        "Anomaly score = distance from cluster centroid (CustomKMeans, k=5). "
        "Higher score = more unusual behavioral pattern. "
        "Use the **catch-rate slider** to see how many users get flagged at each threshold."
    )

    anomaly_df = load_top_anomalies()

    # Top anomalies bar chart
    st.markdown('<div class="section-header">Top Anomalous User-Day Records</div>', unsafe_allow_html=True)
    top_n = 15
    top_n_df = anomaly_df.head(top_n).copy()
    top_n_df["label"] = top_n_df["user"].astype(str) + " | " + top_n_df["day"].astype(str)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=top_n_df["anomaly_score"],
        y=top_n_df["label"],
        orientation="h",
        marker=dict(
            color=top_n_df["anomaly_score"],
            colorscale="Reds",
            cmin=top_n_df["anomaly_score"].min(),
            cmax=top_n_df["anomaly_score"].max(),
        ),
        text=[f"{v:.2f}" for v in top_n_df["anomaly_score"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Anomaly Score: %{x:.4f}<br>"
            "Emails: " + top_n_df["total_emails"].astype(str) + "<br>"
            "External Ratio: " + top_n_df["external_ratio"].apply(lambda v: f"{v:.2%}") + "<br>"
            "<extra></extra>"
        ),
    ))
    dark_layout(fig_bar, height=400, margin=dict(t=10, b=10),
        xaxis=dict(title="Anomaly Score (distance from centroid)"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    col_cal, col_thresh = st.columns([1.2, 1])

    with col_cal:
        st.markdown('<div class="section-header">User-Day Timeline Heatmap</div>', unsafe_allow_html=True)
        anomaly_df["day"] = pd.to_datetime(anomaly_df["day"])
        anomaly_df["month"] = anomaly_df["day"].dt.to_period("M")
        anomaly_df["weekday"] = anomaly_df["day"].dt.day_name()

        heatmap_data = anomaly_df.groupby(["month", "weekday"])["anomaly_score"].mean().reset_index()
        heatmap_data["month"] = heatmap_data["month"].astype(str)
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data = heatmap_data[heatmap_data["weekday"].isin(weekday_order)]

        pivot = heatmap_data.pivot(index="weekday", columns="month", values="anomaly_score")
        pivot = pivot.reindex(index=weekday_order)

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="YlOrRd",
            colorbar=dict(title="Avg Anomaly Score"),
            hovertemplate=" %{y} %{x}<br>Score: %{z:.3f}<extra></extra>",
        ))
        dark_layout(fig_heat, height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_thresh:
        st.markdown('<div class="section-header">Catch Rate Threshold Slider</div>', unsafe_allow_html=True)
        all_records = load_top_anomalies()
        threshold = st.slider(
            "Flag users with anomaly score ≥",
            min_value=float(all_records["anomaly_score"].min()),
            max_value=float(all_records["anomaly_score"].max()),
            value=2.5,
            step=0.1,
        )
        flagged = all_records[all_records["anomaly_score"] >= threshold]
        catch_rate = len(flagged) / len(all_records) * 100
        total_users = all_records["user"].nunique()
        flagged_users = flagged["user"].nunique()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-card red">
              <div class="metric-val" style="color:#EF4444">{len(flagged)}</div>
              <div class="metric-lbl">Records Flagged</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card amber">
              <div class="metric-val" style="color:#F59E0B">{flagged_users}/{total_users}</div>
              <div class="metric-lbl">Users Flagged</div></div>""", unsafe_allow_html=True)

        st.markdown(
            f"**{catch_rate:.1f}%** of user-day records flagged at threshold ≥ `{threshold:.2f}`"
        )

        # CDF curve of catch rate
        all_scores = sorted(all_records["anomaly_score"].unique())
        thresholds = np.linspace(all_records["anomaly_score"].min(),
                                 all_records["anomaly_score"].max(), 100)
        catch_pct = [(all_records["anomaly_score"] >= t).sum() / len(all_records) * 100
                     for t in thresholds]

        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=thresholds, y=catch_pct,
            mode="lines", line=dict(color="#3B82F6", width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
            hovertemplate="Threshold: %{x:.2f}<br>Catch Rate: %{y:.1f}%<extra></extra>",
        ))
        fig_cdf.add_vline(x=threshold, line_dash="dot", line_color="#EF4444", line_width=2)
        dark_layout(fig_cdf, height=260, margin=dict(t=10, b=10),
            xaxis=dict(title="Anomaly Score Threshold"),
            yaxis=dict(title="Records Caught (%)", range=[0, 100]),
            showlegend=False,
        )
        st.plotly_chart(fig_cdf, use_container_width=True)

    # Bottom: full anomaly table
    st.markdown("---")
    st.markdown('<div class="section-header">All Flagged User-Day Records</div>', unsafe_allow_html=True)
    disp_cols = ["user", "day", "anomaly_score", "total_emails", "external_ratio",
                 "off_hours_ratio", "weekend_ratio", "avg_email_size", "custom_cluster"]
    st.dataframe(
        flagged[disp_cols].style
            .background_gradient(subset=["anomaly_score"], cmap="Reds")
            .format({"anomaly_score": "{:.4f}", "external_ratio": "{:.2%}",
                     "off_hours_ratio": "{:.2%}", "weekend_ratio": "{:.2%}",
                     "avg_email_size": "{:,.0f}"}, na_rep="—"),
        use_container_width=True, height=350
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — page navigation via radio buttons
# ══════════════════════════════════════════════════════════════════════════════
PAGES = {
    "🔬 K Selection Lab":        page1_k_selection,
    "⚖️ Model Comparison":       page2_model_comparison,
    "📊 Cluster Profiles":       page3_cluster_profiles,
    "🚨 Anomaly Explorer":       page4_anomaly_explorer,
}

st.markdown("## 🛡️ CERT Insider Threat — Interactive Report")
active_page = st.radio("", list(PAGES.keys()), index=0, label_visibility="collapsed")
PAGES[active_page]()
