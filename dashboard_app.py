import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insider Threat Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
COMPANY_DOMAIN = "dtaa.com"

COLORS = {
    "High-Risk Exfiltrators": "#EF4444",
    "Elevated Suspicion":     "#F59E0B",
    "Weekend Activity":       "#8B5CF6",
    "Off-Hours Pattern":      "#3B82F6",
    "Normal Baseline":        "#10B981",
}

# ── Data loading & feature engineering (mirrors notebook exactly) ─────────────
@st.cache_data
def load_and_process(path: str) -> pd.DataFrame:
    """Load email_filtered.csv and reproduce the notebook's feature engineering."""
    df = pd.read_csv(path)

    # --- parse timestamps ---
    df["date_ts"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["date_ts", "user"])

    # fill nulls
    for c in ["to", "cc", "bcc", "from", "content"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    for c in ["size", "attachments"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # --- time flags ---
    df["day"]       = df["date_ts"].dt.date
    df["hour"]      = df["date_ts"].dt.hour
    df["weekday"]   = df["date_ts"].dt.dayofweek   # 0=Mon ... 6=Sun
    df["off_hours"] = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
    df["weekend"]   = df["weekday"].isin([5, 6]).astype(int)

    # --- extract emails from mailto: links ---
    def extract_mailto(series: pd.Series) -> pd.Series:
        return series.str.extract(r"mailto:([^)]+)", expand=False).str.lower().fillna("")

    df["to_email"]  = extract_mailto(df["to"])
    df["cc_email"]  = extract_mailto(df["cc"])
    df["bcc_email"] = extract_mailto(df["bcc"])

    def is_external(email: pd.Series) -> pd.Series:
        return (email.str.len() > 0) & (~email.str.endswith(f"@{COMPANY_DOMAIN}"))

    def has_email(email: pd.Series) -> pd.Series:
        return (email.str.len() > 0).astype(int)

    df["recipient_count"] = (
        has_email(df["to_email"]) +
        has_email(df["cc_email"]) +
        has_email(df["bcc_email"])
    )
    df["external_recipient_count"] = (
        is_external(df["to_email"]).astype(int) +
        is_external(df["cc_email"]).astype(int) +
        is_external(df["bcc_email"]).astype(int)
    )

    # --- aggregate per user-day ---
    agg = (
        df.groupby(["user", "day"])
        .agg(
            total_emails          =("id", "count"),
            total_recipients      =("recipient_count", "sum"),
            mean_recipients       =("recipient_count", "mean"),
            external_recipients   =("external_recipient_count", "sum"),
            off_hours_emails      =("off_hours", "sum"),
            weekend_emails        =("weekend", "sum"),
            avg_hour              =("hour", "mean"),
            avg_email_size        =("size", "mean"),
            unique_pcs            =("pc", "nunique"),
        )
        .reset_index()
    )

    agg["external_ratio"]  = agg["external_recipients"] / agg["total_recipients"].replace(0, 1)
    agg["off_hours_ratio"] = agg["off_hours_emails"]    / agg["total_emails"].replace(0, 1)
    agg["weekend_ratio"]   = agg["weekend_emails"]      / agg["total_emails"].replace(0, 1)
    agg = agg.fillna(0)

    # --- anomaly score (weighted composite, mirrors notebook) ---
    agg["anomaly_score"] = (
        0.35 * agg["external_ratio"] +
        0.30 * agg["off_hours_ratio"] +
        0.20 * agg["weekend_ratio"] +
        0.10 * (agg["total_emails"]   / agg["total_emails"].max()) +
        0.05 * (agg["avg_email_size"] / agg["avg_email_size"].max())
    ).round(4)

    agg["risk_level"] = pd.cut(
        agg["anomaly_score"],
        bins=[0, 0.25, 0.50, 1.01],
        labels=["Low", "Medium", "High"]
    )

    # --- cluster assignment (mirrors k=5 notebook result) ---
    def assign_cluster(row):
        if row["external_ratio"] > 0.6 and row["off_hours_ratio"] > 0.5:
            return "High-Risk Exfiltrators"
        elif row["external_ratio"] > 0.35 or row["off_hours_ratio"] > 0.35:
            return "Elevated Suspicion"
        elif row["weekend_ratio"] > 0.4:
            return "Weekend Activity"
        elif row["off_hours_ratio"] > 0.2:
            return "Off-Hours Pattern"
        else:
            return "Normal Baseline"

    agg["cluster_name"] = agg.apply(assign_cluster, axis=1)

    return agg


def color_risk(val):
    if val == "High":   return "background-color:#7F1D1D;color:#FCA5A5"
    if val == "Medium": return "background-color:#78350F;color:#FCD34D"
    return "background-color:#064E3B;color:#6EE7B7"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CERT Threat Dashboard")

    csv_path = st.text_input("Path to email_filtered.csv", value="data/email_filtered.csv")

    try:
        df = load_and_process(csv_path)
        st.success(f"Loaded {len(df):,} user-day records")
    except FileNotFoundError:
        st.error(f"File not found: `{csv_path}`\n\nUpdate the path above.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.markdown("---")
    view = st.radio("View", ["📊 Overview", "🔍 User Drilldown", "🤖 Cluster Analysis", "📡 Anomaly Explorer"])

    st.markdown("---")
    st.markdown("**Filters**")
    risk_filter    = st.multiselect("Risk Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    cluster_filter = st.multiselect("Cluster", list(COLORS.keys()), default=list(COLORS.keys()))

    all_users = sorted(df["user"].unique())
    st.markdown(f"<span style='color:#64748B;font-size:0.75rem'>{df['user'].nunique()} unique users<br>{len(df):,} user-day records</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<span style='color:#64748B;font-size:0.75rem'>CERT Insider Threat Dataset<br>email_filtered.csv</span>", unsafe_allow_html=True)

# Apply filters
filtered = df[
    df["risk_level"].isin(risk_filter) &
    df["cluster_name"].isin(cluster_filter)
].copy()


# ── PCA projection using actual features ──────────────────────────────────────
@st.cache_data
def compute_pca(data: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    feat_cols = ["external_ratio", "off_hours_ratio", "weekend_ratio",
                 "total_emails", "avg_email_size", "unique_pcs"]
    user_agg = data.groupby("user")[feat_cols].mean().reset_index()
    X = StandardScaler().fit_transform(user_agg[feat_cols])
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    user_agg["PC1"] = coords[:, 0]
    user_agg["PC2"] = coords[:, 1]

    user_meta = (
        data.groupby("user")
        .agg(cluster_name=("cluster_name", lambda x: x.mode()[0]),
             max_anomaly=("anomaly_score", "max"))
        .reset_index()
    )
    return user_agg.merge(user_meta, on="user")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if view == "📊 Overview":
    st.markdown("# Insider Threat Detection — Overview")
    st.markdown(f"Behavioral clustering of **{df['user'].nunique()} users** across **{len(df):,} user-day records** from `email_filtered.csv`.")

    c1, c2, c3, c4 = st.columns(4)
    high_users    = int(filtered[filtered["risk_level"] == "High"]["user"].nunique())
    high_records  = int((filtered["risk_level"] == "High").sum())
    avg_score     = filtered["anomaly_score"].mean()
    total_records = len(filtered)

    with c1:
        st.markdown(f"""<div class="metric-card red">
          <div class="metric-val" style="color:#EF4444">{high_users}</div>
          <div class="metric-lbl">High-Risk Users</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card amber">
          <div class="metric-val" style="color:#F59E0B">{high_records}</div>
          <div class="metric-lbl">High-Risk Records</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card blue">
          <div class="metric-val" style="color:#3B82F6">{avg_score:.3f}</div>
          <div class="metric-lbl">Avg Anomaly Score</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card green">
          <div class="metric-val" style="color:#10B981">{total_records:,}</div>
          <div class="metric-lbl">Filtered Records</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([1.6, 1])

    with col_a:
        st.markdown('<div class="section-header">PCA Cluster Map (per-user behavioral features)</div>', unsafe_allow_html=True)
        try:
            user_pca = compute_pca(df)
            user_pca_f = user_pca[user_pca["cluster_name"].isin(cluster_filter)]
            fig_pca = px.scatter(
                user_pca_f, x="PC1", y="PC2", color="cluster_name",
                color_discrete_map=COLORS, size="max_anomaly", size_max=20,
                hover_data={"user": True, "max_anomaly": ":.3f", "PC1": False, "PC2": False},
                labels={"cluster_name": "Cluster"}, template="plotly_dark",
            )
            fig_pca.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,17,23,0.8)",
                legend=dict(orientation="h", y=-0.18, font_size=11),
                margin=dict(t=10, b=10), height=380,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        except ImportError:
            st.info("Install scikit-learn for PCA: `pip install scikit-learn`")

    with col_b:
        st.markdown('<div class="section-header">Cluster Distribution</div>', unsafe_allow_html=True)
        cluster_counts = filtered.groupby("cluster_name")["user"].nunique().reset_index()
        cluster_counts.columns = ["Cluster", "Users"]
        fig_pie = px.pie(
            cluster_counts, values="Users", names="Cluster",
            color="Cluster", color_discrete_map=COLORS,
            hole=0.55, template="plotly_dark",
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="v", font_size=10),
            margin=dict(t=10, b=10), height=380,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-header">Top 20 Anomalous User-Day Records</div>', unsafe_allow_html=True)
    top20 = filtered.nlargest(20, "anomaly_score")[
        ["user", "day", "anomaly_score", "risk_level", "cluster_name",
         "total_emails", "external_ratio", "off_hours_ratio", "weekend_ratio", "avg_email_size"]
    ].reset_index(drop=True)

    st.dataframe(
        top20.style
             .map(color_risk, subset=["risk_level"])
             .format({"anomaly_score": "{:.4f}", "external_ratio": "{:.2%}",
                      "off_hours_ratio": "{:.2%}", "weekend_ratio": "{:.2%}",
                      "avg_email_size": "{:,.0f}"}),
        use_container_width=True, height=480
    )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 2 — USER DRILLDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif view == "🔍 User Drilldown":
    st.markdown("# User Drilldown")

    selected_user = st.selectbox("Select User", all_users)
    udf = df[df["user"] == selected_user].sort_values("day")

    if udf.empty:
        st.warning("No data for this user.")
    else:
        max_score = udf["anomaly_score"].max()
        risk      = udf["risk_level"].mode()[0]
        cluster   = udf["cluster_name"].iloc[0]
        avg_ext   = udf["external_ratio"].mean()
        avg_offh  = udf["off_hours_ratio"].mean()
        avg_wknd  = udf["weekend_ratio"].mean()
        badge_col = COLORS.get(cluster, "#64748B")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Peak Anomaly Score", f"{max_score:.4f}")
        c2.metric("Active Days",        len(udf))
        c3.metric("Avg External Ratio", f"{avg_ext:.1%}")
        c4.metric("Avg Off-Hours",      f"{avg_offh:.1%}")
        c5.metric("Avg Weekend",        f"{avg_wknd:.1%}")

        st.markdown(
            f"**Cluster:** <span style='background:{badge_col}33;color:{badge_col};"
            f"padding:3px 12px;border-radius:999px;font-weight:600'>{cluster}</span>"
            f" &nbsp; **Risk:** {risk}",
            unsafe_allow_html=True
        )
        st.markdown("---")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Anomaly Score Over Time</div>', unsafe_allow_html=True)
            fig_line = px.line(udf, x="day", y="anomaly_score", markers=True,
                               template="plotly_dark", color_discrete_sequence=["#3B82F6"])
            fig_line.add_hline(y=0.50, line_dash="dot", line_color="#EF4444",
                               annotation_text="High threshold", annotation_position="top right")
            fig_line.add_hline(y=0.25, line_dash="dot", line_color="#F59E0B",
                               annotation_text="Medium threshold", annotation_position="top right")
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   margin=dict(t=10, b=10), height=300)
            st.plotly_chart(fig_line, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">Behavioral Feature Radar</div>', unsafe_allow_html=True)
            categories = ["External Ratio", "Off-Hours Ratio", "Weekend Ratio",
                          "Email Volume (norm)", "Avg Size (norm)"]
            values = [
                avg_ext, avg_offh, avg_wknd,
                udf["total_emails"].mean() / max(df["total_emails"].max(), 1),
                udf["avg_email_size"].mean() / max(df["avg_email_size"].max(), 1),
            ]
            fig_rad = go.Figure(go.Scatterpolar(
                r=values + [values[0]], theta=categories + [categories[0]],
                fill="toself", fillcolor="rgba(59,130,246,0.2)",
                line=dict(color="#3B82F6", width=2)
            ))
            fig_rad.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(visible=True, range=[0, 1], color="#64748B"),
                           angularaxis=dict(color="#CBD5E1")),
                paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=10), height=300,
                template="plotly_dark",
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        st.markdown('<div class="section-header">Email Volume & External Ratio Over Time</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=udf["day"].astype(str), y=udf["total_emails"],
                              name="Total Emails", marker_color="#3B82F6", opacity=0.8))
        fig2.add_trace(go.Scatter(x=udf["day"].astype(str), y=udf["external_ratio"],
                                  name="External Ratio", yaxis="y2",
                                  line=dict(color="#EF4444", width=2), mode="lines+markers"))
        fig2.update_layout(
            yaxis=dict(title="Email Count", color="#3B82F6"),
            yaxis2=dict(title="External Ratio", overlaying="y", side="right",
                        range=[0, 1], color="#EF4444"),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1), margin=dict(t=10, b=10), height=280
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Daily Activity Log</div>', unsafe_allow_html=True)
        display_cols = ["day", "anomaly_score", "risk_level", "total_emails",
                        "external_ratio", "off_hours_ratio", "weekend_ratio",
                        "avg_email_size", "unique_pcs"]
        st.dataframe(
            udf[display_cols].style
               .map(color_risk, subset=["risk_level"])
               .format({"anomaly_score": "{:.4f}", "external_ratio": "{:.2%}",
                        "off_hours_ratio": "{:.2%}", "weekend_ratio": "{:.2%}",
                        "avg_email_size": "{:,.0f}"}),
            use_container_width=True, height=350
        )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 3 — CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif view == "🤖 Cluster Analysis":
    st.markdown("# Cluster Analysis")
    st.markdown("Behavioral breakdown of each cluster derived from the real email activity features.")

    cluster_summary = (
        df.groupby("cluster_name")
        .agg(
            users        =("user", "nunique"),
            records      =("anomaly_score", "count"),
            avg_score    =("anomaly_score", "mean"),
            avg_external =("external_ratio", "mean"),
            avg_offhours =("off_hours_ratio", "mean"),
            avg_weekend  =("weekend_ratio", "mean"),
            avg_emails   =("total_emails", "mean"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )

    st.markdown('<div class="section-header">Cluster Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        cluster_summary.style
            .background_gradient(subset=["avg_score"], cmap="Reds")
            .background_gradient(subset=["avg_external"], cmap="Oranges")
            .format({"avg_score": "{:.3f}", "avg_external": "{:.2%}",
                     "avg_offhours": "{:.2%}", "avg_weekend": "{:.2%}",
                     "avg_emails": "{:.1f}"}),
        use_container_width=True
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Avg Anomaly Score by Cluster</div>', unsafe_allow_html=True)
        fig_bar = px.bar(
            cluster_summary.sort_values("avg_score"),
            x="avg_score", y="cluster_name", orientation="h",
            color="cluster_name", color_discrete_map=COLORS,
            template="plotly_dark", text="avg_score",
        )
        fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, margin=dict(t=10, b=10), height=320,
                              xaxis_title="Avg Anomaly Score")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Feature Breakdown by Cluster</div>', unsafe_allow_html=True)
        feat_melt = cluster_summary.melt(
            id_vars="cluster_name",
            value_vars=["avg_external", "avg_offhours", "avg_weekend"],
            var_name="Feature", value_name="Value"
        )
        feat_melt["Feature"] = feat_melt["Feature"].map({
            "avg_external": "External Ratio",
            "avg_offhours": "Off-Hours Ratio",
            "avg_weekend":  "Weekend Ratio"
        })
        fig_grp = px.bar(
            feat_melt, x="cluster_name", y="Value", color="Feature",
            barmode="group", template="plotly_dark",
            color_discrete_sequence=["#EF4444", "#F59E0B", "#8B5CF6"],
        )
        fig_grp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=10, b=10), height=320,
                              xaxis_title="", yaxis_title="Ratio",
                              legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_grp, use_container_width=True)

    st.markdown('<div class="section-header">Score Distribution per Cluster</div>', unsafe_allow_html=True)
    fig_box = px.box(
        filtered, x="cluster_name", y="anomaly_score",
        color="cluster_name", color_discrete_map=COLORS,
        template="plotly_dark", points="outliers",
    )
    fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, margin=dict(t=10, b=10), height=350,
                          xaxis_title="", yaxis_title="Anomaly Score")
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="section-header">Users per Cluster</div>', unsafe_allow_html=True)
    user_cluster = (
        df.groupby(["user", "cluster_name"])["anomaly_score"]
        .max().reset_index()
        .sort_values("anomaly_score", ascending=False)
    )
    selected_cluster = st.selectbox("Filter by Cluster", ["All"] + list(COLORS.keys()))
    if selected_cluster != "All":
        user_cluster = user_cluster[user_cluster["cluster_name"] == selected_cluster]
    st.dataframe(
        user_cluster.style.format({"anomaly_score": "{:.4f}"}),
        use_container_width=True, height=350
    )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 4 — ANOMALY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif view == "📡 Anomaly Explorer":
    st.markdown("# Anomaly Explorer")
    st.markdown("Explore raw feature distributions and anomaly patterns across all records.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Anomaly Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            filtered, x="anomaly_score", nbins=40,
            color="risk_level",
            color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
            template="plotly_dark", barmode="overlay", opacity=0.8,
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=10, b=10), height=300,
                               xaxis_title="Anomaly Score", yaxis_title="Record Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">External Ratio vs Off-Hours Ratio</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            filtered, x="external_ratio", y="off_hours_ratio",
            color="cluster_name", color_discrete_map=COLORS,
            size="anomaly_score", size_max=18, opacity=0.75,
            template="plotly_dark",
            hover_data={"user": True, "anomaly_score": ":.3f", "day": True},
        )
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             margin=dict(t=10, b=10), height=300,
                             legend=dict(orientation="h", y=-0.25, font_size=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    feat_cols = ["anomaly_score", "external_ratio", "off_hours_ratio", "weekend_ratio",
                 "total_emails", "avg_email_size", "unique_pcs", "total_recipients"]
    corr = filtered[feat_cols].corr().round(3)
    fig_heat = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template="plotly_dark",
    )
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-header">Anomaly Score Over Time (daily avg & max)</div>', unsafe_allow_html=True)
    filtered["day_dt"] = pd.to_datetime(filtered["day"])
    daily = filtered.groupby("day_dt")["anomaly_score"].agg(["mean", "max"]).reset_index()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=daily["day_dt"], y=daily["mean"],
                                mode="lines", name="Daily Avg",
                                line=dict(color="#3B82F6", width=2)))
    fig_ts.add_trace(go.Scatter(x=daily["day_dt"], y=daily["max"],
                                mode="lines", name="Daily Max",
                                line=dict(color="#EF4444", width=1.5, dash="dot")))
    fig_ts.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=280, margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.1),
        xaxis_title="Date", yaxis_title="Anomaly Score"
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="section-header">Top Users by Peak Anomaly Score</div>', unsafe_allow_html=True)
    top_users = (
        filtered.groupby("user")
        .agg(peak_score  =("anomaly_score", "max"),
             avg_score   =("anomaly_score", "mean"),
             active_days =("day", "count"),
             cluster     =("cluster_name", lambda x: x.mode()[0]))
        .reset_index()
        .sort_values("peak_score", ascending=False)
        .head(30)
    )
    st.dataframe(
        top_users.style
                 .background_gradient(subset=["peak_score"], cmap="Reds")
                 .format({"peak_score": "{:.4f}", "avg_score": "{:.4f}"}),
        use_container_width=True, height=400
    )
