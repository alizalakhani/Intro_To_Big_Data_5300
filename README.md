# Insider Threat Detection - Streamlit Interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## Features

- **Data Upload**: Upload your email CSV file
- **Algorithm Selection**: Choose from Spark KMeans, Custom KMeans, DBSCAN, or HDBSCAN
- **Interactive Visualizations**: PCA projections, heatmaps, and anomaly charts
- **Anomaly Detection**: View and export top anomalous users
- **Results Export**: Download clustering results as CSV

## Expected CSV Format

Your email dataset should have these columns:
- `id`, `date`, `user`, `pc`
- `to`, `cc`, `bcc`, `from`
- `size`, `attachments`, `content`
