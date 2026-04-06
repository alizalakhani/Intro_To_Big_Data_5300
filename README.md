# CERT Insider Threat – PySpark Clustering Pipeline

Automatically detects, loads, and clusters the **CERT Insider Threat r1** dataset
using three methods: baseline KMeans, silhouette-optimised KMeans, and a
density-peak (CDPC-KNN) approximation.

---

## Project Structure

```
cert_clustering/
├── main.py              # entry point
├── requirements.txt
├── r1/                  # ← place your dataset files here
│   ├── logon.csv
│   ├── device.csv
│   ├── HTTP.csv
│   └── LDAP-*.csv
├── output/              # created automatically
│   ├── clustered_users/ # CSV with one row per user + cluster labels
│   └── metrics.json     # silhouette, inertia, runtime per method
└── src/
    ├── loader.py        # data loading & feature engineering
    ├── preprocessor.py  # imputation, encoding, scaling
    ├── clustering.py    # three clustering methods
    └── evaluator.py     # metrics & output helpers
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Java is required for PySpark
#    macOS:  brew install openjdk@17
#    Ubuntu: sudo apt install openjdk-17-jdk
#    Windows: download from https://adoptium.net
```

---

## Running the Pipeline

Place the r1 dataset files in the `r1/` subdirectory, then:

```bash
# Basic run (default: k=5 baseline, sweep k=2–10)
python main.py

# Custom paths and K range
python main.py --data-dir /path/to/r1 --output-dir /path/to/output --k 5 --k-min 2 --k-max 12

# View help
python main.py --help
```

---

## Clustering Methods

| Method | Description |
|---|---|
| **KMeans-Baseline** | Standard KMeans, fixed K (default 5) |
| **KMeans-BestK** | Sweeps K from k-min to k-max, selects K with highest silhouette score |
| **CDPC-KNN-Approx** | Density-peak approximation: coarse KMeans → density proxy → peak selection → final assignment |

### CDPC-KNN Approximation – Design Note

The paper *"Clustering by Fast Search and Find of Density Peaks with K-Nearest Neighbours"*
requires a full O(n²) pairwise distance matrix. This is infeasible at scale in a distributed
setting. The implementation here approximates it as follows:

1. **Density estimation**: Run a coarse KMeans (large K). Each point's distance to its
   cluster centre is a proxy for local density (small distance = high density).
2. **Peak selection**: Among the densest cluster centres, greedily pick K_final that
   are maximally spread (greedy max-min distance), replicating the δ (distance-to-higher-density)
   criterion from the original algorithm.
3. **Assignment**: Assign every point to its nearest selected peak (equivalent to seeded
   KMeans without re-centring).

This reproduces the conceptual flow of CDPC-KNN (density → peaks → Voronoi assignment)
in a fully distributed, scalable manner.

---

## Output Files

| File | Contents |
|---|---|
| `output/clustered_users/` | CSV with one row per user; columns include all features + `prediction_baseline`, `prediction_bestk`, `prediction_cdpc` |
| `output/metrics.json` | Per-method: silhouette score, inertia (KMeans), cluster sizes, runtime |

---

## Supported Input Formats

- **CSV** (auto-detected by filename: `logon*.csv`, `device*.csv`, `HTTP*.csv`, `LDAP*.csv`)
- **Parquet** (generic fallback if CSV files are absent)

The pipeline handles mixed numerical and categorical features automatically.
