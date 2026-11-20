# NYC Transit Pulse: Multi-Pillar Urban Mobility Analytics & Congestion Auditing Engine

An end-to-end NYC taxi auditing pipeline that combines surge-tier classification, carbon-efficiency regression, and 20-hub spatial clustering to pinpoint where urban traffic turns into a wallet-draining, emission-heavy parking lot.

---

## Overview & Problem Statement

Built from the perspective of an ML engineer supporting municipal transit authorities (such as the NYC Taxi & Limousine Commission), **NYC Transit Pulse** addresses two systemic urban mobility challenges across **12,210,952** raw yellow taxi trip records:

1. **Gridlock & Rampant Carbon Emissions:** Simultaneous event releases and rush-hour surges cause severe congestion where vehicles idle and crawl, burning excessive fuel and spiking urban emissions.
2. **Route & Meter Inefficiency:** Standard time-and-distance meters reward slow crawling in gridlock or unnecessarily dragged-out routes rather than penalizing low-efficiency travel.

Instead of relying on a single model, this project connects three complementary machine learning paradigms—**Supervised Classification**, **Supervised Regression**, and **Unsupervised Clustering**—into one unified auditing pipeline.

---

## Multi-Pillar Architecture

### 1. Data Ingestion & Sanity Cleaning
* **Raw Dataset:** `12,210,952` records across `19` columns.
* **Cleaning:** Parsed pickup/dropoff timestamps into `datetime` objects and filtered out invalid trips (`passenger_count <= 0`, `fare_amount <= 0`, `trip_distance <= 0`, or `dropoff < pickup`), yielding **`12,133,876`** clean records.
* **Spatial Filtering:** Restricted coordinates to the core NYC bounding box (`latitude: 40.55–40.90`, `longitude: -74.05–-73.70`), leaving **`11,922,870`** valid urban trips.

### 2. Pillar A — Surge Tier Prediction (Classification)
* **Goal:** Categorize trips into `0` (Normal Traffic, `<70th` percentile fare-per-mile), `1` (Moderate Surge, `70th–90th` percentile), and `2` (Gridlock Nightmare, `>=90th` percentile).
* **Leakage Prevention:** Dropped `fare_amount` and `fare_per_distance` after detecting target leakage (`0.9999` Macro-F1), retraining strictly on **5 physical and temporal features**: `trip_distance`, `passenger_count`, `trip_duration`, `pickup_hour`, and `pickup_day_of_week`.
* **Model & Performance:** `RandomForestClassifier` (`n_estimators=50`, `max_depth=12`) evaluated on a 200,000-row blind test split achieved a **Macro-F1 score of `0.9557`** (and `0.95` F1 on minority Tier 2 Gridlock).

### 3. Pillar B — Carbon & Efficiency Penalty Index (Regression)
* **Goal:** Predict a continuous severity score (`cepi_score`) capturing fuel waste and congestion friction:
  $$\text{CEPI} = \left(\frac{\text{trip\_duration}}{\text{trip\_distance} + \epsilon}\right) \times \text{fare\_per\_distance}$$
* **Outlier Handling:** Capped extreme near-zero-distance GPS anomalies (up to `7.18e+08`) at the **99th percentile (`269.11`)**.
* **Model & Performance:** `RandomForestRegressor` (`n_estimators=50`, `max_depth=12`) trained on the 5 leak-free telemetry features achieved **`R² = 0.9884`** and **`RMSE = 4.4140`** on the 200,000-row test split.

### 4. Pillar C — Ghost Corridor Identification (Clustering)
* **Goal:** Uncover hidden spatial congestion hubs and high-friction pickup epicenters without manual labels.
* **Model & Output:** Fitted `KMeans(n_clusters=20, random_state=42)` on **500,000** sampled pickup coordinates, mapping 20 distinct bottleneck hubs (`Hub-ID` `0`–`19`) and labeling key urban zones (such as `Midtown / Times Square Core`, `Upper East Side / Bridge Approach`, `Financial District / Lower Manhattan`, and `JFK Airport Transit Corridor`).

### 5. Cross-Pillar Integration & Final Predictions
All three trained models were unified on a **`100,000`-row** spatial sample (`integrated_df`):
* **Spatial Assignment:** Assigned each pickup coordinate to its nearest `Hub-ID` via `kmeans.predict()`.
* **Predictive Scoring:** Generated `predicted_surge_tiers` and `predicted_cepi_score` using each estimator's exact feature order (`feature_names_in_`).
* **Validation:** Passed integrity assertions (zero nulls, valid tier labels `{0, 1, 2}`, non-negative CEPI scores) and achieved **`Macro-F1 = 0.9550`**, **`RMSE = 3.9610`**, and **`R² = 0.9906`** on the integrated sample.

---

## Key Integrated Findings (Top Ghost Corridors)

Aggregating all three pillars by `Hub-ID` reveals the trade-off between **sheer pickup volume** and **per-trip congestion severity**:

| Rank | Hub-ID | Corridor Label | Sample Trips | Avg Predicted CEPI | Total Predicted CEPI | Predicted Gridlock Rate (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | `0` | Midtown / Times Square Core | 10,004 | 48.09 | **481,070.99** | 11.27% |
| **2** | `2` | Urban Arterial Bottleneck / Side-Street Hub | 9,330 | 50.59 | **472,032.64** | 14.20% |
| **3** | `6` | Midtown / Times Square Core | 9,392 | 50.18 | **471,287.11** | 13.14% |
| **4** | `11` | Midtown / Times Square Core | 8,653 | **52.24** | **452,035.14** | **14.91%** |
| **5** | `17` | Midtown / Times Square Core | 8,777 | 47.63 | **418,029.85** | 10.65% |

* **Highest Total Impact:** **Hub-ID 0** (`Midtown / Times Square Core`) generates the highest cumulative carbon/efficiency penalty (`481,070.99`) driven by peak pickup density (`10,004` trips).
* **Highest Per-Trip Friction:** **Hub-ID 11** (`Midtown / Times Square Core`) records the steepest average predicted CEPI (`52.24`) and highest predicted Tier 2 gridlock rate (`14.91%`), closely followed by **Hub-ID 2** (`14.20%` gridlock rate).

---

## Repository Structure

```text
├── notebooks/
│   └── NYC_Transit_Pulse.ipynb                  # Full end-to-end pipeline (EDA, Pillars A-C, Integration)
├── outputs/
│   ├── nyc_trip_predictions_final.csv           # Trip-level integrated predictions (100k sample)
│   └── nyc_ghost_corridors_integrated_final.csv # Corridor-level 3-pillar summary & ranking (20 hubs)
├── docs/
│   └── NYC Transit Pulse Report.pdf             # Detailed technical walkthrough & visual report
└── README.md
```

---

## Tech Stack & How to Run

* **Language & Libraries:** Python 3, `pandas`, `numpy`, `scikit-learn`, `matplotlib`
* **Execution Steps:**
  1. Clone this repository and install dependencies:
     ```bash
     pip install pandas numpy scikit-learn matplotlib
     ```
  2. Download the NYC TLC Yellow Taxi trip dataset (March 2016) and update the dataset path in the notebook.
  3. Run the notebook cells sequentially from **Data Cleaning** through **Pillars A, B, C**, and the **5-Block Integration Pipeline** to reproduce the metrics and export both final `.csv` deliverables.