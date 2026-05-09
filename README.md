# DSC511_PROJECT
# DSC511_PROJECT - Weather Analysis for 12 cities (2019-2025)
 
# Project Overview
This project builds an end-to-end big-data pipeline on seven years of hourly weather data (2019–2025) for twelve globally distributed cities, sourced from the Open-Meteo API. The 736,416-row dataset flows through four phases:
-  ingestion
-  exploratory analysis with Apache Spark
-  machine learning with Spark MLlib (a GBT regressor predicting next-hour temperature at R² = 0.884)
-  climate-similarity analysis combining KMeans clustering with GraphFrames and NetworkX.
These techniques were used to cover multiple concepts taught throughout the course.
 
# Dataset
 
The dataset comes from the **Open-Meteo Historical Weather Archive API**, a free reanalysis-based service providing hourly weather data with global coverage. We collected seven years of observations (**2019-01-01 → 2025-12-31**) for **twelve cities** spanning diverse climate zones — Mediterranean (Athens, Nicosia), Desert (Cairo), Continental (Moscow, Berlin), Oceanic (London, Sydney), Humid Subtropical (New York, Tokyo), Tropical (Mumbai, São Paulo), and Tropical Highland (Nairobi). Each city contributes exactly **61,368 hourly rows**, yielding a master dataset of **736,416 rows × 27 columns** (≈ 290 MB in memory, 9.1 MB compressed as Snappy Parquet) with **zero missing values** after ingestion.
 
### Variables
 
Ten hourly weather variables were fetched per city:
- **`temperature_2m`** (air temperature 2 m above ground, °C)
- **`apparent_temperature`** (perceived temperature accounting for humidity and wind, °C)
- **`relative_humidity_2m`** (%)
- **`precipitation`** (mm/hr)
- **`wind_speed_10m`** (km/h)
- **`wind_direction_10m`** (degrees)
- **`surface_pressure`** (hPa)
- **`cloud_cover`** (%)
- **`shortwave_radiation`** (incoming solar, W/m²)
- **`et0_fao_evapotranspiration`** (FAO Penman–Monteith reference evapotranspiration, mm/hr)
-  metadata (city, latitude, longitude, climate label) and time features (year, month, day, hour, day-of-week, day-of-year, hemisphere-aware season, plus sine/cosine cyclical encodings of hour and month for ML).

-  ## Phase 1 — Exploratory Data Analysis (EDA)
 
 
### EDA SUMMARY
 
Total observations  : 736,416
Cities              : 12
Years               : 7
Variables analysed  : 8
 
Key findings:
  - Cairo and Mumbai show the highest mean temperatures
  - Moscow and Berlin show the widest seasonal temperature swings
  - Mumbai and Sao Paulo receive the most precipitation (monsoon/tropical)
  - Strong negative correlation between cloud cover and solar radiation
  - Temperature and apparent_temperature are highly correlated (r > 0.99)
  - Heatwave days are concentrated in Mediterranean and Desert climates
 
 
EDA is performed in Apache Spark 4.0 (4 GB driver, 8 shuffle partitions) directly on the 736,416-row DataFrame, using SQL aggregates, window functions, and Spark MLlib's `Correlation` API. Plots are produced with Matplotlib/Seaborn after aggregation collapses the data to summary size.
 
### Descriptive statistics
 
Global ranges confirm the data is physically plausible — temperatures span -31 °C (Moscow) to 45 °C (Cairo), pressure ranges from 833 to 1046 hPa, and wind gusts up to 92.7 km/h:
 
| Variable | Min | Mean | Max | Std |
|---|---|---|---|---|
| Temperature (°C) | -31.0 | 16.91 | 45.1 | 9.22 |
| Apparent temperature (°C) | -36.9 | 16.01 | 46.1 | 11.27 |
| Relative humidity (%) | 3 | 69.93 | 100 | 20.64 |
| Precipitation (mm/hr) | 0.0 | 0.11 | 47.6 | 0.59 |
| Wind speed (km/h) | 0.0 | 11.69 | 92.7 | 6.73 |
| Surface pressure (hPa) | 833.1 | 986.69 | 1045.8 | 49.71 |
| Cloud cover (%) | 0 | 52.73 | 100 | 42.50 |
| Shortwave radiation (W/m²) | 0.0 | 186.37 | 1116.0 | 268.82 |
 
Per-city aggregates rank cities from Mumbai (mean 26.8 °C) at the top to Moscow (6.8 °C) at the bottom — a 20 °C spread that reflects the geographic diversity of the sample.
 
### Key visualisations 
 
**1. Monthly average temperature heatmap** — exposes three distinct seasonal regimes: Northern-Hemisphere cities (Moscow, Berlin, London, New York, Tokyo, the Mediterraneans, Cairo) peak in June-August; Southern-Hemisphere cities-(Sydney, São Paulo) mirror that pattern with peaks in December-February; equatorial cities (Mumbai, Nairobi) stay within ~-4 °C all year, showing year-round isothermality.
 
**2. Diurnal temperature cycle** — Cairo has the largest daily swing (11.8 °C) and Moscow the smallest (5.6 °C). The pattern is physical: dry, clear-sky climates allow rapid longwave cooling at night and strong solar heating by day, while humid or high-latitude cities damp the cycle through cloud cover and persistent moisture.
 
**3. Precipitation by city and season** — Mumbai dominates at 17,550 mm of total rainfall over the period, more than triple Athens (3,774 mm) and over 60× Cairo (280 mm). The seasonal heatmap shows the South-Asian monsoon concentrates virtually all of Mumbai's rainfall in summer, while London and Berlin distribute precipitation evenly across the year.
 
 
## EDA part Key plots part2 

**4. Correlation matrix** — `apparent_temperature` correlates with `temperature_2m` at r > 0.99 (essentially redundant as a predictor), `cloud_cover` and `shortwave_radiation` are strongly negatively correlated as expected, and `wind_speed` carries independent information uncorrelated with the rest.


**5. Heatwave and coldwave detection** — using window functions to flag rolling 3-day extremes, heatwaves (max > 35 °C) saturate in Cairo (671 days) and Nicosia (477 days), while coldwaves (min < 0 °C) saturate in Moscow (789 days) and New York (338 days). The two rankings are almost complementary — cities appear in one list or the other, rarely both.
 

**6. Seasonal temperature range** — the single most discriminating climate feature. Moscow's summer-minus-winter gap is 23.9 °C while Nairobi's is 2.1 °C, an eleven-fold spread that drives the Phase 2B clustering directly.

## Phase 2A — Machine Learning with Spark MLlib


We train a **Gradient-Boosted Trees (GBT) regressor** to predict next-hour 2 m air temperature, using a Spark MLlib pipeline (`VectorAssembler` → `StandardScaler` → `GBTRegressor`). More precisely, for every hour in the dataset, the model predicts that hour's temperature using only the weather observations from one hour earlier (humidity, wind, pressure, cloud, radiation) plus static time and location features.
 

A deliberate design choice: we **exclude the previous-hour temperature** from the features. With `prev_temp` included the task collapses into trivial persistence forecasting (R² > 0.99 on any model), so removing it forces the model to learn from the actual weather context. 

### Setup

- **Features (12):** lagged humidity, wind, pressure, cloud, radiation; cyclical encodings of hour and month (`hour_sin/cos`, `month_sin/cos`); indexed climate label; latitude; longitude.

- **Train/test split:** temporal hold-out — train on 2019–2024 (631,284 rows, 85.7%), test on 2025 (105,120 rows). More realistic than a random split since it mirrors operational forecasting.

- **Model:** `GBTRegressor(maxIter=50, maxDepth=5, stepSize=0.1)`.


### Test-set performance (2025)


| Metric | Value |
|---|---|
| RMSE | **3.15 °C** |
| MAE | 2.36 °C |
| R² | **0.883** |

 
The model explains roughly 88% of variance with a typical hourly error of ~3 °C — a strong result given that `prev_temp` was withheld.

### Key visualisations

**1. Predicted vs. actual scatter** — points cluster tightly along the y = x diagonal across the full -10 °C to 40 °C range. Residuals form a roughly symmetric distribution around zero, with no systematic curvature, confirming the model isn't biased at temperature extremes.
 

**2. Feature importance** — `latitude` dominates at **0.313**, followed by the seasonal cyclical features `month_cos` (0.173) and `month_sin` (0.123). Together latitude + month account for ~61% of the model's predictive signal, confirming that **geographic position and time of year matter more than recent weather state** when previous temperature is hidden. `prev_radiation` (0.109) and `climate_idx` (0.076) round out the top five, which together account for ~79% of total importance.
 

**3. Per-city RMSE** — performance varies sharply by climate. Tropical cities (Nairobi 1.17 °C, Mumbai 1.69 °C) are easiest because intra-day and intra-year variance is small. Continental cities (Moscow 4.90 °C, New York 4.26 °C) are hardest — large daily swings and irregular winter cold events expand the error envelope.
 

**4. Seasonal performance** — Summer is the easiest season (RMSE 2.75 °C), closely followed by Autumn (2.81 °C). Spring is the hardest (3.73 °C), with Winter close behind (3.21 °C). Summer and Autumn benefit from more regular solar-driven daily cycles, while Spring and Winter add irregular synoptic events — cold fronts, rapid warm-ups, late-season storms — that the model cannot anticipate without recent temperature. 

### Cross-validation 

A 3-fold CV across 8 hyperparameter combinations on 20% of training data confirmed the chosen settings are near-optimal: best CV RMSE = **2.86 °C** at `maxIter=50, maxDepth=6, stepSize=0.1` — within ~10% of the production model's test RMSE, with no signs of overfitting.

## Phase 2B Part 1 — KMeans Clustering

We collapse each city's 61,368 hourly rows into a single **9-dimensional climate fingerprint** (mean temperature, temperature std, total precipitation, precipitation std, mean humidity, mean wind, mean cloud, mean radiation, and the summer-minus-winter `seasonal_range`), then standardise all features to zero mean and unit variance — essential because raw `total_precip` ranges from 280 mm (Cairo) to 17,550 mm (Mumbai) and would otherwise swamp the distance metric.

### Choosing k

We swept k = 2..7 and tracked two metrics — **WSSSE (Elbow)** (within-cluster error, lower is better) and **silhouette score** (separation quality, higher is better):

| k | WSSSE (Elbow)| Silhouette |
|---|---|---|
| 2 | 64.77 | 0.472 |
| 3 | 46.19 | 0.409 |
| 4 | 22.65 | 0.572 |
| **5** | **14.63** | **0.586  ← max** |
| 6 | 18.00 | 0.271 |
| 7 | 11.66 | 0.323 |

**k = 5** is the consensus optimum: highest silhouette and on the elbow of the WSSSE curve.

### Final clusters

| Cluster | Cities | Character |
|---|---|---|
| 0 | Athens, Cairo, Nicosia | **Hot-dry Mediterranean / Desert** — high temp, low precip, max radiation |
| 1 | Moscow, London, Berlin | **Cold temperate** — low temp, moderate seasonality, high humidity & cloud |
| 2 | Sydney, São Paulo, Nairobi | **Mild Southern / Equatorial** — small seasonal range (2–10 °C), moderate precip |
| 3 | Mumbai | **Monsoon Tropical** — alone, with 17,550 mm precipitation and a 2.4 °C seasonal range |
| 4 | New York, Tokyo | **Humid Subtropical** — strong seasonality (~21 °C swing), high precipitation |

### Key insights

**1. Data-driven clusters partly disagree with textbook climate labels** — and the disagreements are the interesting findings:
- Cairo (Desert) joins the two Mediterranean cities — its dry summer / mild winter pattern dominates over the desert label.
- London (Oceanic) joins Moscow and Berlin (Continental) — climatologically London is closer to cool continental Europe than to Sydney.
- Sydney (Oceanic), São Paulo (Tropical), and Nairobi (Tropical Highland) collapse into one cluster because they share a small seasonal range despite belonging to three different Köppen classes.

**2. Mumbai is a true outlier** — its 17,550 mm rainfall and 2.4 °C seasonal range place it so far from any other city in scaled feature space that it forms its own cluster.

**3. PCA confirms the structure** — projecting the 9-dim feature vectors into 2D retains **80.1% of variance** (PC1 = 42.9%, PC2 = 37.3%). PC1 separates cold from hot cities, PC2 separates dry-seasonal from humid-equatorial. The visual cluster boundaries match the KMeans assignments cleanly.

---

## Phase 2B Part 2 — Graph Analysis

 Graph analysis complements the earlier clustering by revealing structural relationships that centroid-based methods can miss: connected components show whether the similarity network fragments into isolated subgraphs — groups of cities whose climates are so distinct that no above-threshold link bridges them — while Louvain community detection identifies tightly-knit clusters within the connected portion of the graph. By using all of these techniques, we wanted to see if they will give use similar results.

### Graph structure at a glance

| Property | Value | Meaning |
|---|---|---|
| Nodes / Edges | 12 / 9 | Sparse — only ~14% of possible edges |
| Connected components | **4** | Graph splits into 4 disjoint groups |
| Triangles | **1** (Athens–Cairo–Nicosia) | The only fully-connected sub-triangle |
| Diameter | 3 | Longest shortest path in largest component |
| **Cluster assortativity** | **0.851** | Edges overwhelmingly connect same-cluster cities |

The **0.851 cluster assortativity** is the single most important number — a value near 1.0 means the graph's connectivity respects the KMeans cluster boundaries almost perfectly. The graph and the clustering tell the same story.

### Centrality — different metrics, different stories

| Metric | Top city | Why |
|---|---|---|
| **PageRank** | Berlin (1.44) | Hub of the cold-temperate group, links Moscow and London simultaneously |
| **Betweenness** | São Paulo (0.036) | The graph's *bridge* — sits on the Mumbai → Sydney/Nairobi path |
| **Closeness** | Athens (3.86) | Inside the only triangle, short weighted distances to neighbours |
| **Clustering coefficient** | Athens / Cairo / Nicosia (0.987) | All three sit inside the only triangle |

Mumbai is **last on every single metric** — it's a peripheral pendant node with a single incoming edge.

### KMeans vs. Graph methods — the key comparison

We compare three independent grouping methods:

| Method | Groups | Source |
|---|---|---|
| KMeans | **5** | Distance-based clustering on continuous features |
| Connected Components | **4** | Reachability in the similarity graph |
| Louvain | **4** | Modularity-maximising communities |

**Agreement: ARI = 0.843, NMI = 0.935** — extremely high (1.0 = perfect, 0.0 = random). The three methods agree on **11 out of 12 cities**.

### The single disagreement: Mumbai

- **KMeans** puts Mumbai in its own cluster (#3) — its feature vector is genuinely far from every other city.
- **Graph methods** (Components and Louvain) merge Mumbai into the tropical group (Sydney / São Paulo / Nairobi).

**Why?** The graph contains a single edge **Mumbai → São Paulo at similarity 0.759** — just barely above the 0.75 threshold. Once two nodes share an edge, graph methods consider them connected, regardless of how different their underlying features are. **Raising the threshold to 0.80 would remove that edge** and bring the two methods into perfect agreement (ARI = 1.0).

### Why this matters

The Mumbai case illustrates a deeper point: **graph methods see only connectivity (binary), while KMeans sees continuous distance**. When edge density is sparse near the threshold, a single borderline edge can flip a group's composition. The disagreement isn't a failure — it flags exactly the boundary case where the data has just enough similarity to bridge two otherwise-distinct climates. The 11/12 agreement confirms that groups validated by **both** methods are structurally robust climate classes.

### Connectivity reflects geography

Shortest-path analysis from extreme landmarks shows the cluster boundaries sharply: **Cairo can only reach Athens and Nicosia (its cluster mates); Moscow can only reach Berlin and London**. There is no path between hot-dry, cold-temperate, and tropical communities at the 0.75 threshold — the graph is genuinely disconnected, showing real climatic separation.

### Why clusters and graphs
Clustering and graph analytics, instead of predicting a known target, they let the data reveal its own structure — KMeans groups cities in continuous feature space, while the graph exposes how climates connect through pairwise similarity. Running both on the same data provides mutual validation.
