# PathSnap: Map-Matching and Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus).

PathSnap matches noisy GPS points to a road network and reconstructs the route a vehicle drove.
For each GPS point it generates candidate road arcs, scores them, and decodes the most likely
connected route with an HMM and the Viterbi algorithm. The model and the API run on one laptop,
and the evaluator UI runs on a second laptop.

**Final result on the 10 evaluator trips: 94.3% Top-1 accuracy, 0.967 macro F1 and 99.0% route continuity**
(distance emission + HMM/Viterbi; all settings frozen on training data before the evaluator set was scored).
All numbers in this README are read from files committed under `outputs/`.

---

## Dataset

The data is the public map-matching dataset of Kubicka et al. (2015), *Dataset for testing and
training of map-matching algorithms*, IEEE Intelligent Vehicles Symposium, pp. 1088–1093
([Zenodo](https://doi.org/10.5281/zenodo.57731), CC BY-SA 4.0). Trips are spread across 25 UTM zones
worldwide, and each comes with its own road network.

| Partition | Trips | GPS points | Usable points* | Used for |
|---|---|---|---|---|
| train | 80 (78 usable) | 208,417 | 180,274 | Training, 5-fold CV, every fitted threshold and all tuning |
| student_test | 10 | 19,160 | 19,119 | Model comparison during development |
| evaluator_test | 10 | 19,674 | 19,634 | Final evaluation only, with frozen models and settings |

\*Usable = within 40 m of the reference route. Trips 00000033 and 00000054 have reference routes that
do not follow the GPS and are excluded from training (see [docs/data_quality.md](docs/data_quality.md)).

Each trip folder holds `.track` (lon, lat, t), `.nodes` (lon, lat), `.arcs` (from_node, to_node)
and `.route` (reference arc ids in driving order). Raw data is **not** committed.

## Repository layout

```
common/            Shared loader (load_trip) and geometry (per-trip UTM projection, distances, headings)
eda/               Dataset exploration: inventory, statistics, plots, Folium maps
data_processing/   Track cleaning flags, GPS-noise estimate (sigma), gap segments
features/          Candidate generation (k=8 nearest arcs within 50 m), labels, 24 features
models/            Training, CV and evaluation of every model; Bayesian / Optuna tuning
hmm/               HMM/Viterbi decoder, emission-weight tuning, benchmarks, final scores
viz/               Folium map builder (GPS, road network, reference and predicted route layers)
api/               FastAPI backend (laptop 1): live candidates -> model -> HMM, server-side metrics
ui/                Streamlit evaluator UI (laptop 2)
docs/              Data-quality findings; experiment notes for the baseline/linear models
outputs/           All results: CSV, JSON, plots, confusion matrices, maps, saved models
```

## Setup

Python 3.10+. About 1 GB of free disk space.

```
python -m venv .venv
.venv\Scripts\activate            (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Laptop 2 (UI only) needs just `pip install -r ui/requirements.txt` (streamlit, requests, pandas): no data, models or geo libraries.

Place the raw data next to the repository, e.g.

```
<parent folder>/
    pathsnap/          (this repo)
    train/  student_test/  evaluator_test/
```

Scripts take `--data-root` pointing at the folder that contains `train/`. The API searches the
repo's parent folder and its sub-folders for the partitions (or uses `PATHSNAP_DATA`).

## Reproducing the results

Run from the repository root. Times are approximate on a laptop.

| Step | Command | Output |
|---|---|---|
| 1. EDA (about 2 min) | `python -m eda.run_eda --data-root ..` | `outputs/eda/` |
| 2. Cleaning (about 1 min) | `python -m data_processing.clean --data-root ..` | `outputs/clean/` |
| 3. Candidates and features (about 40 s) | `python -m features.build_candidates --data-root ..` | `outputs/features/` |
| 3b. Evaluator features | `python -m features.build_candidates --data-root .. --partitions evaluator_test` | `features_evaluator_test.parquet` |
| 4. Baseline, Naive Bayes, conditional logit | `python -m models.baselines_linear --model all` | `outputs/models/{nearest_arc,gaussian_nb,conditional_logit}/` |
| 5. Extra Trees, MLP | `python -m models.trees_nn --model extra_trees` / `--model mlp` | `outputs/models/{extra_trees,mlp}/` |
| 6. LightGBM | `python -m models.lgbm --variant both` | `outputs/models/{lgbm_binary,lgbm_rank}/` |
| 7. Tuning: conditional logit | `python -m models.bayesian_optimization --trials 10` | `outputs/models/conditional_logit/bayes_opt_*` |
| 8. Tuning: MLP | `python -m models.tune_trees_nn --model mlp --trials 10` | `outputs/models/mlp_tuned/` |
| 9. Evaluator scores (baseline/NB/logit) | `python -m models.evaluate_evaluator_test` | `outputs/models/*/evaluator_test_metrics.json` |
| 10. HMM emission weight (train only) | `python -m hmm.tune_emission_weight --data-root ..` | `outputs/hmm/emission_weight*.{csv,json}` |
| 11. HMM benchmark | `python -m hmm.benchmark_test_set --data-root ..` (add `--partition evaluator_test` for the evaluator set) | `outputs/hmm/benchmark_*.csv` |
| 12. Final scores | `python -m hmm.final_scores --data-root .. --partition student_test` (or `evaluator_test`) | `outputs/hmm/final_scores_*.csv` |

`outputs/clean/tracks/` and the Extra Trees model file are not committed (size); steps 2 and 5 regenerate them.

## Evaluation protocol

- Train only on rows with `label_usable == 1`; inputs are the 24 `FEATURE_COLS` only.
- 5-fold CV grouped by trip: fold = (position in the sorted list of usable training trips) % 5. All
  models use the same folds (identical validation row counts per fold).
- Scalers and transforms are fitted on training folds only. Probabilities are normalised per GPS point.
- **Top-1 accuracy**: for each usable GPS point, the highest-probability candidate is correct if its
  label is 1; points whose true arc is not among the candidates count as wrong.
- **Macro F1**: chosen candidate = 1, all others = 0, macro-averaged over classes 0/1, usable rows.
- **Log loss**: row-level, on per-point normalised probabilities.
- **Route continuity**: share of consecutive points whose arcs are the same (or duplicate) or directly connected.
- student_test is used for comparison only; evaluator_test is scored once with frozen models.

## Results

### Model comparison (classifier only, no HMM)

CV = mean ± std over the 5 trip-grouped folds. Inference = mean CV ms per 1,000 candidate rows.

| Model | Family | CV Top-1 | CV macro F1 | CV log loss | Student Top-1 | Student macro F1 | Student log loss | Evaluator Top-1 | Inference (ms/1k) |
|---|---|---|---|---|---|---|---|---|---|
| Nearest arc (σ = 6.68 m) | Baseline, no training | 56.9% ± 4.9 | 0.740 | 0.283 | 54.1% | 0.724 | 0.276 | 49.5% | 0.26 |
| Gaussian Naive Bayes | Probabilistic | 83.0% ± 2.0 | 0.898 | 0.335 | 83.8% | 0.903 | 0.309 | 78.7% | 2.59 |
| Conditional logit (λ = 1e-3) | Linear | 85.3% ± 2.2 | 0.912 | 0.138 | 85.3% | 0.912 | 0.131 | 79.9%† | 0.69 |
| Extra Trees (200 trees) | Bagging ensemble | 85.6% ± 2.5 | 0.914 | 0.131 | 85.7% | 0.915 | 0.123 | 80.1% | 2.04 |
| MLP (64-32) | Neural network | 86.3% ± 2.6 | 0.918 | 0.115 | 86.2% | 0.918 | 0.105 | 80.6% | 1.19 |
| LightGBM binary | Gradient boosting | 86.6% ± 2.5 | 0.920 | 0.108 | 88.1% | 0.929 | 0.093 | 80.4% | 4.84 |
| **LightGBM LambdaRank** | Gradient boosting (ranking) | **87.0% ± 2.7** | **0.923** | **0.106** | **88.7%** | **0.933** | **0.090** | 79.6% | 1.99 |

Sources: `outputs/models/<model>/cv_folds.csv`, `summary.json`, `evaluator_test_metrics.json`
(baseline, NB, logit), `summary.json["evaluator_test_metrics"]` (LightGBM),
`outputs/models/arpan_final_evaluation.json` (Extra Trees, MLP). †Evaluator score of the tuned logit (λ = 0.0105).

Route continuity of the classifier-only predictions was computed for the nearest-arc baseline
(82.0% student / 82.7% evaluator) and the conditional logit (93.5% / 93.7%); it was not computed for the other models.

**Best single classifier: LightGBM LambdaRank** (highest CV Top-1, lowest CV log loss).
Most important features (gain): `dist_minus_min_m`, `heading_diff_deg`, `cand_rank`, `frac_to_end`.

Why the nearest-arc baseline is so low: most roads are stored as two arcs in opposite directions at
exactly the same distance, so "nearest" picks the wrong direction about half the time on two-way
roads. Models fix this mainly through the heading difference. (The 90.3% in `docs/data_quality.md`
counts the reference arc as found when it is tied with its reverse twin, so it is not a Top-1 figure.)

**Over/underfitting evidence** (train-fold vs validation-fold, CV means): Extra Trees Top-1 86.5% vs
85.6%, log loss 0.123 vs 0.131 (small gap); MLP Top-1 91.2% vs 86.3%, log loss 0.071 vs 0.115 (moderate
overfitting, controlled by early stopping on validation loss: `outputs/models/mlp/loss_curves.png`).
LightGBM learning curves per fold: `outputs/models/lgbm_*/learning_curve_fold_*.png`.
Every classifier scores 5–9 points lower on the evaluator trips than on student test.

### Hyperparameter optimisation

Two searches were run; both used training trips only.

**Conditional logit**: Gaussian-process Bayesian optimisation (Matérn 5/2 kernel, Expected
Improvement), 10 trials, λ log-uniform in [1e-6, 1], objective = 5-fold CV macro F1.
Best λ = 0.0105 (trial 3). Files: `outputs/models/conditional_logit/bayes_opt_trials.csv`,
`bayes_opt_convergence.png`, `untuned_vs_tuned_comparison.csv`.

| Metric | Untuned (λ = 1e-3) | Tuned (λ = 0.0105) |
|---|---|---|
| CV macro F1 | 0.9119 | 0.9122 |
| Student Top-1 | 85.33% | 85.41% |
| Student macro F1 | 0.9124 | 0.9129 |
| Student log loss | 0.1306 | 0.1338 |

**MLP**: Optuna TPE sampler (seed 42, 4 random start-up trials), 10 trials, objective = mean 5-fold
CV Top-1 (about 22 min). Search space: hidden layers {32-16, 64-32, 64-64, 96-48}, alpha log-uniform
[1e-6, 1e-2], batch size {512, 1024, 2048}, learning rate log-uniform [3e-4, 3e-3]. Best (trial 1):
64-32, alpha 4.2e-6, batch 2048, learning rate 0.0012. Files: `outputs/models/mlp_tuned/`
(`trials.csv`, `best_trial.json`, `search_config.json`, `baseline_vs_tuned.csv`, `study.sqlite3`).

| Metric | Untuned | Tuned |
|---|---|---|
| CV Top-1 | 86.296% | 86.299% |
| CV macro F1 | 0.9181 | 0.9181 |
| CV log loss | 0.1154 | 0.1159 |
| Student Top-1 | 86.22% | 86.09% |
| Student macro F1 | 0.9178 | 0.9170 |
| Student log loss | 0.1045 | 0.1063 |
| Inference (ms/1k, student) | 1.13 | 0.64 |

In both cases tuning changed accuracy by less than 0.2 points: the default configurations were
already near the optimum. A LightGBM LambdaRank search script exists (`models/bayes_opt.py`,
Optuna, 3-fold CV log loss) but was **not run**, so there are no tuned LightGBM results.

### HMM / Viterbi route reconstruction

- **Hidden states**: Person B's candidate arcs for each GPS point (k = 8, radius 50 m).
- **Emissions**: per-point-normalised probabilities, either from the Gaussian distance score
  exp(−½(d/σ)²) with σ = 6.68 m, or from the conditional logit with an emission weight w
  (log-emission × w; w tuned on 20 training trips with out-of-fold probabilities).
- **Transitions**: road topology: same/duplicate arc, direct successor, two-hop successor, otherwise a
  fixed penalty (so the decoder never dead-ends where the network is disconnected).
- **Gaps**: a trip is split wherever the GPS gap exceeds 180 s, and each segment is decoded separately.

Emission-weight tuning on training trips (`outputs/hmm/emission_weight_tuning.csv`): HMM Top-1 rose
steadily as w decreased (89.3% at w = 1 to 93.0% at w = 0.05), and the distance-only emission scored
highest (94.3%). The settings were frozen from this before either test set was scored.

| Pipeline | Student Top-1 | Student macro F1 | Student continuity | Evaluator Top-1 | Evaluator macro F1 | Evaluator continuity |
|---|---|---|---|---|---|---|
| Nearest arc, no HMM | 54.1% | 0.724 | 82.0% | 49.5% | 0.697 | 82.7% |
| Conditional logit, no HMM | 85.4% | 0.913 | 93.5% | 79.9% | 0.880 | 93.7% |
| Conditional logit + HMM (w = 1) | 88.2% | 0.929 | 98.7% | 83.9% | 0.904 | 98.3% |
| Conditional logit + HMM (w = 0.05) | 89.9% | 0.940 | 99.7% | 93.4% | 0.961 | 99.4% |
| **Distance emission + HMM** | **91.9%** | **0.952** | **99.3%** | **94.3%** | **0.967** | **99.0%** |

Sources: `outputs/hmm/final_scores_student_test.csv`, `final_scores_evaluator_test.csv`, per-trip
detail in `benchmark_*.csv`. The HMM rows were computed with the tuned conditional logit (λ = 0.0105).
Hardest evaluator trip: 00000017 (84.9% with distance + HMM).

The HMM improves every pipeline and nearly removes route breaks. It works best with the distance
emission: the trained models are confidently wrong in runs of consecutive points (their direction
choice), which the HMM cannot override, while the distance emission leaves direction to road topology.

### Folium maps

`viz/folium_map.py` builds maps with toggleable layers (road network, GPS points, reference route,
predicted route) and a legend. Committed maps: `outputs/maps/map_{train_00000001,student_test_00000000,evaluator_test_00000015}.html`
and the EDA maps in `outputs/eda/maps/`.

## Deployment (two laptops)

**Laptop 1: backend (data, models, HMM, maps)**

```
cd pathsnap
.venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Data: keep `train/`, `student_test/` and `evaluator_test/` in the repo's parent folder or one level
  below it (e.g. `Student folder/Student folder/`); the API finds them automatically and prints the
  folder it uses at startup. To point elsewhere: `set PATHSNAP_DATA=D:\path\to\data` before starting.
- Allow TCP port 8000 in Windows Firewall (Private network), and note the IPv4 address from `ipconfig`.

**Laptop 2: evaluator UI**

```
git clone https://github.com/venkamita/ML_hackathon_26.git
cd ML_hackathon_26
python -m venv .venv
.venv\Scripts\activate
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

- In the sidebar, enter `http://<laptop-1 IP>:8000` and click **Ping API**
  (or `set PATHSNAP_API=http://<laptop-1 IP>:8000` before starting). Use the same network for both
  laptops; a phone hotspot is the safest.
- Pick a partition and trip, choose the emission, and click **Reconstruct Route**: metrics come from
  `/predict`, and the Folium map is built on Laptop 1 and served by `/map`.

**API endpoints**: `GET /health`, `GET /trips?partition=...`, `GET /models`,
`GET /predict/{trip_id}?partition=...&model=...`, `GET /map/{trip_id}?partition=...&model=...`.

- `model=nearest_arc` (default) = distance emission + HMM, the final pipeline; `conditional_logit` =
  logit with the tuned emission weight; `gaussian_nb` also works. `/models` lists only these three,
  because LightGBM, MLP and Extra Trees are not wired into the live API.
- The API scores each trip live and never reads saved predictions. The reference route stays on
  Laptop 1: it is used to compute the metrics and is drawn as a layer on the map.

## Data-integrity rules

- No tuning or threshold fitting on student_test or evaluator_test; evaluator results come from frozen settings.
- Columns derived from the reference route (`label_*`, `point_has_positive`) are never model features.
- No predictions are hardcoded.

## Documentation

- [docs/data_quality.md](docs/data_quality.md): data-quality findings and decisions
- [docs/MODEL_EXPERIMENTATION_REPORT.md](docs/MODEL_EXPERIMENTATION_REPORT.md): baseline, Naive Bayes and conditional-logit experiments
- [README_A.md](README_A.md): exploration and cleaning code and their output files

## References

- M. Kubicka, A. Cela, P. Moulin, H. Mounier, S. I. Niculescu. *Dataset for testing and training of map-matching algorithms.* IEEE IV 2015.
- P. Newson, J. Krumm. *Hidden Markov Map Matching Through Noise and Sparseness.* ACM SIGSPATIAL GIS 2009.
- P. Chao, Y. Xu, W. Hua, X. Zhou. *A Survey on Map-Matching Algorithms.* arXiv:1910.13065, 2019.

## AI assistance

AI tools (including ChatGPT) were used extensively to assist with code generation, implementation, debugging, and understanding of the data-processing pipeline. This included data loading, exploratory data analysis, road network analysis, GPS candidate generation, feature extraction, and map visualisation using Folium. The generated code and outputs were reviewed, tested, and integrated as part of the project implementation.

## Team

Sharat Doddihal-PES1UG24CS430
Arpan Joardar- PES1UG25CS805
Shantheri Shenoy- PES1UG24CS428
Ruthik -PES1UG25CS837
