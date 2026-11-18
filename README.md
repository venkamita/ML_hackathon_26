# PathSnap: Map-Matching and Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus).

PathSnap matches noisy GPS points to a road network and reconstructs the route a vehicle drove.
For each GPS point it generates candidate road segments, scores them with a trained classifier,
and decodes the most likely connected route with an HMM and the Viterbi algorithm. The model and
the API run on one laptop, and the evaluator UI runs on a second laptop.

## Dataset

The data is the public map-matching dataset of Kubicka et al. (2015), *Dataset for testing and
training of map-matching algorithms*, IEEE Intelligent Vehicles Symposium, pp. 1088–1093
([Zenodo](https://doi.org/10.5281/zenodo.57731), CC BY-SA 4.0). Trips are spread across the
world, and each comes with its own road network.

| Partition | Trips | GPS points | Used for |
|---|---|---|---|
| train | 80 | 208,417 | Training, cross-validation, every fitted threshold |
| student_test | 10 | 19,160 | Comparing models during development |
| evaluator test | 10 | – | Final evaluation only. Never read during development |

Each trip folder holds `.track` (lon, lat, t), `.nodes` (lon, lat), `.arcs` (from_node, to_node)
and `.route` (the reference arc ids in driving order). The raw data is **not** committed; place it
next to this repo (see Setup).

## Repository layout

```
common/            Shared loader and geometry (UTM projection, distances, headings)
eda/               Dataset exploration: inventory, statistics, plots, Folium maps
data_processing/   Track cleaning and flags; noise estimate; candidate miss rates
viz/               Folium map builder (GPS, network, reference and predicted route layers)
docs/              Data-quality findings and decisions
outputs/           Generated results (CSV, plots, maps)
features/          Candidate generation, labels, features          (in progress)
models/            Model training, comparison, Bayesian optimisation (in progress)
hmm/               HMM and Viterbi route reconstruction               (in progress)
api/               FastAPI backend (laptop 1)                         (in progress)
ui/                Evaluator UI (laptop 2)                            (in progress)
```

## Setup

Python 3.10 or newer. Put the raw data folder (the one containing `train/` and
`student_test/`) next to this repository, or pass its path to `--data-root`.

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

About 1 GB of free disk space is needed for the packages and outputs.

## Running the pipeline

Run from the repository root, in this order:

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Exploration | `python -m eda.run_eda --data-root ..` | about 2 min | `outputs/eda/` |
| 2. Cleaning | `python -m data_processing.clean --data-root ..` | about 1 min | `outputs/clean/` |
| 3. Features | *to be added* | | |
| 4. Model training and comparison | *to be added* | | |
| 5. Bayesian optimisation | *to be added* | | |
| 6. HMM / Viterbi evaluation | *to be added* | | |

To map specific trips: `python -m eda.run_eda --data-root .. --map-trips train/00000005 student_test/00000052`

`outputs/clean/tracks/` is not committed (about 50 MB). Regenerate it with step 2.

## Key results so far

From the full run on train and student_test. Details are in [docs/data_quality.md](docs/data_quality.md).

- GPS noise σz = **6.68 m**, estimated on train only (the median over trips of 1.4826 × the median distance from GPS points to the reference route).
- The nearest road arc is the reference arc for only **90.3%** of clean train points. Keeping the
  5 nearest arcs misses the reference 0.96% of the time; keeping 10 misses it 0.33%.
- Two training trips (00000033 and 00000054) have reference routes that don't follow the GPS.
  They are left out of the noise estimate and out of the training labels.
- Trips are split at GPS gaps over 180 s: 108 segments in train, 11 in student_test.

## Deployment (two laptops)

| Laptop | Runs | Command |
|---|---|---|
| 1 | Model, HMM, FastAPI backend | *to be added* |
| 2 | Evaluator UI | *to be added* |

The reference routes for evaluator trips stay on laptop 1 and are never sent to the model or the UI.

## Data-integrity rules we follow

- The evaluator test partition is never read during development.
- Every learned threshold and transformation is fitted on train only.
- Columns prefixed `label_` (and anything else derived from `.route`) are used only to build
  labels and training masks, never as model features.
- No predictions are hardcoded.

## Documentation

- [docs/data_quality.md](docs/data_quality.md): data-quality findings and the decisions taken
- [README_A.md](README_A.md): details of the exploration and cleaning code and their output files

## References

- M. Kubicka, A. Cela, P. Moulin, H. Mounier, S. I. Niculescu. *Dataset for testing and training of map-matching algorithms.* IEEE IV 2015.
- P. Newson, J. Krumm. *Hidden Markov Map Matching Through Noise and Sparseness.* ACM SIGSPATIAL GIS 2009.
- P. Chao, Y. Xu, W. Hua, X. Zhou. *A Survey on Map-Matching Algorithms.* arXiv:1910.13065, 2019.

## AI assistance

*Team to fill in: which parts were written or drafted with AI tools, as required by the event's policy.*

## Team

*Team to fill in: names and who owns which part.*
