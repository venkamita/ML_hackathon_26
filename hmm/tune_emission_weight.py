"""
Tune the HMM emission weight w for the conditional-logit emissions, on
TRAINING trips only (student_test and evaluator data are never used here).

Tuning trips: every 4th usable training trip (sorted by id), 20 trips.
Each tuning trip is scored out-of-fold: the conditional logit is refitted on
the usable rows of all training trips in the OTHER folds (same fold rule as
the model comparison: fold = position in sorted trip list % 5), so the
probabilities are as honest as on unseen trips.

Run from the repo root:
    python -m hmm.tune_emission_weight --data-root ..
Writes outputs/hmm/emission_weight_tuning.csv and prints the best w.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data_loader import load_trip, resolve_data_root  # noqa: E402
from hmm.benchmark_test_set import SORT, TUNED_LAMBDA, top1_on_usable  # noqa: E402
from hmm.viterbi import RouteDecoder  # noqa: E402
from models.baselines_linear import ConditionalLogitModel  # noqa: E402

WEIGHTS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/hmm")
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_parquet(REPO_ROOT / "outputs" / "features" / "features_train.parquet")
    tr = tr.sort_values(SORT, kind="mergesort").reset_index(drop=True)
    usable_trips = sorted(tr.loc[tr.label_usable == 1, "trip_id"].unique())
    fold_of = {t: i % 5 for i, t in enumerate(usable_trips)}
    tune_trips = usable_trips[::4]
    print(f"{len(tune_trips)} tuning trips: {tune_trips}")

    # out-of-fold conditional-logit probabilities for the tuning trips
    probs = {}
    for f in range(5):
        trips_f = [t for t in tune_trips if fold_of[t] == f]
        if not trips_f:
            continue
        fit_rows = tr[(tr.label_usable == 1) & tr.trip_id.map(lambda t: fold_of.get(t, -1) != f)]
        m = ConditionalLogitModel(lmb=TUNED_LAMBDA, max_iter=100)
        m.fit(fit_rows.reset_index(drop=True))
        for t in trips_f:
            c = tr[tr.trip_id == t].reset_index(drop=True)
            c["p"], _ = m.predict_probs(c)
            probs[t] = c
        print(f"  fold {f}: fitted on {fit_rows.trip_id.nunique()} trips, scored {trips_f}")
    del tr

    rows = []
    trips = {t: load_trip(root, "train", t) for t in tune_trips}
    for w in WEIGHTS + ["nearest_arc"]:
        dec = RouteDecoder(emission_weight=1.0 if w == "nearest_arc" else w)
        c_tot = n_tot = 0
        for t in tune_trips:
            c = probs[t]
            res = dec.decode_candidates(c, trips[t].arcs, None if w == "nearest_arc" else c["p"].to_numpy())
            cc, nn = top1_on_usable(c, res["point_idx"], res["predicted_route"])
            c_tot += cc
            n_tot += nn
        rows.append({"emission": "nearest_arc" if w == "nearest_arc" else "conditional_logit",
                     "emission_weight": 1.0 if w == "nearest_arc" else w,
                     "hmm_top1_acc": c_tot / n_tot, "n_usable_points": n_tot})
        print(f"  {str(w):>11}: HMM top-1 = {100 * c_tot / n_tot:.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(out / "emission_weight_tuning.csv", index=False)
    best = df[df.emission == "conditional_logit"].sort_values("hmm_top1_acc", ascending=False).iloc[0]
    (out / "emission_weight.json").write_text(json.dumps(
        {"emission_weight": float(best.emission_weight), "tuned_on": tune_trips,
         "hmm_top1_acc_train_tuning_trips": float(best.hmm_top1_acc)}, indent=2))
    print(f"Best w = {best.emission_weight} (train tuning top-1 {best.hmm_top1_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
