"""
Final scores with FROZEN settings: Top-1, row-level macro F1 (chosen candidate
= 1, all others = 0, usable rows only; same rule as the model comparison) and
route continuity, for each pipeline, on a given partition.

    python -m hmm.final_scores --data-root .. --partition evaluator_test [--features file.parquet]
Writes outputs/hmm/final_scores_<partition>.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data_loader import load_trip, resolve_data_root  # noqa: E402
from hmm.benchmark_test_set import SORT, TUNED_LAMBDA  # noqa: E402
from hmm.viterbi import RouteDecoder  # noqa: E402
from models.baselines_linear import ConditionalLogitModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--partition", default="evaluator_test")
    ap.add_argument("--features", default=None)
    a = ap.parse_args()
    root = resolve_data_root(a.data_root)
    feat = REPO_ROOT / "outputs" / "features"
    w = json.loads((REPO_ROOT / "outputs" / "hmm" / "emission_weight.json").read_text())["emission_weight"]

    tr = pd.read_parquet(feat / "features_train.parquet")
    tr = tr[tr.label_usable == 1].sort_values(SORT, kind="mergesort").reset_index(drop=True)
    model = ConditionalLogitModel(lmb=TUNED_LAMBDA)
    model.fit(tr)
    del tr

    te = pd.read_parquet(a.features or feat / f"features_{a.partition}.parquet")
    te = te.sort_values(SORT, kind="mergesort").reset_index(drop=True)
    te["p"], _ = model.predict_probs(te)

    pipelines = {"nearest_arc + greedy": (None, None), "nearest_arc + HMM": (None, 1.0),
                 "conditional_logit + greedy": ("p", None), "conditional_logit + HMM (w=1)": ("p", 1.0),
                 f"conditional_logit + HMM (w={w:g})": ("p", w)}
    picks = {k: [] for k in pipelines}
    cont = {k: [] for k in pipelines}
    for tid, c in te.groupby("trip_id", sort=True):
        c = c.reset_index(drop=True)
        arcs = load_trip(root, a.partition, tid).arcs
        for name, (col, weight) in pipelines.items():
            dec = RouteDecoder(emission_weight=weight or 1.0)
            r = dec.decode_candidates(c, arcs, None if col is None else c[col].to_numpy())
            route = r["greedy_route"] if weight is None else r["predicted_route"]
            picks[name].append(pd.DataFrame({"trip_id": tid, "point_idx": r["point_idx"], "arc_id": route}))
            cont[name].append((dec.compute_route_continuity(route, arcs, r["segment_id"]), r["num_points"]))

    usable = te[te.label_usable == 1]
    n_pts = usable.groupby(["trip_id", "point_idx"]).ngroups
    rows = []
    for name in pipelines:
        ch = pd.concat(picks[name]).assign(chosen=1)
        u = usable.merge(ch, on=["trip_id", "point_idx", "arc_id"], how="left")
        pred = u["chosen"].fillna(0).astype(int).to_numpy()
        y = u["label"].to_numpy()
        top1 = int(((pred == 1) & (y == 1)).sum()) / n_pts
        cv = np.average([x for x, _ in cont[name]], weights=[n for _, n in cont[name]])
        rows.append({"pipeline": name, "top1_acc": round(top1, 6),
                     "macro_f1": round(f1_score(y, pred, average="macro"), 6),
                     "route_continuity": round(cv, 6), "n_usable_points": n_pts})
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "outputs" / "hmm" / f"final_scores_{a.partition}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
