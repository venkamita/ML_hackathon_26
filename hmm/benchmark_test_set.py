"""
Benchmark: does the HMM/Viterbi decoder improve on picking the single most
likely candidate per GPS point? Run on the 10 student-test trips.

Decodes over Person B's exact candidate rows (outputs/features/
features_student_test.parquet), so Top-1 here uses the SAME definition as
the model comparison:
    a usable GPS point is correct if the chosen candidate row has label == 1;
    points whose true arc is not among the candidates count as wrong.

Emission sources compared:
  * conditional_logit : the selected model (lambda from Bayesian optimisation),
                        refitted on all usable TRAIN rows only
  * conditional_logit_w<w> : same, with the emission weight w tuned on
                        TRAIN trips (hmm/tune_emission_weight.py)
  * nearest_arc       : Gaussian distance emission, sigma = 6.68 m (no model)

For each: greedy (per-point argmax) vs HMM/Viterbi.

Run from the repo root:
    python -m hmm.benchmark_test_set --data-root ..
Writes outputs/hmm/benchmark_student_test.csv and outputs/hmm/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data_loader import load_trip, resolve_data_root  # noqa: E402
from hmm.viterbi import RouteDecoder  # noqa: E402
from models.baselines_linear import ConditionalLogitModel  # noqa: E402

TUNED_LAMBDA = 10 ** -1.9768   # best trial in outputs/models/conditional_logit/bayes_opt_trials.csv
SORT = ["trip_id", "point_idx", "cand_rank"]


def top1_on_usable(cands: pd.DataFrame, point_idx, chosen_arcs) -> tuple[int, int]:
    """(correct, n_usable_points) for one trip, same rule as the model metrics."""
    chosen = pd.DataFrame({"point_idx": point_idx, "arc_id": chosen_arcs})
    usable_pts = cands.loc[cands.label_usable == 1, "point_idx"].unique()
    hit = chosen.merge(cands[["point_idx", "arc_id", "label"]], on=["point_idx", "arc_id"], how="left")
    hit = hit[hit.point_idx.isin(usable_pts)]
    return int((hit.label == 1).sum()), int(len(usable_pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder with train/ and student_test/")
    ap.add_argument("--out", default="outputs/hmm")
    ap.add_argument("--partition", default="student_test",
                    help="student_test (development) or evaluator_test (final, frozen settings)")
    ap.add_argument("--features", default=None,
                    help="candidate/feature parquet; default outputs/features/features_<partition>.parquet")
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    feat_dir = REPO_ROOT / "outputs" / "features"
    print("Fitting conditional logit on usable TRAIN rows only ...")
    tr = pd.read_parquet(feat_dir / "features_train.parquet")
    tr = tr[tr.label_usable == 1].sort_values(SORT, kind="mergesort").reset_index(drop=True)
    model = ConditionalLogitModel(lmb=TUNED_LAMBDA, max_iter=100)
    fit_s = model.fit(tr)
    del tr
    print(f"  lambda={TUNED_LAMBDA:.4g}, fit {fit_s:.1f}s")

    te = pd.read_parquet(args.features or feat_dir / f"features_{args.partition}.parquet")
    te = te.sort_values(SORT, kind="mergesort").reset_index(drop=True)
    te["p_logit"], _ = model.predict_probs(te)          # all rows, incl. not-usable points

    wfile = REPO_ROOT / "outputs" / "hmm" / "emission_weight.json"   # tuned on train only
    w = json.loads(wfile.read_text())["emission_weight"] if wfile.exists() else 1.0
    print(f"Conditional-logit emission weight (tuned on train): {w}")
    decoders = {"conditional_logit": RouteDecoder(),                       # w = 1
                f"conditional_logit_w{w:g}": RouteDecoder(emission_weight=w),
                "nearest_arc": RouteDecoder()}
    decoder = decoders["nearest_arc"]
    rows = []
    totals = {}
    for tid, cands in te.groupby("trip_id", sort=True):
        cands = cands.reset_index(drop=True)
        trip = load_trip(root, args.partition, tid)
        for source, dec in decoders.items():
            probs = None if source == "nearest_arc" else cands["p_logit"].to_numpy()
            t0 = time.time()
            res = dec.decode_candidates(cands, trip.arcs, probs)
            dec_s = time.time() - t0
            for mode in ("greedy", "hmm"):
                route = res["greedy_route"] if mode == "greedy" else res["predicted_route"]
                c, n = top1_on_usable(cands, res["point_idx"], route)
                cont = decoder.compute_route_continuity(route, trip.arcs, res["segment_id"])
                rows.append({"trip_id": tid, "emission": source, "decoder": mode,
                             "n_points": res["num_points"], "n_usable_points": n,
                             "n_segments": res["num_segments"], "correct": c,
                             "top1_acc": c / n if n else np.nan, "route_continuity": cont,
                             "decode_s": dec_s if mode == "hmm" else 0.0})
                agg = totals.setdefault((source, mode), {"c": 0, "n": 0, "cont": [], "w": [], "s": 0.0})
                agg["c"] += c
                agg["n"] += n
                agg["cont"].append(cont)
                agg["w"].append(res["num_points"])
                agg["s"] += rows[-1]["decode_s"]
        print(f"  {tid}: " + "  ".join(
            f"{r['emission'][:5]}-{r['decoder']}={r['top1_acc']*100:.1f}%/{r['route_continuity']*100:.1f}%"
            for r in rows[-2 * len(decoders):]))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"benchmark_{args.partition}.csv", index=False)
    summary = {}
    for (source, mode), a in totals.items():
        summary[f"{source}+{mode}"] = {
            "top1_acc": round(a["c"] / a["n"], 6),
            "route_continuity_pointweighted": round(float(np.average(a["cont"], weights=a["w"])), 6),
            "n_usable_points": a["n"],
            "decode_seconds_total": round(a["s"], 2),
        }
    summary["_settings"] = {
        "candidates": "Person B, k=8, radius 50 m (features_student_test.parquet)",
        "conditional_logit_lambda": TUNED_LAMBDA,
        "conditional_logit_emission_weight_tuned_on_train": w,
        "sigma_z_m": decoder.sigma_z, "gap_split_s": decoder.gap_s,
        "transition_log_scores": {"same_arc": decoder.log_self, "direct_successor": decoder.log_direct,
                                  "two_hop": decoder.log_two_hop, "disconnected": decoder.disconnected_penalty},
        "top1_definition": "usable points; chosen candidate row has label == 1; missing true arc = wrong",
    }
    summary["_settings"]["partition"] = args.partition
    (out / ("summary.json" if args.partition == "student_test" else f"summary_{args.partition}.json")
     ).write_text(json.dumps(summary, indent=2))
    print("\n" + "\n".join(f"{k:28s} top1={v['top1_acc']*100:6.2f}%  continuity={v['route_continuity_pointweighted']*100:6.2f}%"
                           for k, v in summary.items() if not k.startswith("_")))
    print(f"Saved results for {args.partition} in {out}")


if __name__ == "__main__":
    main()
