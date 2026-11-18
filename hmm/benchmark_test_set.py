"""
Benchmark Route Reconstruction across all 10 student-test trips.
Person D component: Compares Greedy selection vs HMM/Viterbi sequence decoding
on accuracy, route continuity, and latency.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data_loader import load_trip, list_trips
from common.geo import add_xy, arc_geometry
from hmm.viterbi import RouteDecoder


def run_benchmark():
    decoder = RouteDecoder(radius_m=50.0, max_k=8, sigma_z=4.07)
    trips = list_trips(PROJECT_ROOT, "student_test")
    print(f"Benchmarking {len(trips)} student-test trips...")

    records = []

    for tid in trips:
        t0 = time.time()
        trip = load_trip(PROJECT_ROOT, "student_test", tid)
        epsg = add_xy(trip.track)
        add_xy(trip.nodes, epsg)
        arcs_geom = arc_geometry(trip.nodes, trip.arcs)
        t_prep = time.time() - t0

        t_dec0 = time.time()
        res = decoder.decode(trip.track, trip.arcs, arcs_geom)
        t_decode = time.time() - t_dec0

        n_pts = res["num_points"]
        greedy_route = res["greedy_route"]
        viterbi_route = res["predicted_route"]

        greedy_cont = decoder.compute_route_continuity(greedy_route, trip.arcs)
        viterbi_cont = decoder.compute_route_continuity(viterbi_route, trip.arcs)

        # Ground truth route comparison
        ref_arcs_set = set(trip.route.values) if trip.has_route else set()
        greedy_acc = sum(1 for a in greedy_route if a in ref_arcs_set) / n_pts if ref_arcs_set else None
        viterbi_acc = sum(1 for a in viterbi_route if a in ref_arcs_set) / n_pts if ref_arcs_set else None

        records.append({
            "trip_id": tid,
            "gps_points": n_pts,
            "greedy_accuracy": round(greedy_acc * 100, 2) if greedy_acc else None,
            "viterbi_accuracy": round(viterbi_acc * 100, 2) if viterbi_acc else None,
            "accuracy_delta": round((viterbi_acc - greedy_acc) * 100, 2) if greedy_acc and viterbi_acc else None,
            "greedy_continuity": round(greedy_cont * 100, 2),
            "viterbi_continuity": round(viterbi_cont * 100, 2),
            "continuity_delta": round((viterbi_cont - greedy_cont) * 100, 2),
            "prep_time_s": round(t_prep, 2),
            "decode_time_s": round(t_decode, 2),
            "ms_per_point": round((t_decode * 1000) / n_pts, 2),
        })

        print(
            f"Trip {tid:8s} | Points: {n_pts:4d} | "
            f"Cont: {greedy_cont*100:5.1f}% -> {viterbi_cont*100:5.1f}% (+{(viterbi_cont-greedy_cont)*100:4.1f}%) | "
            f"Acc: {greedy_acc*100:5.1f}% -> {viterbi_acc*100:5.1f}% | "
            f"Time: {t_decode*1000:6.1f}ms"
        )

    df = pd.DataFrame(records)
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "student_test_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to {out_csv}")

    print("\n--- Summary Statistics (Mean across all 10 trips) ---")
    summary = df[[
        "greedy_continuity", "viterbi_continuity", "continuity_delta",
        "greedy_accuracy", "viterbi_accuracy", "accuracy_delta", "ms_per_point"
    ]].mean()
    print(summary.to_string())

    return df


if __name__ == "__main__":
    run_benchmark()
