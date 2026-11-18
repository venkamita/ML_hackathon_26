"""
Quick check of the HMM/Viterbi decoder on one trip.

Run from the repo root:
    python -m hmm.test_viterbi --data-root .. [--partition student_test --trip 00000052]
"""
import argparse
import time

from common.data_loader import load_trip, resolve_data_root
from hmm.viterbi import RouteDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trip", default="00000052")
    a = ap.parse_args()

    trip = load_trip(resolve_data_root(a.data_root), a.partition, a.trip)
    dec = RouteDecoder()                       # Gaussian distance emission, sigma 6.68 m
    t0 = time.time()
    res = dec.decode_trip(trip)                # Person B's candidates, no model needed
    dt = time.time() - t0
    seg = res["segment_id"]
    print(f"Trip {a.partition}/{a.trip}: {res['num_points']} points, {res['num_segments']} segment(s), "
          f"decoded in {dt:.2f}s ({dt * 1000 / max(res['num_points'], 1):.2f} ms/point)")
    print(f"Continuity  greedy {dec.compute_route_continuity(res['greedy_route'], trip.arcs, seg) * 100:.2f}%"
          f"  ->  HMM {dec.compute_route_continuity(res['predicted_route'], trip.arcs, seg) * 100:.2f}%")


if __name__ == "__main__":
    main()
