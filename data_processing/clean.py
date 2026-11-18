"""PathSnap track cleaning (Person A).

Cleaning ideas adapted from the literature:
  * Newson & Krumm (2009), "Hidden Markov Map Matching Through Noise and
    Sparseness": robust GPS-noise estimate sigma_z = 1.4826 * median distance
    to the true road (MAD); 2*sigma_z proximity filter; 50 m/s implausible
    speed; split a trip when a gap exceeds 180 s.
    Adaptations: sigma is estimated per trip and the median taken; the
    2*sigma filter is only a flag (at 1 Hz it would remove most points);
    50 m/s marks only out-and-back spikes as errors, because sustained fast
    motorway driving exists in the training data.
  * Chao et al. (2019) map-matching survey: outliers cause "matching breaks"
    and should be identified before matching.
  * Map Matching 2 (Woltche): removal of spatially duplicate points.
  * Kubicka et al. (2015) dataset notes: reference routes are hand-corrected
    and contain some errors -> exclude far-off points from training labels.

The task needs a prediction for EVERY GPS point, so nothing is deleted here.
Every point is kept and gets flag columns; later stages decide what to do
(e.g. drop flagged points from training, fill them in at prediction time).

All thresholds that are learned from data (sigma_z, the label-exclusion
distance) are fitted on the TRAIN partition only and then applied unchanged
to student_test. The reference route is used only for the label-quality
columns (prefixed `label_`), which must never be used as model features.

Usage (from the pathsnap/ folder):
    python -m data_processing.clean --data-root ..
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import STRtree, linestrings, points
from shapely import distance as shp_distance

from common.data_loader import PARTITIONS, list_trips, load_trip, resolve_data_root
from common.geo import add_xy, arc_geometry, track_kinematics

MAD_SCALE = 1.4826            # MAD -> standard deviation for a Gaussian
MAX_SPEED_MPS = 50.0          # Newson & Krumm: faster than this is implausible
SPLIT_GAP_S = 180.0           # Newson & Krumm: split trip at gaps longer than this
LABEL_EXCLUDE_SIGMAS = 6.0    # far-from-route cut-off = max(6*sigma_z, 30 m)
LABEL_EXCLUDE_MIN_M = 30.0
RADII_M = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]


# --------------------------------------------------------------------------
def network_lines(geom):
    g = geom[geom["valid"]]
    lines = linestrings(np.stack([g[["x1", "y1"]].values,
                                  g[["x2", "y2"]].values], axis=1))
    return g.index.values, lines


def route_distance_and_rank(trip, geom, pts):
    """For each GPS point: distance to the nearest reference-route arc, and
    how many network arcs are strictly closer than that (0 = the route arc
    is the nearest arc). Used for sigma_z and for B's candidate k/radius."""
    arc_ids, lines = network_lines(geom)
    route_set = set(int(a) for a in trip.route.values)
    is_route = np.array([a in route_set for a in arc_ids])
    if not is_route.any():
        return np.full(len(pts), np.nan), np.full(len(pts), -1)

    _, d_route = STRtree(lines[is_route]).query_nearest(
        pts, return_distance=True, all_matches=False)

    # count network arcs within d_route of each point (minus a hair so the
    # route arc itself and exact ties are not counted)
    tree = STRtree(lines)
    pi, li = tree.query(pts, predicate="dwithin", distance=np.maximum(d_route - 1e-6, 0))
    dd = shp_distance(pts[pi], lines[li])
    closer = (dd < d_route[pi] - 1e-6) & ~is_route[li]
    rank = np.bincount(pi[closer], minlength=len(pts))
    return d_route, rank


def proximity_flag(x, y, min_dist):
    """Newson & Krumm proximity filter, as a flag: a point is 'redundant' if
    it lies within min_dist of the previous point that was kept."""
    flag = np.zeros(len(x), dtype=bool)
    if len(x) == 0:
        return flag
    kx, ky = x[0], y[0]
    for i in range(1, len(x)):
        if np.hypot(x[i] - kx, y[i] - ky) < min_dist:
            flag[i] = True
        else:
            kx, ky = x[i], y[i]
    return flag


def spike_flag(x, y, t, max_speed):
    """Out-and-back outlier: the point is reached AND left at an impossible
    speed, while its two neighbours are consistent with each other. A plain
    speed threshold is not used as an error, because genuine motorway driving
    can exceed it for minutes at a time (seen in the training data)."""
    n = len(x)
    flag = np.zeros(n, dtype=bool)
    if n < 3:
        return flag
    d_in = np.hypot(x[1:-1] - x[:-2], y[1:-1] - y[:-2])
    d_out = np.hypot(x[2:] - x[1:-1], y[2:] - y[1:-1])
    d_skip = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2])
    dt_in = np.maximum(t[1:-1] - t[:-2], 1e-9)
    dt_out = np.maximum(t[2:] - t[1:-1], 1e-9)
    v_in, v_out = d_in / dt_in, d_out / dt_out
    v_skip = d_skip / (dt_in + dt_out)
    flag[1:-1] = (v_in > max_speed) & (v_out > max_speed) & (v_skip <= max_speed)
    return flag


def clean_trip(trip, sigma_z, label_cut_m):
    tr = trip.track.copy()
    epsg = add_xy(tr)
    nodes = trip.nodes.copy()
    add_xy(nodes, epsg)
    geom = arc_geometry(nodes, trip.arcs)
    kin = track_kinematics(tr)

    out = pd.DataFrame({"point_idx": np.arange(len(tr)), "t": tr["t"],
                        "lon": tr["lon"], "lat": tr["lat"],
                        "x": tr["x"], "y": tr["y"]})
    out["dt"] = kin["dt"]
    out["step_m"] = kin["step_m"]
    out["speed_mps"] = kin["speed_mps"]
    out["heading_deg"] = kin["heading_deg"]

    out["flag_nan"] = tr[["lon", "lat", "t"]].isna().any(axis=1)
    out["flag_out_of_range"] = ~(tr["lon"].between(-180, 180) & tr["lat"].between(-90, 90))
    out["flag_bad_time"] = kin["dt"].fillna(1) <= 0          # duplicate / backwards
    out["flag_duplicate_pos"] = tr.duplicated(["lon", "lat"], keep="first")
    out["flag_speed_over_50mps"] = kin["speed_mps"].fillna(0) > MAX_SPEED_MPS  # info only
    out["flag_spike"] = spike_flag(tr["x"].values, tr["y"].values, tr["t"].values,
                                   MAX_SPEED_MPS)
    out["flag_redundant_2sigma"] = proximity_flag(tr["x"].values, tr["y"].values,
                                                  2 * sigma_z)
    out["gap_before_s"] = kin["dt"].fillna(0)
    out["segment_id"] = (out["gap_before_s"] > SPLIT_GAP_S).cumsum()

    # "usable for training" = none of the hard-error flags
    hard = ["flag_nan", "flag_out_of_range", "flag_bad_time", "flag_spike"]
    out["is_clean"] = ~out[hard].any(axis=1)

    if trip.has_route:
        ok = out[["x", "y"]].notna().all(axis=1).values
        pts = points(out.loc[ok, "x"].values, out.loc[ok, "y"].values)
        d, rank = route_distance_and_rank(trip, geom, pts)
        out["label_dist_to_route_m"] = np.nan
        out["label_route_rank"] = -1
        out.loc[ok, "label_dist_to_route_m"] = d
        out.loc[ok, "label_route_rank"] = rank
        out["label_exclude_far"] = out["label_dist_to_route_m"].fillna(np.inf) > label_cut_m
    return out


# --------------------------------------------------------------------------
def estimate_sigma(root, trips):
    """sigma_z from TRAIN only: 1.4826 * median distance GPS -> route,
    computed per trip, then the median across trips."""
    ds, per_trip = [], []
    for tid in trips:
        trip = load_trip(root, "train", tid)
        if not trip.has_route:
            continue
        epsg = add_xy(trip.track)
        add_xy(trip.nodes, epsg)
        geom = arc_geometry(trip.nodes, trip.arcs)
        tr = trip.track.dropna(subset=["x", "y"])
        arc_ids, lines = network_lines(geom)
        route_set = set(int(a) for a in trip.route.values)
        is_route = np.array([a in route_set for a in arc_ids])
        if not is_route.any():
            continue
        _, d = STRtree(lines[is_route]).query_nearest(
            points(tr["x"].values, tr["y"].values), return_distance=True,
            all_matches=False)
        ds.append(d)
        per_trip.append({"trip_id": tid, "n": len(d),
                         "median_dist_m": float(np.median(d)),
                         "sigma_m": float(MAD_SCALE * np.median(d))})
    per_trip = pd.DataFrame(per_trip)
    # median of per-trip estimates: one trip with a bad reference route
    # (e.g. most of its GPS far from the route) cannot drag the estimate.
    return float(per_trip["sigma_m"].median()), np.concatenate(ds), per_trip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/clean")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    out = Path(args.out)
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train_ids = list_trips(root, "train")
    sigma_z, d_train, sig_trips = estimate_sigma(root, train_ids)
    sig_trips["suspect_reference"] = sig_trips["median_dist_m"] > 3 * sigma_z
    sig_trips.to_csv(out / "sigma_per_trip_train.csv", index=False)
    label_cut = max(LABEL_EXCLUDE_SIGMAS * sigma_z, LABEL_EXCLUDE_MIN_M)
    config = {
        "fitted_on": "train",
        "n_train_trips": len(train_ids),
        "n_train_points": int(len(d_train)),
        "sigma_z_m": round(sigma_z, 3),
        "sigma_method": "median over train trips of 1.4826 * median(distance GPS -> nearest reference-route arc)",
        "suspect_reference_trips": sig_trips.loc[sig_trips.suspect_reference, "trip_id"].tolist(),
        "redundant_point_dist_m": round(2 * sigma_z, 3),
        "max_speed_mps": MAX_SPEED_MPS,
        "split_gap_s": SPLIT_GAP_S,
        "label_exclude_far_m": round(label_cut, 3),
    }
    (out / "cleaning_config.json").write_text(json.dumps(config, indent=2))
    print(f"sigma_z (train) = {sigma_z:.2f} m; label cut-off = {label_cut:.1f} m")

    summary, ranks, dists = [], {p: [] for p in PARTITIONS}, {p: [] for p in PARTITIONS}
    for p in PARTITIONS:
        for tid in list_trips(root, p):
            trip = load_trip(root, p, tid)
            c = clean_trip(trip, sigma_z, label_cut)
            c.to_csv(out / "tracks" / f"{p}_{tid}.csv", index=False)
            row = {"partition": p, "trip_id": tid, "n_points": len(c),
                   "n_segments": int(c["segment_id"].nunique())}
            for col in [k for k in c.columns if k.startswith("flag_")]:
                row[col] = int(c[col].sum())
            row["n_clean"] = int(c["is_clean"].sum())
            if "label_exclude_far" in c:
                row["n_label_exclude_far"] = int(c["label_exclude_far"].sum())
                row["pct_label_exclude_far"] = float(c["label_exclude_far"].mean() * 100)
                keep = c["is_clean"] & ~c["label_exclude_far"]
                ranks[p].append(c.loc[keep, "label_route_rank"].values)
                dists[p].append(c.loc[keep, "label_dist_to_route_m"].values)
            summary.append(row)
            print(f"  {p}/{tid}: {len(c)} pts, {row['n_segments']} segment(s)")

    s = pd.DataFrame(summary)
    s.to_csv(out / "cleaning_summary.csv", index=False)

    # Candidate-generation guidance for Person B (train only!)
    r = np.concatenate(ranks["train"]) if ranks["train"] else np.array([])
    d = np.concatenate(dists["train"]) if dists["train"] else np.array([])
    if len(r):
        k_tab = pd.DataFrame({"k": K_VALUES,
                              "miss_rate_pct": [float((r >= k).mean() * 100) for k in K_VALUES]})
        r_tab = pd.DataFrame({"radius_m": RADII_M,
                              "miss_rate_pct": [float((d > R).mean() * 100) for R in RADII_M]})
        k_tab.to_csv(out / "candidate_miss_rate_by_k_train.csv", index=False)
        r_tab.to_csv(out / "candidate_miss_rate_by_radius_train.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(k_tab.k, k_tab.miss_rate_pct, marker="o")
        axes[0].set(xlabel="k nearest arcs kept", ylabel="Reference arc missed (%)",
                    title="Candidate miss rate vs k - train")
        axes[1].plot(r_tab.radius_m, r_tab.miss_rate_pct, marker="o")
        axes[1].set(xlabel="Search radius (m)", ylabel="Reference arc missed (%)",
                    title="Candidate miss rate vs radius - train")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "plots" / "candidate_miss_rate_train.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(d_train, 0, 40), bins=80)
    ax.axvline(sigma_z, color="r", ls="--", label=f"sigma_z = {sigma_z:.2f} m")
    ax.axvline(label_cut if label_cut < 40 else 40, color="k", ls=":",
               label=f"label cut-off = {label_cut:.0f} m")
    ax.set(xlabel="Distance GPS -> reference route (m, 40 = 40 or more)",
           ylabel="GPS fixes", title="GPS noise estimate - train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plots" / "sigma_estimate_train.png", dpi=130)
    plt.close(fig)

    tot = s.groupby("partition")[[c for c in s.columns if c.startswith(("flag_", "n_"))]].sum()
    print(tot.T.to_string())
    print(f"Done in {time.time() - t0:.1f}s. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
