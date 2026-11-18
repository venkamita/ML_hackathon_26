"""PathSnap - Person B: candidate generation, labels and features.

For every GPS point:
  1. Candidates: all arcs within RADIUS_M (UTM metres), keep the K nearest.
     A point with nothing inside the radius gets its single nearest arc
     (is_fallback = 1) so the HMM always has at least one state.
  2. Label: the reference route is aligned to the GPS points IN ROUTE ORDER
     (monotone dynamic programming: point i can only match route position
     >= the position of point i-1). This avoids labelling the wrong arc where
     the route passes the same place twice (loops, overpasses, a road driven
     in both directions). The candidate whose arc_id equals the aligned route
     arc is the positive (label = 1); all other candidates are negatives.
  3. Features for every (point, candidate) pair - built only from GPS + road
     network, never from the route, so the same function runs on evaluator
     trips that have no .route file.

Run (from the repo root, same --data-root as Person A):
    python -m features.build_candidates --data-root .. --max-trips 3   # quick file
    python -m features.build_candidates --data-root ..                 # everything

Outputs (outputs/features/):
    features_<partition>.parquet   one row per (point, candidate)
    candidate_stats_<partition>.csv   miss rate etc. per trip
    miss_rate_table_train.csv      miss rate for smaller k / radius (report)
    feature_config.json            the k, radius and column lists used

For Person C:   from features.build_candidates import FEATURE_COLS
Train on rows where label_usable == 1. Group CV by trip_id.
Columns starting with label_ / point_has_positive must NEVER be features.

For Person D (API / HMM):   build_trip_candidates(trip, k, radius) works on a
trip without a route and returns the same feature columns.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import STRtree, linestrings, points

from common.data_loader import PARTITIONS, list_trips, load_trip, resolve_data_root
from common.geo import add_xy, angle_diff_deg, arc_geometry, point_to_segments

DEFAULT_K = 8
DEFAULT_RADIUS_M = 50.0
MIN_MOVE_FOR_HEADING_M = 1.0   # below this the GPS heading is noise
NEIGHBOUR_OFFSETS = (-2, -1, 1, 2)
DEFAULT_LABEL_CUT_M = 30.0     # used if Person A's cleaning_config.json is missing

FEATURE_COLS = [
    # geometry of this candidate vs this point
    "dist_m", "cand_rank", "dist_minus_min_m", "dist_ratio_min",
    "n_cands_in_radius", "frac_along", "frac_to_end",
    "arc_length_m",
    # direction
    "heading_diff_deg", "heading_diff_undirected_deg", "heading_valid",
    # point kinematics
    "speed_mps", "step_next_m", "dt_prev_s", "gap_before",
    # temporal context: distance from neighbouring GPS points to the SAME arc
    "d_prev2_m", "d_prev1_m", "d_next1_m", "d_next2_m", "d_nbr_mean_m",
    # network
    "deg_u", "deg_v", "has_reverse_twin", "is_fallback",
]
META_COLS = ["partition", "trip_id", "point_idx", "t", "x", "y", "arc_id"]
LABEL_COLS = ["label", "label_arc_id", "label_route_pos", "label_dist_m",
              "label_usable", "point_has_positive"]


# --------------------------------------------------------------------------
# helpers
def _pair_dist(px, py, x1, y1, x2, y2):
    """Distance and projection fraction for aligned arrays (one point per segment)."""
    sx, sy = x2 - x1, y2 - y1
    l2 = sx * sx + sy * sy
    with np.errstate(divide="ignore", invalid="ignore"):
        f = ((px - x1) * sx + (py - y1) * sy) / l2
    f = np.where(l2 > 0, np.clip(f, 0.0, 1.0), 0.0)
    return np.hypot(px - (x1 + f * sx), py - (y1 + f * sy)), f


def _point_kinematics(x, y, t):
    """Centred heading/speed (uses i-1 and i+1: less noisy than a one-sided diff)."""
    n = len(x)
    i = np.arange(n)
    ip, inx = np.clip(i - 1, 0, n - 1), np.clip(i + 1, 0, n - 1)
    dx, dy = x[inx] - x[ip], y[inx] - y[ip]
    move = np.hypot(dx, dy)
    dt_c = t[inx] - t[ip]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt_c > 0, move / dt_c, 0.0)
    heading = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    valid = move >= MIN_MOVE_FOR_HEADING_M
    step_next = np.hypot(x[inx] - x, y[inx] - y)
    dt_prev = np.concatenate([[0.0], np.diff(t)])
    return heading, valid, speed, step_next, dt_prev


def _network_tables(trip, geom):
    arcs = trip.arcs
    deg = pd.concat([arcs["u"], arcs["v"]]).value_counts()
    deg_u = arcs["u"].map(deg).fillna(0).values
    deg_v = arcs["v"].map(deg).fillna(0).values
    fwd = set(zip(arcs["u"].values, arcs["v"].values))
    twin = np.array([(v, u) in fwd for u, v in zip(arcs["u"].values, arcs["v"].values)])
    return deg_u, deg_v, twin


# --------------------------------------------------------------------------
# candidate generation + features (no route needed)
def build_trip_candidates(trip, k=DEFAULT_K, radius_m=DEFAULT_RADIUS_M):
    tr = trip.track.reset_index(drop=True).copy()
    epsg = add_xy(tr)          # UTM zone chosen per trip from its mean lon/lat
    tr.attrs["epsg"] = epsg
    nodes = trip.nodes.copy()
    add_xy(nodes, epsg)
    geom = arc_geometry(nodes, trip.arcs)

    x, y, t = tr["x"].values, tr["y"].values, tr["t"].values
    n = len(tr)
    heading, h_valid, speed, step_next, dt_prev = _point_kinematics(x, y, t)

    g = geom[geom["valid"]]
    arc_ids = g.index.values
    tree = STRtree(linestrings(np.stack([g[["x1", "y1"]].values,
                                         g[["x2", "y2"]].values], axis=1)))
    ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    pts = points(x[ok], y[ok])

    qi, li = tree.query(pts, predicate="dwithin", distance=radius_m)
    p_idx, a_idx = ok[qi], arc_ids[li]
    fallback = np.zeros(len(p_idx), dtype=bool)

    has = np.zeros(n, dtype=bool)
    has[p_idx] = True
    lonely = ok[~has[ok]]
    if len(lonely):
        nq = tree.query_nearest(points(x[lonely], y[lonely]), all_matches=False)
        p_idx = np.concatenate([p_idx, lonely[nq[0]]])
        a_idx = np.concatenate([a_idx, arc_ids[nq[1]]])
        fallback = np.concatenate([fallback, np.ones(len(nq[0]), dtype=bool)])

    G = geom.loc[a_idx]
    x1, y1, x2, y2 = (G[c].values for c in ("x1", "y1", "x2", "y2"))
    d, frac = _pair_dist(x[p_idx], y[p_idx], x1, y1, x2, y2)

    df = pd.DataFrame({"point_idx": p_idx, "arc_id": a_idx, "dist_m": d,
                       "frac_along": frac, "is_fallback": fallback.astype(np.int8)})
    df["n_cands_in_radius"] = df.groupby("point_idx")["arc_id"].transform("size")
    df = df.sort_values(["point_idx", "dist_m", "arc_id"], kind="mergesort")
    df["cand_rank"] = df.groupby("point_idx").cumcount()
    df = df[df["cand_rank"] < k].reset_index(drop=True)

    p, a = df["point_idx"].values, df["arc_id"].values
    G = geom.loc[a]
    x1, y1, x2, y2 = (G[c].values for c in ("x1", "y1", "x2", "y2"))

    dmin = df.groupby("point_idx")["dist_m"].transform("min").values
    df["dist_minus_min_m"] = df["dist_m"].values - dmin
    df["dist_ratio_min"] = df["dist_m"].values / (dmin + 1.0)
    df["frac_to_end"] = np.minimum(df["frac_along"], 1 - df["frac_along"])
    df["arc_length_m"] = G["length_m"].values

    hd = angle_diff_deg(heading[p], G["bearing_deg"].values)
    valid = h_valid[p]
    df["heading_diff_deg"] = np.where(valid, hd, 90.0)          # 90 = "no information"
    df["heading_diff_undirected_deg"] = np.where(valid, np.minimum(hd, 180 - hd), 45.0)
    df["heading_valid"] = valid.astype(np.int8)

    df["speed_mps"] = speed[p]
    df["step_next_m"] = step_next[p]
    df["dt_prev_s"] = dt_prev[p]
    df["gap_before"] = (dt_prev[p] > 1.5).astype(np.int8)

    nb = []
    for off in NEIGHBOUR_OFFSETS:
        q = np.clip(p + off, 0, n - 1)
        dq, _ = _pair_dist(x[q], y[q], x1, y1, x2, y2)
        name = f"d_{'prev' if off < 0 else 'next'}{abs(off)}_m"
        df[name] = dq
        nb.append(dq)
    df["d_nbr_mean_m"] = np.nanmean(np.vstack(nb + [df["dist_m"].values]), axis=0)

    deg_u, deg_v, twin = _network_tables(trip, geom)
    df["deg_u"] = deg_u[a]
    df["deg_v"] = deg_v[a]
    df["has_reverse_twin"] = twin[a].astype(np.int8)

    df.insert(0, "trip_id", trip.trip_id)
    df.insert(0, "partition", trip.partition)
    df["t"] = t[p]
    df["x"] = x[p]
    df["y"] = y[p]
    return df, tr, geom


# --------------------------------------------------------------------------
# labels (route-ordered alignment)
def align_route(x, y, route, geom, cap_m=100.0, chunk=2000):
    """Monotone DP: assign each GPS point a route position so positions never
    go backwards and the summed (capped) distance is minimal.
    Returns route position per point and the raw distance to that arc.
    Built in float32 chunks: the longest trips are ~19k points x ~2.3k arcs."""
    r = np.asarray(route, dtype=np.int64)
    in_range = (r >= 0) & (r < len(geom))
    rv = np.where(in_range, r, 0)
    G = geom.iloc[rv]
    gx1, gy1, gx2, gy2 = (G[c].values for c in ("x1", "y1", "x2", "y2"))
    bad_arc = ~in_range | ~G["valid"].values
    finite = np.isfinite(x) & np.isfinite(y)

    P, R = len(x), len(r)
    D = np.empty((P, R), dtype=np.float32)
    for s0 in range(0, P, chunk):
        d, _ = point_to_segments(x[s0:s0 + chunk], y[s0:s0 + chunk], gx1, gy1, gx2, gy2)
        D[s0:s0 + chunk] = np.minimum(np.nan_to_num(d, nan=1e6), cap_m)
    D[:, bad_arc] = 1e6
    D[~finite, :] = 0.0

    back = np.zeros((P, R), dtype=np.int32)
    dp = D[0].astype(np.float64)
    ar = np.arange(R)
    for i in range(1, P):
        pm = np.minimum.accumulate(dp)
        back[i] = np.maximum.accumulate(np.where(dp == pm, ar, 0))
        dp = D[i] + pm
    del D
    pos = np.empty(P, dtype=np.int64)
    pos[-1] = int(np.argmin(dp))
    for i in range(P - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist, _ = _pair_dist(x, y, gx1[pos], gy1[pos], gx2[pos], gy2[pos])
    return pos, dist


def add_labels(df, tr, geom, route, label_cut_m, arcs=None, suspect=False):
    x, y = tr["x"].values, tr["y"].values
    pos, dist = align_route(x, y, route.values, geom)
    lab_arc = route.values[pos]
    p = df["point_idx"].values
    df["label_arc_id"] = lab_arc[p]
    df["label_route_pos"] = pos[p]
    df["label_dist_m"] = dist[p]
    if arcs is not None:
        # some networks contain duplicate arcs (same from/to node, different id);
        # they are geometrically identical, so they count as positives too
        u, v = arcs["u"].values, arcs["v"].values
        a, la = df["arc_id"].values, lab_arc[p]
        df["label"] = ((u[a] == u[la]) & (v[a] == v[la])).astype(np.int8)
    else:
        df["label"] = (df["arc_id"].values == lab_arc[p]).astype(np.int8)
    df["point_has_positive"] = df.groupby("point_idx")["label"].transform("max").astype(np.int8)
    df["label_usable"] = ((df["label_dist_m"] <= label_cut_m) & (not suspect)).astype(np.int8)
    point_tbl = pd.DataFrame({"point_idx": np.arange(len(tr)), "label_arc_id": lab_arc,
                              "label_dist_m": dist})
    return df, point_tbl


# --------------------------------------------------------------------------
def _write(df, path_no_ext):
    try:
        df.to_parquet(f"{path_no_ext}.parquet", index=False)
        return f"{path_no_ext}.parquet"
    except ImportError:
        df.to_csv(f"{path_no_ext}.csv.gz", index=False)
        return f"{path_no_ext}.csv.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/features")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M)
    ap.add_argument("--partitions", nargs="+", default=list(PARTITIONS))
    ap.add_argument("--max-trips", type=int, default=None, help="quick run on the first N trips")
    ap.add_argument("--clean-config", default="outputs/clean/cleaning_config.json")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.clean_config)
    label_cut, suspect_trips = DEFAULT_LABEL_CUT_M, set()
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        label_cut = float(cfg["label_exclude_far_m"])
        suspect_trips = set(cfg.get("suspect_reference_trips", []))
    else:
        print(f"WARNING: {cfg_path} not found, run data_processing.clean first. Using {label_cut} m")
    print(f"k={args.k}  radius={args.radius} m  label cut-off={label_cut:.1f} m  "
          f"suspect trips (not used for training): {sorted(suspect_trips)}")

    t0 = time.time()
    for part in args.partitions:
        trips = list_trips(root, part)[: args.max_trips]
        frames, stats, pos_rank, pos_dist = [], [], [], []
        for tid in trips:
            ts = time.time()
            trip = load_trip(root, part, tid)
            df, tr, geom = build_trip_candidates(trip, args.k, args.radius)
            row = {"trip_id": tid, "epsg": int(tr.attrs.get("epsg", 0)), "n_points": len(tr), "n_rows": len(df),
                   "cands_per_point": len(df) / max(len(tr), 1),
                   "n_fallback_points": int(df.loc[df.is_fallback == 1, "point_idx"].nunique())}
            if trip.has_route:
                df, ptbl = add_labels(df, tr, geom, trip.route, label_cut, trip.arcs,
                                     suspect=(part == "train" and tid in suspect_trips))
                pts = df.groupby("point_idx").agg(hp=("point_has_positive", "max"),
                                                 use=("label_usable", "max"))
                use = pts[pts.use == 1]
                row["miss_rate_all_pct"] = 100 * (1 - pts.hp.mean())
                row["miss_rate_usable_pct"] = 100 * (1 - use.hp.mean()) if len(use) else np.nan
                row["pct_points_usable"] = 100 * pts.use.mean()
                row["median_label_dist_m"] = float(np.median(ptbl.label_dist_m))
                # for the k / radius table: rank & dist of the true arc per point
                pos = (df[(df.label == 1) & (df.label_usable == 1)]
                       .sort_values("cand_rank").drop_duplicates("point_idx"))
                pos_rank.append(pos.cand_rank.values)
                pos_dist.append(pos.dist_m.values)
                pos_rank.append(np.full(len(use) - len(pos), 10 ** 6))   # true arc not found
                pos_dist.append(np.full(len(use) - len(pos), np.inf))
            row["seconds"] = round(time.time() - ts, 2)
            stats.append(row)
            frames.append(df)
            print(f"  {part}/{tid}: {len(tr)} pts -> {len(df)} rows"
                  + (f", miss {row['miss_rate_usable_pct']:.2f}%" if "miss_rate_usable_pct" in row else "")
                  + f"  ({row['seconds']}s)")

        if not frames:
            continue
        allf = pd.concat(frames, ignore_index=True)
        f64 = [c for c in allf.select_dtypes("float64").columns if c not in ("x", "y", "t")]
        allf[f64] = allf[f64].astype(np.float32)
        cols = META_COLS + FEATURE_COLS + [c for c in LABEL_COLS if c in allf]
        path = _write(allf[cols], out / f"features_{part}")
        st = pd.DataFrame(stats)
        st.to_csv(out / f"candidate_stats_{part}.csv", index=False)
        msg = f"{part}: {len(allf):,} rows from {len(trips)} trips -> {path}"
        if "label" in allf:
            u = allf[allf.label_usable == 1].groupby(["trip_id", "point_idx"]).label.max()
            msg += f" | overall miss rate (usable points) = {100 * (1 - u.mean()):.2f}%"
            msg += f" | positives = {allf.label.mean() * 100:.1f}% of rows"
        print(msg)

        if pos_rank and part == "train":
            r, dd = np.concatenate(pos_rank), np.concatenate(pos_dist)
            radii = [rr for rr in (10, 15, 20, 25, 30, 40, 50, 75, 100) if rr <= args.radius]
            tab = pd.DataFrame([{"k": kk, "radius_m": rr,
                                 "miss_rate_pct": 100 * float(np.mean((r >= kk) | (dd > rr)))}
                                for kk in range(1, args.k + 1) for rr in radii])
            tab.to_csv(out / "miss_rate_table_train.csv", index=False)

    (out / "feature_config.json").write_text(json.dumps({
        "k": args.k, "radius_m": args.radius, "label_cut_m": label_cut,
        "crs": "UTM, zone chosen per trip from its mean lon/lat (trips are not all in zone 36N)",
        "label_method": "monotone DP alignment of GPS points to the reference route (route order kept)",
        "feature_cols": FEATURE_COLS, "meta_cols": META_COLS, "label_cols": LABEL_COLS,
    }, indent=2))
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
