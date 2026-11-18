"""PathSnap dataset exploration (Person A).

Usage (from the pathsnap/ folder):
    python -m eda.run_eda --data-root "D:/Dowloads/Downloads/Student folder"

Writes CSV tables, PNG plots, Folium maps and a data-quality summary to
outputs/eda/. Only the train and student_test partitions are read.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely import STRtree, linestrings, points

from common.data_loader import (PARTITIONS, file_inventory, list_trips,
                                load_trip, resolve_data_root)
from common.geo import add_xy, arc_geometry, track_kinematics
from viz.folium_map import make_trip_map

SPEED_JUMP_MPS = 50.0          # > 180 km/h between fixes = implausible jump
GAP_THRESHOLDS = (1, 5, 30)    # seconds


# --------------------------------------------------------------------------
# Per-trip statistics
# --------------------------------------------------------------------------
def track_stats(trip, kin) -> dict:
    tr = trip.track
    lon_bad = ~tr["lon"].between(-180, 180)
    lat_bad = ~tr["lat"].between(-90, 90)
    dt = kin["dt"].dropna()
    s = {
        "n_points": len(tr),
        "n_nan_lon": int(tr["lon"].isna().sum()),
        "n_nan_lat": int(tr["lat"].isna().sum()),
        "n_nan_t": int(tr["t"].isna().sum()),
        "n_out_of_range": int((lon_bad | lat_bad).sum()),
        "n_duplicate_t": int(tr["t"].duplicated().sum()),
        "n_non_increasing_t": int((dt <= 0).sum()),
        "n_duplicate_positions": int(tr.duplicated(["lon", "lat"]).sum()),
        "duration_s": float(tr["t"].max() - tr["t"].min()),
        "dt_median": float(dt.median()),
        "pct_dt_exactly_1s": float((dt == 1).mean() * 100),
        "max_gap_s": float(dt.max()),
        "path_km": float(kin["step_m"].sum() / 1000),
        "median_speed_kmh": float(kin["speed_mps"].median() * 3.6),
        "p95_speed_kmh": float(kin["speed_mps"].quantile(0.95) * 3.6),
        "n_speed_jumps": int((kin["speed_mps"] > SPEED_JUMP_MPS).sum()),
        "pct_stationary": float((kin["step_m"] < 0.5).mean() * 100),
        "bbox_w_km": float((tr["x"].max() - tr["x"].min()) / 1000),
        "bbox_h_km": float((tr["y"].max() - tr["y"].min()) / 1000),
        "lon_min": tr["lon"].min(), "lon_max": tr["lon"].max(),
        "lat_min": tr["lat"].min(), "lat_max": tr["lat"].max(),
    }
    for g in GAP_THRESHOLDS:
        s[f"n_gaps_gt_{g}s"] = int((dt > g).sum())
    return s


def network_stats(trip, geom) -> dict:
    nodes, arcs = trip.nodes, trip.arcs
    n = len(nodes)
    valid = geom["valid"]
    a = arcs[valid]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(a["u"], a["v"]))
    pairs = set(zip(a["u"], a["v"]))
    bidir = sum((v, u) in pairs for u, v in pairs)
    wcc = list(nx.weakly_connected_components(G))
    scc = list(nx.strongly_connected_components(G))
    used = set(a["u"]) | set(a["v"])
    return {
        "n_nodes": n,
        "n_arcs": len(arcs),
        "n_invalid_arc_refs": int((~valid).sum()),
        "n_self_loops": int((a["u"] == a["v"]).sum()),
        "n_duplicate_arcs": int(a.duplicated(["u", "v"]).sum()),
        "pct_arcs_bidirectional": float(bidir / max(len(pairs), 1) * 100),
        "n_isolated_nodes": int(n - len(used)),
        "n_weak_components": len(wcc),
        "largest_wcc_pct_nodes": float(max(map(len, wcc)) / n * 100),
        "n_strong_components": len(scc),
        "largest_scc_pct_nodes": float(max(map(len, scc)) / n * 100),
        "arc_len_median_m": float(geom.loc[valid, "length_m"].median()),
        "arc_len_max_m": float(geom.loc[valid, "length_m"].max()),
        "n_zero_length_arcs": int((geom.loc[valid, "length_m"] < 0.01).sum()),
    }


def route_stats(trip, geom) -> dict:
    if not trip.has_route:
        return {"has_route": False}
    r = trip.route.values
    in_range = (r >= 0) & (r < len(trip.arcs))
    rv = r[in_range]
    u = trip.arcs["u"].values
    v = trip.arcs["v"].values
    # consecutive arcs are connected if the end node of one is the start of the next
    joins = v[rv[:-1]] == u[rv[1:]] if len(rv) > 1 else np.array([])
    return {
        "has_route": True,
        "route_n_arcs": len(r),
        "route_n_unique": int(len(np.unique(r))),
        "route_n_invalid_ids": int((~in_range).sum()),
        "route_n_consecutive_repeats": int((np.diff(r) == 0).sum()),
        "route_pct_connected_steps": float(joins.mean() * 100) if len(joins) else np.nan,
        "route_n_breaks": int((~joins).sum()) if len(joins) else 0,
        "route_km": float(geom.loc[rv, "length_m"].sum() / 1000),
    }


def gps_to_network(trip, geom) -> tuple[dict, np.ndarray, np.ndarray]:
    """Distance from each GPS fix to the nearest arc of the whole network,
    and to the nearest arc of the reference route (if there is one)."""
    g = geom[geom["valid"]]
    lines = linestrings(np.stack([g[["x1", "y1"]].values,
                                  g[["x2", "y2"]].values], axis=1))
    tr = trip.track.dropna(subset=["x", "y"])
    pts = points(tr["x"].values, tr["y"].values)
    tree = STRtree(lines)
    _, d_any = tree.query_nearest(pts, return_distance=True, all_matches=False)
    out = {
        "gps_to_net_median_m": float(np.median(d_any)),
        "gps_to_net_p95_m": float(np.percentile(d_any, 95)),
        "gps_to_net_max_m": float(np.max(d_any)),
        "pct_gps_gt_30m_from_net": float((d_any > 30).mean() * 100),
    }
    d_route = np.array([])
    if trip.has_route:
        rv = [a for a in np.unique(trip.route.values) if a in g.index]
        rg = g.loc[rv]
        rlines = linestrings(np.stack([rg[["x1", "y1"]].values,
                                       rg[["x2", "y2"]].values], axis=1))
        _, d_route = STRtree(rlines).query_nearest(pts, return_distance=True,
                                                   all_matches=False)
        out.update({
            "gps_to_route_median_m": float(np.median(d_route)),
            "gps_to_route_p95_m": float(np.percentile(d_route, 95)),
            "pct_gps_gt_30m_from_route": float((d_route > 30).mean() * 100),
        })
    return out, d_any, d_route


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_plots(df, dt_all, d_net_all, d_route_all, tracks, plot_dir, example):
    plot_dir.mkdir(parents=True, exist_ok=True)
    parts = df["partition"].unique()

    fig, ax = plt.subplots(figsize=(7, 4))
    for p in parts:
        ax.hist(df.loc[df.partition == p, "n_points"], bins=20, alpha=0.6, label=p)
    ax.set(xlabel="GPS points per trip", ylabel="Trips",
           title="Trajectory length (points)")
    ax.legend()
    save(fig, plot_dir / "01_points_per_trip.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    vc = pd.Series(dt_all["train"]).round(0).clip(upper=30).value_counts().sort_index()
    ax.bar(vc.index, vc.values)
    ax.set_yscale("log")
    ax.set(xlabel="Seconds between consecutive fixes (30 = 30 or more)",
           ylabel="Count (log)", title="Sampling interval distribution - train")
    save(fig, plot_dir / "02_sampling_intervals_train.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    d = df.sort_values("max_gap_s", ascending=False)
    ax.bar(d["trip_id"], d["max_gap_s"],
           color=["#1f77b4" if p == "train" else "#ff7f0e" for p in d["partition"]])
    ax.set_yscale("log")
    ax.set(xlabel="Trip (blue = train, orange = student_test)",
           ylabel="Largest gap (s, log)", title="Largest GPS gap per trip")
    ax.set_xticks([])
    save(fig, plot_dir / "03_max_gap_per_trip.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 60, 61)
    ax.hist(np.clip(d_net_all["train"], 0, 60), bins=bins, alpha=0.6,
            label="nearest arc in network")
    if len(d_route_all["train"]):
        ax.hist(np.clip(d_route_all["train"], 0, 60), bins=bins, alpha=0.6,
                label="nearest arc on reference route")
    ax.set(xlabel="Distance from GPS fix (m, 60 = 60 or more)", ylabel="GPS fixes",
           title="GPS noise: distance to road - train")
    ax.legend()
    save(fig, plot_dir / "04_gps_distance_to_road_train.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    for (p, tid), tr in tracks.items():
        ax.plot(tr["lon"], tr["lat"], lw=0.7,
                color="#1f77b4" if p == "train" else "#ff7f0e", alpha=0.7)
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title="All trajectories (blue = train, orange = student_test)")
    ax.set_aspect(1 / np.cos(np.radians(np.mean(df[["lat_min", "lat_max"]].values))))
    save(fig, plot_dir / "05_all_trajectories.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for p in parts:
        s = df[df.partition == p]
        axes[0].scatter(s["n_nodes"], s["n_arcs"], label=p, s=18)
        axes[1].scatter(s["n_nodes"], s["n_weak_components"], label=p, s=18)
    axes[0].set(xscale="log", yscale="log", xlabel="Nodes", ylabel="Arcs",
                title="Road-network size per trip")
    axes[1].set(xscale="log", xlabel="Nodes", ylabel="Weakly connected components",
                title="Road-network connectivity per trip")
    axes[0].legend()
    save(fig, plot_dir / "06_network_size_connectivity.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    cols = ["n_nan_lon", "n_nan_lat", "n_nan_t", "n_out_of_range",
            "n_duplicate_t", "n_non_increasing_t", "n_speed_jumps"]
    tot = df.groupby("partition")[cols].sum().T
    tot.plot.barh(ax=ax)
    ax.set(xlabel="Count across all trips", title="Missing / invalid GPS values")
    save(fig, plot_dir / "07_missing_invalid_values.png")

    # one trip on its road network
    trip, geom = example
    fig, ax = plt.subplots(figsize=(7, 7))
    g = geom[geom.valid]
    for x1, y1, x2, y2 in g[["x1", "y1", "x2", "y2"]].values:
        ax.plot([x1, x2], [y1, y2], color="#bbbbbb", lw=0.5)
    if trip.has_route:
        r = g.loc[[a for a in trip.route if a in g.index]]
        for x1, y1, x2, y2 in r[["x1", "y1", "x2", "y2"]].values:
            ax.plot([x1, x2], [y1, y2], color="#2ca02c", lw=3, zorder=4)
    ax.scatter(trip.track["x"], trip.track["y"], s=1, color="#1f77b4", alpha=0.5, zorder=3)
    ax.set(xlabel="Easting (m)", ylabel="Northing (m)", aspect="equal",
           title=f"Trip {trip.trip_id}: network (grey), route (green), GPS (blue)")
    pad = 300
    ax.set_xlim(trip.track.x.min() - pad, trip.track.x.max() + pad)
    ax.set_ylim(trip.track.y.min() - pad, trip.track.y.max() + pad)
    save(fig, plot_dir / f"08_trip_{trip.trip_id}_on_network.png")


# --------------------------------------------------------------------------
# Data-quality summary
# --------------------------------------------------------------------------
def quality_notes(df) -> str:
    lines = ["# Data-quality findings (auto-generated counts)", "",
             "Partitions read: " + ", ".join(sorted(df.partition.unique())), ""]

    def flag(mask, text):
        hits = df.loc[mask, ["partition", "trip_id"]]
        if len(hits):
            ids = ", ".join(f"{p}/{t}" for p, t in hits.values[:15])
            more = " ..." if len(hits) > 15 else ""
            lines.append(f"- {text}: {len(hits)} trips ({ids}{more})")
        else:
            lines.append(f"- {text}: none")

    flag(df.n_nan_lon + df.n_nan_lat + df.n_nan_t > 0, "Trips with NaN values")
    flag(df.n_out_of_range > 0, "Trips with out-of-range coordinates")
    flag(df.n_non_increasing_t > 0, "Trips with non-increasing timestamps")
    flag(df.n_gaps_gt_30s > 0, "Trips with a GPS gap over 30 s")
    flag(df.n_speed_jumps > 0, f"Trips with jumps over {SPEED_JUMP_MPS*3.6:.0f} km/h")
    flag(df.n_invalid_arc_refs > 0, "Trips whose arcs reference missing nodes")
    flag(df.n_weak_components > 1, "Trips whose network has >1 weak component")
    flag(df.n_isolated_nodes > 0, "Trips with isolated nodes")
    if "route_n_breaks" in df:
        flag(df.route_n_breaks.fillna(0) > 0, "Routes with disconnected consecutive arcs")
        flag(df.route_n_invalid_ids.fillna(0) > 0, "Routes with out-of-range arc ids")
    if "pct_gps_gt_30m_from_route" in df:
        flag(df.pct_gps_gt_30m_from_route.fillna(0) > 10,
             "Trips with >10% of GPS fixes over 30 m from the reference route")
    lines += ["", "Overall (median across trips):", ""]
    for c in ["n_points", "duration_s", "pct_dt_exactly_1s", "max_gap_s",
              "path_km", "median_speed_kmh", "n_nodes", "n_arcs",
              "pct_arcs_bidirectional", "gps_to_net_median_m",
              "gps_to_route_median_m"]:
        if c in df:
            lines.append(f"- {c}: {df[c].median():.2f}")
    lines += ["", "Add the team's own interpretation and decisions below.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Folder containing train/ and student_test/")
    ap.add_argument("--out", default="outputs/eda")
    ap.add_argument("--map-trips", nargs="*", default=None,
                    help="partition/trip_id to map, e.g. train/00000005")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    out = Path(args.out)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    print(f"Data root: {root}")

    inv = pd.concat([file_inventory(root, p) for p in PARTITIONS], ignore_index=True)
    inv.to_csv(out / "inventory.csv", index=False)
    print(inv.groupby("partition").agg(
        trips=("trip_id", "count"), with_route=("has_route", "sum"),
        gps_rows=("track_rows", "sum"), node_rows=("nodes_rows", "sum"),
        arc_rows=("arcs_rows", "sum")).to_string())

    rows, tracks = [], {}
    dt_all = {p: [] for p in PARTITIONS}
    d_net_all = {p: [] for p in PARTITIONS}
    d_route_all = {p: [] for p in PARTITIONS}
    example = None
    t0 = time.time()
    for p in PARTITIONS:
        for tid in list_trips(root, p):
            trip = load_trip(root, p, tid)
            epsg = add_xy(trip.track)
            add_xy(trip.nodes, epsg)
            kin = track_kinematics(trip.track)
            geom = arc_geometry(trip.nodes, trip.arcs)

            row = {"partition": p, "trip_id": tid, "epsg": epsg}
            row.update(track_stats(trip, kin))
            row.update(network_stats(trip, geom))
            row.update(route_stats(trip, geom))
            gstats, d_any, d_route = gps_to_network(trip, geom)
            row.update(gstats)
            rows.append(row)

            dt_all[p].extend(kin["dt"].dropna().tolist())
            d_net_all[p].extend(d_any.tolist())
            d_route_all[p].extend(d_route.tolist())
            tracks[(p, tid)] = trip.track[["lon", "lat"]]
            if example is None and p == "train":
                example = (trip, geom)
            print(f"  {p}/{tid}: {row['n_points']} pts, {row['n_arcs']} arcs")
    print(f"Processed {len(rows)} trips in {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "trip_stats.csv", index=False)
    df.groupby("partition").describe().T.to_csv(out / "summary_by_partition.csv")
    miss = df.groupby("partition")[["n_nan_lon", "n_nan_lat", "n_nan_t",
                                     "n_out_of_range", "n_duplicate_t",
                                     "n_non_increasing_t", "n_duplicate_positions",
                                     "n_speed_jumps"]].sum()
    miss.to_csv(out / "missing_invalid_values.csv")
    (out / "data_quality_notes.md").write_text(quality_notes(df), encoding="utf-8")

    make_plots(df, dt_all, d_net_all, d_route_all, tracks, out / "plots", example)

    targets = args.map_trips
    if targets is None:  # default: first train trip and first student-test trip
        targets = [f"{p}/{list_trips(root, p)[0]}" for p in PARTITIONS]
    for t in targets:
        p, tid = t.replace("\\", "/").split("/")
        path = make_trip_map(load_trip(root, p, tid), out / "maps" / f"{p}_{tid}.html")
        print(f"Map written: {path}")
    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
