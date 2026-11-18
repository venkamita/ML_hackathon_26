"""Folium map of one trip: road network, GPS points, reference route and
(later) a predicted route, each on its own toggleable layer."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import folium
import numpy as np

from common.data_loader import Trip
from common.geo import add_xy, arc_geometry, project_lonlat

COLORS = {"network": "#8c8c8c", "gps": "#1f77b4", "route": "#2ca02c",
          "pred": "#d62728"}


def _arc_lines(trip: Trip, arc_ids) -> list[list[tuple[float, float]]]:
    arcs = trip.arcs.loc[list(arc_ids)]
    n = len(trip.nodes)
    ok = arcs["u"].between(0, n - 1) & arcs["v"].between(0, n - 1)
    arcs = arcs[ok]
    lat, lon = trip.nodes["lat"].values, trip.nodes["lon"].values
    return [[(lat[u], lon[u]), (lat[v], lon[v])] for u, v in zip(arcs["u"], arcs["v"])]


def _network_arcs_near_track(trip: Trip, buffer_m: float, max_arcs: int):
    """Only draw arcs whose end-point is within the track's bounding box plus
    a buffer, so big networks don't produce a 50 MB HTML file."""
    tx, ty, epsg = project_lonlat(trip.track["lon"], trip.track["lat"])
    nodes = trip.nodes.copy()
    add_xy(nodes, epsg)
    geom = arc_geometry(nodes, trip.arcs)
    xmin, xmax = np.nanmin(tx) - buffer_m, np.nanmax(tx) + buffer_m
    ymin, ymax = np.nanmin(ty) - buffer_m, np.nanmax(ty) + buffer_m
    inside = geom["valid"] & (
        (geom["x1"].between(xmin, xmax) & geom["y1"].between(ymin, ymax)) |
        (geom["x2"].between(xmin, xmax) & geom["y2"].between(ymin, ymax)))
    ids = geom.index[inside]
    return ids[:max_arcs], len(ids) > max_arcs


def _legend_html(items: Sequence[tuple[str, str]]) -> str:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;'
        f'background:{c};margin-right:6px;vertical-align:middle"></span>{name}</div>'
        for name, c in items)
    return ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
            'background:white;padding:8px 10px;border:1px solid #999;'
            'border-radius:4px;font:12px sans-serif">' + rows + '</div>')


def make_trip_map(trip: Trip, out_html: str | Path,
                  predicted_arcs: Optional[Sequence[int]] = None,
                  greedy_arcs: Optional[Sequence[int]] = None,
                  gps_every: Optional[int] = None, buffer_m: float = 300,
                  max_network_arcs: int = 20000) -> Path:
    tr = trip.track.dropna(subset=["lon", "lat"])
    if gps_every is None:  # keep the HTML small: at most ~2000 markers
        gps_every = max(1, len(tr) // 2000)
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)

    # Road network
    ids, capped = _network_arcs_near_track(trip, buffer_m, max_network_arcs)
    net = folium.FeatureGroup(name=f"Road network ({len(ids)} arcs"
                                   f"{', capped' if capped else ''})")
    folium.PolyLine(_arc_lines(trip, ids), color=COLORS["network"],
                    weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    # Reference route
    legend = [("Road network", COLORS["network"]), ("GPS points", COLORS["gps"])]
    if trip.has_route:
        rt = folium.FeatureGroup(name="Reference route (Ground Truth)")
        valid = [a for a in trip.route if 0 <= a < len(trip.arcs)]
        folium.PolyLine(_arc_lines(trip, valid), color=COLORS["route"],
                        weight=5, opacity=0.8).add_to(rt)
        rt.add_to(m)
        legend.append(("Reference route", COLORS["route"]))

    # Greedy route (raw classifier emission without HMM)
    if greedy_arcs is not None:
        gr = folium.FeatureGroup(name="Greedy route (No HMM)", show=False)
        valid = [a for a in greedy_arcs if 0 <= a < len(trip.arcs)]
        folium.PolyLine(_arc_lines(trip, valid), color="#ff7f0e",
                        weight=2, opacity=0.7, dash_array="3 4").add_to(gr)
        gr.add_to(m)
        legend.append(("Greedy route (No HMM)", "#ff7f0e"))

    # Predicted route (Viterbi sequence decoding)
    if predicted_arcs is not None:
        pr = folium.FeatureGroup(name="Predicted route (Viterbi HMM)")
        valid = [a for a in predicted_arcs if 0 <= a < len(trip.arcs)]
        folium.PolyLine(_arc_lines(trip, valid), color=COLORS["pred"],
                        weight=4, opacity=0.9, dash_array="6 4").add_to(pr)
        pr.add_to(m)
        legend.append(("Predicted route (HMM)", COLORS["pred"]))

    # GPS observations
    gps = folium.FeatureGroup(name=f"GPS points (every {gps_every})")
    for _, r in tr.iloc[::gps_every].iterrows():
        folium.CircleMarker((r.lat, r.lon), radius=2, color=COLORS["gps"],
                            fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"t={r.t:.0f}s").add_to(gps)
    gps.add_to(m)

    first, last = tr.iloc[0], tr.iloc[-1]
    folium.Marker((first.lat, first.lon), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((last.lat, last.lon), tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html(legend)))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return out_html
