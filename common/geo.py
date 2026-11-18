"""Shared geometry helpers.

Raw coordinates are lon/lat in degrees (WGS84). Degrees are not metres, so
everything is projected to UTM before measuring distance. The data sits at
about 35.5 E / 49.6 N, which is UTM zone 36N (EPSG:32636). The zone is picked
from the data's mean longitude so this still works if a trip falls elsewhere.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pyproj import Transformer


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=8)
def _transformer(epsg: int) -> Transformer:
    # always_xy=True -> input order is (lon, lat), output is (easting, northing)
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def project_lonlat(lon, lat, epsg: int | None = None):
    """Return (x, y) in metres and the EPSG code used."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if epsg is None:
        epsg = utm_epsg_for(float(np.nanmean(lon)), float(np.nanmean(lat)))
    x, y = _transformer(epsg).transform(lon, lat)
    return np.asarray(x), np.asarray(y), epsg


def add_xy(df: pd.DataFrame, epsg: int | None = None) -> int:
    """Add x, y (metres) columns to a frame that has lon, lat. Returns EPSG."""
    df["x"], df["y"], epsg = project_lonlat(df["lon"].values, df["lat"].values, epsg)
    return epsg


def arc_geometry(nodes_xy: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    """Start/end coordinates, length (m) and bearing (deg, 0 = north,
    clockwise) for every arc. Arcs that reference missing nodes get NaN."""
    valid = arcs["u"].between(0, len(nodes_xy) - 1) & arcs["v"].between(0, len(nodes_xy) - 1)
    out = pd.DataFrame(index=arcs.index)
    out["valid"] = valid
    u = arcs["u"].where(valid, 0).values
    v = arcs["v"].where(valid, 0).values
    out["x1"], out["y1"] = nodes_xy["x"].values[u], nodes_xy["y"].values[u]
    out["x2"], out["y2"] = nodes_xy["x"].values[v], nodes_xy["y"].values[v]
    out.loc[~valid, ["x1", "y1", "x2", "y2"]] = np.nan
    dx, dy = out["x2"] - out["x1"], out["y2"] - out["y1"]
    out["length_m"] = np.hypot(dx, dy)
    out["bearing_deg"] = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    return out


def track_kinematics(track_xy: pd.DataFrame) -> pd.DataFrame:
    """Per-point step distance (m), time step (s), speed (m/s) and heading
    (deg) computed from the previous point. First row is NaN."""
    out = pd.DataFrame(index=track_xy.index)
    dx = track_xy["x"].diff()
    dy = track_xy["y"].diff()
    out["dt"] = track_xy["t"].diff()
    out["step_m"] = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["speed_mps"] = out["step_m"] / out["dt"]
    out["heading_deg"] = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    return out


def point_to_segments(px, py, x1, y1, x2, y2):
    """Distance (m) from each point to each segment, plus the projection
    fraction along the segment (0 = start, 1 = end).

    px, py: shape (P,)     segments: shape (S,)
    returns dist (P, S), frac (P, S)
    """
    px = np.asarray(px)[:, None]
    py = np.asarray(py)[:, None]
    sx, sy = (x2 - x1)[None, :], (y2 - y1)[None, :]
    seg_len2 = sx ** 2 + sy ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = ((px - x1[None, :]) * sx + (py - y1[None, :]) * sy) / seg_len2
    frac = np.where(seg_len2 > 0, np.clip(frac, 0.0, 1.0), 0.0)
    cx = x1[None, :] + frac * sx
    cy = y1[None, :] + frac * sy
    return np.hypot(px - cx, py - cy), frac


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)
    return d
