"""Shared data loading for PathSnap.

Every team member should load trips through these functions so the whole
pipeline reads the raw files in exactly one way.

Raw layout (per trip folder <id>/):
    <id>.track  lon  lat  t_seconds      (whitespace separated, no header)
    <id>.nodes  lon  lat                 (row index = node_id)
    <id>.arcs   from_node  to_node       (row index = arc_id)
    <id>.route  arc_id                   (chronological ground-truth arcs)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Partitions we are allowed to touch during development.
PARTITIONS = ("train", "student_test")


@dataclass
class Trip:
    trip_id: str
    partition: str
    folder: Path
    track: pd.DataFrame            # columns: lon, lat, t
    nodes: pd.DataFrame            # columns: lon, lat   (index = node_id)
    arcs: pd.DataFrame             # columns: u, v       (index = arc_id)
    route: Optional[pd.Series]     # arc_ids in order, or None if no .route file

    @property
    def has_route(self) -> bool:
        return self.route is not None


def resolve_data_root(path: str | Path) -> Path:
    """Accept either the folder that contains train/ and student_test/,
    or its parent (the download unzips as 'Student folder/Student folder')."""
    p = Path(path).expanduser().resolve()
    if (p / "train").is_dir():
        return p
    for child in p.iterdir():
        if child.is_dir() and (child / "train").is_dir():
            return child
    raise FileNotFoundError(f"Could not find a 'train' folder under {p}")


def list_trips(data_root: str | Path, partition: str) -> list[str]:
    root = resolve_data_root(data_root)
    part_dir = root / partition
    if not part_dir.is_dir():
        raise FileNotFoundError(part_dir)
    return sorted(d.name for d in part_dir.iterdir() if d.is_dir())


def _read_table(path: Path, names: list[str], dtypes: dict) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names,
                       dtype=dtypes, engine="c")


def load_trip(data_root: str | Path, partition: str, trip_id: str) -> Trip:
    root = resolve_data_root(data_root)
    folder = root / partition / trip_id
    f = lambda ext: folder / f"{trip_id}.{ext}"

    track = _read_table(f("track"), ["lon", "lat", "t"],
                        {"lon": float, "lat": float, "t": float})
    nodes = _read_table(f("nodes"), ["lon", "lat"], {"lon": float, "lat": float})
    arcs = _read_table(f("arcs"), ["u", "v"], {"u": "int64", "v": "int64"})
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"

    route = None
    if f("route").exists():
        route = _read_table(f("route"), ["arc_id"], {"arc_id": "int64"})["arc_id"]

    return Trip(trip_id, partition, folder, track, nodes, arcs, route)


def file_inventory(data_root: str | Path, partition: str) -> pd.DataFrame:
    """One row per trip: which files exist, their sizes and row counts."""
    root = resolve_data_root(data_root)
    rows = []
    for tid in list_trips(root, partition):
        row = {"partition": partition, "trip_id": tid}
        for ext in ("track", "nodes", "arcs", "route"):
            p = root / partition / tid / f"{tid}.{ext}"
            row[f"has_{ext}"] = p.exists()
            row[f"{ext}_bytes"] = p.stat().st_size if p.exists() else 0
            if p.exists():
                with open(p, "rb") as fh:
                    row[f"{ext}_rows"] = sum(1 for line in fh if line.strip())
            else:
                row[f"{ext}_rows"] = 0
        rows.append(row)
    return pd.DataFrame(rows)
