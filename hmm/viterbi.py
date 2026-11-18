"""
PathSnap HMM / Viterbi route reconstruction (Person D).

The decoder works on Person B's candidate table, the same rows the models
are trained and evaluated on:

    features.build_candidates.build_trip_candidates(trip)  ->  one row per
    (GPS point, candidate arc), columns point_idx, arc_id, dist_m, t, ...

Hidden states  = the candidate arcs of each GPS point.
Emissions      = log P(arc | point) from a model, normalised per point.
                 Without a model, the classic Gaussian GPS-noise emission
                 exp(-0.5 (d / sigma)^2), also normalised per point, is used
                 (Newson & Krumm 2009), so both cases are on the same scale.
Transitions    = road-network topology between consecutive points:
                 same arc, direct successor, 2-hop successor, else a penalty.
Segments       = a trip is split where the GPS gap exceeds gap_s (180 s);
                 each segment is decoded independently.

Nothing here reads the reference route, so it runs on evaluator trips too.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SIGMA_Z_M = 6.68          # GPS noise, estimated on train by data_processing.clean
GAP_SPLIT_S = 180.0       # Newson & Krumm; same rule as the cleaning step
PROB_FLOOR = 1e-6


class RouteDecoder:
    def __init__(
        self,
        sigma_z: float = SIGMA_Z_M,
        gap_s: float = GAP_SPLIT_S,
        self_trans_prob: float = 0.65,
        direct_trans_prob: float = 0.30,
        two_hop_trans_prob: float = 0.04,
        disconnected_penalty: float = -12.0,
        emission_weight: float = 1.0,
        radius_m: float = 50.0,
        max_k: int = 8,
    ):
        """
        sigma_z: GPS noise (m), only used when no model probabilities are given.
        gap_s: a time gap larger than this starts a new, independent segment.
        self_trans_prob: staying on the same arc between consecutive points.
        direct_trans_prob: moving to an arc that starts where this one ends.
        two_hop_trans_prob: moving two arcs ahead (short arcs, fast driving).
        disconnected_penalty: log-score for any other move (keeps Viterbi
            from dead-ending where the network is broken into pieces).
        emission_weight: multiplies the log-emission. A sharp, confident model
            (one that already uses neighbouring points as features) can
            overpower the transitions; w < 1 softens it. Tuned on TRAIN trips
            only (hmm/tune_emission_weight.py).
        radius_m, max_k: candidate settings for decode_trip; keep them equal
            to Person B's (outputs/features/feature_config.json).
        """
        self.sigma_z = sigma_z
        self.gap_s = gap_s
        self.log_self = math.log(self_trans_prob)
        self.log_direct = math.log(direct_trans_prob)
        self.log_two_hop = math.log(two_hop_trans_prob)
        self.disconnected_penalty = disconnected_penalty
        self.emission_weight = emission_weight
        self.radius_m = radius_m
        self.max_k = max_k

    # ------------------------------------------------------------------ graph
    @staticmethod
    def build_network_graph(arcs_df: pd.DataFrame):
        """successors[a] = set of arcs starting at a's end node;
        endpoints[a] = (u, v)."""
        u = arcs_df["u"].to_numpy()
        v = arcs_df["v"].to_numpy()
        ids = arcs_df.index.to_numpy()
        out_by_node: Dict[int, List[int]] = defaultdict(list)
        for a, uu in zip(ids, u):
            out_by_node[int(uu)].append(int(a))
        successors = {int(a): set(out_by_node.get(int(vv), ())) for a, vv in zip(ids, v)}
        endpoints = {int(a): (int(uu), int(vv)) for a, uu, vv in zip(ids, u, v)}
        return successors, endpoints

    def log_transition(self, a: int, b: int, successors, endpoints) -> float:
        # duplicate arcs (same from/to node, different id) count as "the same arc"
        if a == b or endpoints.get(a) == endpoints.get(b):
            return self.log_self
        succ = successors.get(a, ())
        if b in succ:
            return self.log_direct
        for s in succ:
            if b in successors.get(s, ()):
                return self.log_two_hop
        return self.disconnected_penalty

    # -------------------------------------------------------------- emissions
    def gaussian_probs(self, cands: pd.DataFrame) -> np.ndarray:
        """Distance-only emission, normalised per point (no model needed)."""
        s = np.exp(-0.5 * (cands["dist_m"].to_numpy(float) / self.sigma_z) ** 2)
        tot = pd.Series(s).groupby(cands["point_idx"].to_numpy()).transform("sum").to_numpy()
        return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1.0)

    # ---------------------------------------------------------------- decode
    def decode_candidates(
        self,
        cands: pd.DataFrame,
        arcs_df: pd.DataFrame,
        probs: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        cands: Person B's candidate rows for ONE trip (point_idx, arc_id,
               dist_m, t, cand_rank). Row order does not matter.
        probs: per-row model probabilities aligned with `cands`, each point's
               candidates summing to 1. None -> Gaussian distance emission.
        Returns, per GPS point in time order: the Viterbi arc, the greedy
        (highest-probability) arc, and the segment id.
        """
        df = cands[["point_idx", "arc_id", "t", "cand_rank"]].copy()
        df["p"] = self.gaussian_probs(cands) if probs is None else np.asarray(probs, float)
        df["row"] = np.arange(len(df))
        df = df.sort_values(["point_idx", "cand_rank"], kind="mergesort")

        point_ids = df["point_idx"].to_numpy()
        starts = np.r_[0, np.flatnonzero(point_ids[1:] != point_ids[:-1]) + 1]
        ends = np.r_[starts[1:], len(df)]
        arcs = df["arc_id"].to_numpy().astype(int)
        logp = self.emission_weight * np.log(np.maximum(df["p"].to_numpy(), PROB_FLOOR))
        p = df["p"].to_numpy()
        t_pt = df["t"].to_numpy()[starts]
        pts = point_ids[starts]

        successors, endpoints = self.build_network_graph(arcs_df)
        n = len(starts)
        seg = np.zeros(n, dtype=int)
        if n > 1:
            seg[1:] = np.cumsum(np.diff(t_pt) > self.gap_s)

        greedy = np.array([arcs[s + int(np.argmax(p[s:e]))] for s, e in zip(starts, ends)])
        viterbi = np.empty(n, dtype=int)
        trans_cache: Dict[tuple, float] = {}

        def trans(a, b):
            key = (a, b)
            v = trans_cache.get(key)
            if v is None:
                v = self.log_transition(a, b, successors, endpoints)
                trans_cache[key] = v
            return v

        for sid in np.unique(seg):
            idx = np.flatnonzero(seg == sid)
            score = logp[starts[idx[0]]:ends[idx[0]]].copy()
            back: List[np.ndarray] = []
            for k in range(1, len(idx)):
                ps, pe = starts[idx[k - 1]], ends[idx[k - 1]]
                cs, ce = starts[idx[k]], ends[idx[k]]
                prev_arcs, cur_arcs = arcs[ps:pe], arcs[cs:ce]
                T = np.array([[trans(int(a), int(b)) for b in cur_arcs] for a in prev_arcs])
                tot = score[:, None] + T
                bp = np.argmax(tot, axis=0)
                score = tot[bp, np.arange(len(cur_arcs))] + logp[cs:ce]
                back.append(bp)
            j = int(np.argmax(score))
            for k in range(len(idx) - 1, -1, -1):
                viterbi[idx[k]] = arcs[starts[idx[k]] + j]
                if k > 0:
                    j = int(back[k - 1][j])

        return {
            "point_idx": pts,
            "segment_id": seg,
            "predicted_route": viterbi.tolist(),
            "greedy_route": greedy.tolist(),
            "num_points": int(n),
            "num_segments": int(seg.max() + 1) if n else 0,
        }

    def decode_trip(
        self,
        trip,
        score_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        k: Optional[int] = None,
        radius_m: Optional[float] = None,
    ) -> Dict:
        """Full pipeline for one trip (also evaluator trips with no route):
        B's candidates + features -> model probabilities -> Viterbi."""
        from features.build_candidates import build_trip_candidates
        cands, tr, geom = build_trip_candidates(trip, k or self.max_k, radius_m or self.radius_m)
        cands = cands.sort_values(["point_idx", "cand_rank"], kind="mergesort").reset_index(drop=True)
        probs = None if score_fn is None else score_fn(cands)
        res = self.decode_candidates(cands, trip.arcs, probs)
        res.update({"candidates": cands, "probs": probs, "track_xy": tr, "geom": geom})
        return res

    # --------------------------------------------------------------- metrics
    def compute_route_continuity(
        self, route_arcs: Sequence[int], arcs_df: pd.DataFrame,
        segment_id: Optional[Sequence[int]] = None,
    ) -> float:
        """% of consecutive points whose arcs are the same (or a duplicate)
        or directly connected. Steps across a gap split are not counted."""
        if len(route_arcs) <= 1:
            return 1.0
        successors, endpoints = self.build_network_graph(arcs_df)
        ok = tot = 0
        for i in range(len(route_arcs) - 1):
            if segment_id is not None and segment_id[i] != segment_id[i + 1]:
                continue
            a, b = int(route_arcs[i]), int(route_arcs[i + 1])
            tot += 1
            if a == b or endpoints.get(a) == endpoints.get(b) or b in successors.get(a, ()):
                ok += 1
        return ok / tot if tot else 1.0
