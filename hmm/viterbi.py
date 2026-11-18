"""
PathSnap HMM and Viterbi Route Reconstruction Engine.
Person D component: Implements state-space candidate generation,
emission and transition modeling, and dynamic programming Viterbi decoding.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any

from common.geo import add_xy, arc_geometry


class RouteDecoder:
    def __init__(
        self,
        radius_m: float = 60.0,
        max_k: int = 8,
        sigma_z: float = 4.07,
        beta: float = 5.0,
        self_trans_prob: float = 0.65,
        direct_trans_prob: float = 0.30,
        disconnected_penalty: float = -12.0
    ):
        """
        HMM parameters:
          radius_m: Spatial candidate search radius in meters.
          max_k: Max candidate road segments to keep per GPS point.
          sigma_z: Gaussian GPS measurement noise standard deviation in meters (from EDA).
          beta: Transition difference penalty parameter (Newson & Krumm 2009).
          self_trans_prob: Probability of remaining on the same segment between consecutive seconds.
          direct_trans_prob: Probability of directly transitioning to a connected successor segment.
          disconnected_penalty: Log-probability penalty for topologically disconnected hops.
        """
        self.radius_m = radius_m
        self.max_k = max_k
        self.sigma_z = sigma_z
        self.beta = beta
        self.self_trans_prob = self_trans_prob
        self.direct_trans_prob = direct_trans_prob
        self.disconnected_penalty = disconnected_penalty

    def build_network_graph(self, arcs_df: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, Tuple[int, int]]]:
        """
        Constructs adjacency map where arc_id -> list of successor arc_ids.
        Arc i: u_i -> v_i. Arc j is a direct successor if u_j == v_i.
        """
        node_outgoing_arcs = defaultdict(list)
        arc_endpoints = {}

        for arc_id, row in arcs_df.iterrows():
            u, v = int(row['u']), int(row['v'])
            arc_endpoints[arc_id] = (u, v)
            node_outgoing_arcs[u].append(arc_id)

        # Adjacency: arc_id -> list of successor arcs
        arc_successors = defaultdict(list)
        for arc_id, (u, v) in arc_endpoints.items():
            successors = node_outgoing_arcs.get(v, [])
            arc_successors[arc_id] = successors

        return arc_successors, arc_endpoints

    def build_spatial_index(self, arcs_geom: pd.DataFrame) -> Tuple[cKDTree, np.ndarray, np.ndarray]:
        """
        Builds KDTree on segment midpoints and precomputes segment coordinates.
        """
        valid_mask = arcs_geom['valid'].values
        valid_indices = arcs_geom.index[valid_mask].values

        x1 = arcs_geom.loc[valid_mask, 'x1'].values
        y1 = arcs_geom.loc[valid_mask, 'y1'].values
        x2 = arcs_geom.loc[valid_mask, 'x2'].values
        y2 = arcs_geom.loc[valid_mask, 'y2'].values

        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        midpoints = np.column_stack([mid_x, mid_y])

        kdtree = cKDTree(midpoints)
        coords = np.column_stack([x1, y1, x2, y2])
        return kdtree, valid_indices, coords

    def point_segment_distance(
        self, px: float, py: float, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized orthogonal distance from point (px, py) to segments.
        Returns:
            distances: orthogonal distance in meters
            t_clamped: projection fraction [0.0, 1.0] along segment
        """
        dx = x2 - x1
        dy = y2 - y1
        l2 = dx * dx + dy * dy
        l2 = np.where(l2 == 0, 1e-6, l2)

        t = ((px - x1) * dx + (py - y1) * dy) / l2
        t_clamped = np.clip(t, 0.0, 1.0)

        proj_x = x1 + t_clamped * dx
        proj_y = y1 + t_clamped * dy
        dist = np.hypot(px - proj_x, py - proj_y)
        return dist, t_clamped

    def find_candidates(
        self,
        px: float,
        py: float,
        kdtree: cKDTree,
        valid_arc_ids: np.ndarray,
        seg_coords: np.ndarray,
        radius_m: float
    ) -> List[Tuple[int, float]]:
        """
        Finds candidate road segments within search radius, sorted by orthogonal distance.
        Returns list of (arc_id, distance_m).
        """
        # Query KDTree within buffer (radius + segment length margin)
        query_radius = radius_m + 50.0
        idx_list = kdtree.query_ball_point([px, py], r=query_radius)

        if not idx_list:
            # Fallback to nearest midpoint if no segments within radius
            _, nearest_idx = kdtree.query([px, py], k=1)
            idx_list = [nearest_idx]

        c_indices = np.array(idx_list)
        sub_coords = seg_coords[c_indices]
        dists, _ = self.point_segment_distance(
            px, py, sub_coords[:, 0], sub_coords[:, 1], sub_coords[:, 2], sub_coords[:, 3]
        )

        within_radius = dists <= radius_m
        if not np.any(within_radius):
            # Keep top 2 nearest if none strictly within radius
            top_k = min(len(dists), 2)
            sort_order = np.argsort(dists)[:top_k]
        else:
            filtered_indices = np.where(within_radius)[0]
            sort_order = filtered_indices[np.argsort(dists[filtered_indices])[:self.max_k]]

        candidates = [
            (int(valid_arc_ids[c_indices[i]]), float(dists[i])) for i in sort_order
        ]
        return candidates

    def compute_log_emission(self, dist_m: float, classifier_prob: Optional[float] = None) -> float:
        """
        Calculates log emission probability.
        If classifier probability is available, uses log(p).
        Otherwise uses Gaussian measurement model: N(0, sigma_z^2).
        """
        if classifier_prob is not None and classifier_prob > 0:
            return math.log(max(classifier_prob, 1e-6))

        # Gaussian measurement model
        var = self.sigma_z ** 2
        log_prob = -0.5 * math.log(2.0 * math.pi * var) - (dist_m ** 2) / (2.0 * var)
        return log_prob

    def compute_log_transition(
        self,
        from_arc: int,
        to_arc: int,
        arc_successors: Dict[int, List[int]],
        gps_dist_m: float
    ) -> float:
        """
        Calculates log transition probability P(to_arc | from_arc).
        """
        if from_arc == to_arc:
            return math.log(self.self_trans_prob)

        successors = arc_successors.get(from_arc, [])
        if to_arc in successors:
            # Direct successor transition
            log_p = math.log(self.direct_trans_prob)
            return log_p

        # Check 2-hop reachability
        for succ in successors:
            if to_arc in arc_successors.get(succ, []):
                return math.log(0.04)

        # Disconnected hop penalty
        return self.disconnected_penalty

    def decode(
        self,
        trip_track: pd.DataFrame,
        trip_arcs: pd.DataFrame,
        arcs_geom: pd.DataFrame,
        classifier_probs: Optional[Dict[Tuple[int, int], float]] = None
    ) -> Dict[str, Any]:
        """
        Runs full Viterbi sequence decoding on the trip trajectory.
        Returns:
            dict containing predicted_route, greedy_route, candidates, and runtime metrics.
        """
        arc_successors, _ = self.build_network_graph(trip_arcs)
        kdtree, valid_arc_ids, seg_coords = self.build_spatial_index(arcs_geom)

        N = len(trip_track)
        px_arr = trip_track['x'].values
        py_arr = trip_track['y'].values
        t_arr = trip_track['t'].values

        # 1. Generate candidates and emissions for all points
        candidates_by_step = []
        greedy_route = []

        for t in range(N):
            px, py = px_arr[t], py_arr[t]
            cands = self.find_candidates(px, py, kdtree, valid_arc_ids, seg_coords, self.radius_m)
            candidates_by_step.append(cands)
            # Greedy prediction: arc with minimum distance
            greedy_route.append(cands[0][0])

        # 2. Viterbi Dynamic Programming in log-space
        # dp[t][k] = max log-prob ending at candidate k at time step t
        dp: List[np.ndarray] = []
        backpointer: List[np.ndarray] = []

        # Initialization at t = 0
        init_cands = candidates_by_step[0]
        init_scores = np.zeros(len(init_cands))
        for k_idx, (arc_id, dist_m) in enumerate(init_cands):
            c_prob = classifier_probs.get((0, arc_id)) if classifier_probs else None
            init_scores[k_idx] = self.compute_log_emission(dist_m, c_prob)
        dp.append(init_scores)

        # Forward pass
        for t in range(1, N):
            curr_cands = candidates_by_step[t]
            prev_cands = candidates_by_step[t - 1]
            prev_dp = dp[t - 1]

            dt = t_arr[t] - t_arr[t - 1]
            dx = px_arr[t] - px_arr[t - 1]
            dy = py_arr[t] - py_arr[t - 1]
            gps_dist = math.hypot(dx, dy)

            # If GPS gap > 180s, reset sequence
            is_large_gap = (dt > 180.0)

            curr_scores = np.zeros(len(curr_cands))
            curr_bp = np.zeros(len(curr_cands), dtype=int)

            for j, (to_arc, dist_m) in enumerate(curr_cands):
                c_prob = classifier_probs.get((t, to_arc)) if classifier_probs else None
                log_emission = self.compute_log_emission(dist_m, c_prob)

                if is_large_gap:
                    # Reset transition dependency
                    best_prev_idx = 0
                    max_trans_score = prev_dp[0]
                else:
                    trans_scores = np.zeros(len(prev_cands))
                    for i, (from_arc, _) in enumerate(prev_cands):
                        log_trans = self.compute_log_transition(from_arc, to_arc, arc_successors, gps_dist)
                        trans_scores[i] = prev_dp[i] + log_trans

                    best_prev_idx = int(np.argmax(trans_scores))
                    max_trans_score = trans_scores[best_prev_idx]

                curr_scores[j] = max_trans_score + log_emission
                curr_bp[j] = best_prev_idx

            dp.append(curr_scores)
            backpointer.append(curr_bp)

        # 3. Backtracking to reconstruct optimal sequence
        best_last_idx = int(np.argmax(dp[-1]))
        predicted_route = [candidates_by_step[-1][best_last_idx][0]]

        for t in range(N - 1, 0, -1):
            best_prev_idx = backpointer[t - 1][best_last_idx]
            predicted_route.append(candidates_by_step[t - 1][best_prev_idx][0])
            best_last_idx = best_prev_idx

        predicted_route.reverse()

        return {
            "predicted_route": predicted_route,
            "greedy_route": greedy_route,
            "candidates_by_step": candidates_by_step,
            "num_points": N,
        }

    def compute_route_continuity(self, route_arcs: List[int], arcs_df: pd.DataFrame) -> float:
        """
        Computes Route Continuity: % of consecutive transitions (s_t -> s_{t+1})
        that are either identical (vehicle on same road) or directly connected in graph.
        """
        if len(route_arcs) <= 1:
            return 1.0

        arc_successors, _ = self.build_network_graph(arcs_df)
        valid_transitions = 0
        total_transitions = len(route_arcs) - 1

        for i in range(total_transitions):
            u_arc = route_arcs[i]
            v_arc = route_arcs[i + 1]
            if u_arc == v_arc or v_arc in arc_successors.get(u_arc, []):
                valid_transitions += 1

        return float(valid_transitions / total_transitions)
