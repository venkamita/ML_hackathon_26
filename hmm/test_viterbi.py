"""
Unit and benchmark test for HMM/Viterbi Route Decoder.
"""
import time
from common.data_loader import load_trip
from common.geo import add_xy, arc_geometry
from hmm.viterbi import RouteDecoder

def main():
    print("Testing HMM/Viterbi route decoder on trip 00000001...")
    t0 = time.time()
    trip = load_trip(".", "train", "00000001")
    epsg = add_xy(trip.track)
    add_xy(trip.nodes, epsg)
    arcs_geom = arc_geometry(trip.nodes, trip.arcs)
    t_load = time.time()
    print(f"Data load & projection: {t_load - t0:.2f} s")

    decoder = RouteDecoder(radius_m=50.0, max_k=8)
    t_start_decode = time.time()
    res = decoder.decode(trip.track, trip.arcs, arcs_geom)
    t_decode = time.time() - t_start_decode
    print(f"Decoding {res['num_points']} GPS points took: {t_decode:.2f} s ({t_decode*1000/res['num_points']:.2f} ms/point)")

    greedy_cont = decoder.compute_route_continuity(res['greedy_route'], trip.arcs)
    viterbi_cont = decoder.compute_route_continuity(res['predicted_route'], trip.arcs)

    print(f"Greedy Top-1 Continuity:  {greedy_cont * 100:.2f}%")
    print(f"HMM/Viterbi Continuity:   {viterbi_cont * 100:.2f}%")

    if trip.has_route:
        # Compute reference route overlap
        ref_arcs_set = set(trip.route.values)
        greedy_in_route = sum(1 for a in res['greedy_route'] if a in ref_arcs_set) / len(res['greedy_route'])
        viterbi_in_route = sum(1 for a in res['predicted_route'] if a in ref_arcs_set) / len(res['predicted_route'])
        print(f"Greedy in Reference Route:  {greedy_in_route * 100:.2f}%")
        print(f"Viterbi in Reference Route: {viterbi_in_route * 100:.2f}%")

if __name__ == "__main__":
    main()
