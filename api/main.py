"""
PathSnap Inference & Evaluation API (Person D).
Runs on Laptop 1 (Model and API Server).
Provides endpoints for health check, listing evaluation trips, running map-matching,
evaluating server-side metrics, and outputting real-time console audit logs.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Data root is the folder containing train/, student_test/, and evaluator_test/
DATA_ROOT = PROJECT_ROOT.parent if (PROJECT_ROOT.parent / "train").is_dir() else PROJECT_ROOT

from common.data_loader import load_trip, list_trips
from common.geo import add_xy, arc_geometry
from hmm.viterbi import RouteDecoder

# Configure server console logging for live demo evaluation audit
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PATHSNAP-API] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pathsnap_api")

# Initialize RouteDecoder
decoder = RouteDecoder(radius_m=50.0, max_k=8, sigma_z=6.68)

# Cache precomputed predictions for instantaneous responses if available
PRECOMPUTED_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def get_available_partitions() -> List[str]:
    """Finds all partitions present in data root."""
    parts = []
    for candidate in ["evaluator_test", "student_test", "train"]:
        if (DATA_ROOT / candidate).is_dir():
            parts.append(candidate)
    return parts


def run_prediction_pipeline(
    trip_id: str,
    partition: Optional[str] = None,
    model_name: str = "conditional_logit"
) -> Dict[str, Any]:
    """
    Executes end-to-end route reconstruction pipeline for a given trip:
    1. Loads track, arcs, and nodes
    2. Projects coordinates to local UTM Zone
    3. Runs HMM/Viterbi route decoding (using model probabilities or spatial baseline)
    4. Computes server-side metrics against ground-truth (kept private on server)
    5. Emits detailed console audit log
    """
    start_time = time.time()
    logger.info(f">>> [INCOMING REQUEST] Processing trip_id='{trip_id}', partition='{partition}', model='{model_name}' ...")

    # Locate partition
    if partition is None:
        for p in ["evaluator_test", "student_test", "train"]:
            if (DATA_ROOT / p / trip_id).is_dir():
                partition = p
                break
        if partition is None:
            logger.warning(f"Trip {trip_id} not found in any partition.")
            return {"error": f"Trip {trip_id} not found", "status_code": 404}

    try:
        trip = load_trip(DATA_ROOT, partition, trip_id)
        epsg = add_xy(trip.track)
        add_xy(trip.nodes, epsg)
        arcs_geom = arc_geometry(trip.nodes, trip.arcs)

        # Check for model prediction probabilities in cache or file
        classifier_probs: Optional[Dict[Tuple[int, int], float]] = None
        pred_parquet = PROJECT_ROOT / "outputs" / "models" / model_name / f"pred_{partition}.parquet"
        if pred_parquet.exists():
            try:
                cache_key = (model_name, partition)
                if cache_key not in PRECOMPUTED_CACHE:
                    logger.info(f"Loading predictions cache from {pred_parquet.name} ...")
                    PRECOMPUTED_CACHE[cache_key] = pd.read_parquet(pred_parquet)
                
                df_preds = PRECOMPUTED_CACHE[cache_key]
                trip_preds = df_preds[df_preds["trip_id"] == str(trip_id)]
                if len(trip_preds) > 0:
                    classifier_probs = {
                        (int(row["point_idx"]), int(row["arc_id"])): float(row["prob"])
                        for _, row in trip_preds.iterrows()
                    }
                    logger.info(f"Loaded {len(classifier_probs)} candidate probabilities from {model_name} cache.")
            except Exception as e_cache:
                logger.warning(f"Could not load prediction cache: {e_cache}")

        # Run HMM Viterbi decoding
        decode_res = decoder.decode(trip.track, trip.arcs, arcs_geom, classifier_probs=classifier_probs)
        predicted_route = decode_res["predicted_route"]
        greedy_route = decode_res["greedy_route"]
        n_points = decode_res["num_points"]

        # Compute Route Continuity
        continuity = decoder.compute_route_continuity(predicted_route, trip.arcs)
        greedy_continuity = decoder.compute_route_continuity(greedy_route, trip.arcs)

        # Server-side ground truth evaluation (never exposed to UI or client)
        accuracy = None
        greedy_acc = None
        has_route = trip.has_route

        if has_route:
            ref_arcs_set = set(trip.route.values)
            correct_points = sum(1 for a in predicted_route if a in ref_arcs_set)
            accuracy = float(correct_points / n_points)

            greedy_correct = sum(1 for a in greedy_route if a in ref_arcs_set)
            greedy_acc = float(greedy_correct / n_points)

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Mandatory console log for evaluators during live demonstration
        acc_str = f"Accuracy: {accuracy*100:.2f}%" if accuracy is not None else "Accuracy: N/A (unlabeled)"
        logger.info(
            f"<<< [SUCCESS] Trip {trip_id} ({partition}) | Points: {n_points} | "
            f"Viterbi Continuity: {continuity*100:.2f}% (Greedy: {greedy_continuity*100:.2f}%) | "
            f"{acc_str} | Latency: {elapsed_ms:.1f} ms"
        )

        return {
            "trip_id": trip_id,
            "partition": partition,
            "model_used": model_name,
            "status": "success",
            "num_gps_points": n_points,
            "predicted_route": predicted_route,
            "unique_predicted_arcs": len(set(predicted_route)),
            "route_continuity": round(continuity, 4),
            "greedy_continuity": round(greedy_continuity, 4),
            "top1_route_accuracy": round(accuracy, 4) if accuracy is not None else None,
            "greedy_accuracy": round(greedy_acc, 4) if greedy_acc is not None else None,
            "inference_time_ms": round(elapsed_ms, 2),
            "has_server_ground_truth": has_route
        }

    except Exception as e:
        logger.error(f"Error processing trip {trip_id}: {e}", exc_info=True)
        return {"error": str(e), "status_code": 500}


# ==============================================================================
# ASGI Application (Starlette / FastAPI Compatible)
# ==============================================================================

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="PathSnap Map-Matching & Route Reconstruction API",
        description="Two-laptop distributed map-matching system (Laptop 1 Server)",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index():
        return {
            "system": "PathSnap Map-Matching Backend",
            "role": "Laptop 1 (Model and API Server)",
            "status": "online",
            "partitions": get_available_partitions(),
            "endpoints": ["/trips", "/predict/{trip_id}", "/health"]
        }

    @app.get("/health")
    def health():
        return {"status": "healthy", "timestamp": time.time(), "partitions": get_available_partitions()}

    @app.get("/trips")
    def get_trips(partition: str = "evaluator_test"):
        try:
            trips = list_trips(DATA_ROOT, partition)
            logger.info(f"Listed {len(trips)} trips for partition '{partition}'")
            return {"partition": partition, "count": len(trips), "trips": trips}
        except Exception as e:
            logger.error(f"Error listing trips for '{partition}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/predict/{trip_id}")
    def predict(trip_id: str, partition: Optional[str] = None, model: str = "conditional_logit"):
        return run_prediction_pipeline(trip_id, partition, model_name=model)

except ImportError:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    async def index(request):
        return JSONResponse({
            "system": "PathSnap Map-Matching Backend",
            "role": "Laptop 1 (Model and API Server)",
            "status": "online",
            "partitions": get_available_partitions(),
            "endpoints": ["/trips", "/predict/{trip_id}", "/health"]
        })

    async def health(request):
        return JSONResponse({"status": "healthy", "timestamp": time.time(), "partitions": get_available_partitions()})

    async def get_trips(request):
        partition = request.query_params.get("partition", "evaluator_test")
        try:
            trips = list_trips(DATA_ROOT, partition)
            logger.info(f"Listed {len(trips)} trips for partition '{partition}'")
            return JSONResponse({"partition": partition, "count": len(trips), "trips": trips})
        except Exception as e:
            logger.error(f"Error listing trips for '{partition}': {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def predict(request):
        trip_id = request.path_params.get("trip_id")
        partition = request.query_params.get("partition")
        model = request.query_params.get("model", "conditional_logit")
        res = run_prediction_pipeline(trip_id, partition, model_name=model)
        return JSONResponse(res, status_code=res.get("status_code", 200))

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/trips", get_trips, methods=["GET"]),
        Route("/predict/{trip_id}", predict, methods=["GET"]),
    ]

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    app = Starlette(debug=True, routes=routes, middleware=middleware)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PathSnap API Server on host 0.0.0.0:8000 ...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
