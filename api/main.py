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
from hmm.viterbi import RouteDecoder

# Configure server console logging for live demo evaluation audit
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PATHSNAP-API] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pathsnap_api")

# HMM decoders. Candidates come from Person B's build_trip_candidates
# (k=8, radius 50 m), so the HMM sees exactly the rows the models were trained on.
decoder = RouteDecoder(radius_m=50.0, max_k=8, sigma_z=6.68)

# Emission weight for model probabilities, tuned on TRAIN trips only
# (python -m hmm.tune_emission_weight). 1.0 if the file is missing.
_w_file = PROJECT_ROOT / "outputs" / "hmm" / "emission_weight.json"
EMISSION_WEIGHT = json.loads(_w_file.read_text())["emission_weight"] if _w_file.exists() else 1.0
model_decoder = RouteDecoder(radius_m=50.0, max_k=8, sigma_z=6.68, emission_weight=EMISSION_WEIGHT)

# Trained models are loaded once and score each requested trip live.
# "nearest_arc" means the Gaussian distance emission (no trained model).
MODEL_CACHE: Dict[str, Any] = {}
LABEL_CUT_M = 40.0
_clean_cfg = PROJECT_ROOT / "outputs" / "clean" / "cleaning_config.json"
if _clean_cfg.exists():
    LABEL_CUT_M = float(json.loads(_clean_cfg.read_text())["label_exclude_far_m"])


def get_scorer(model_name: str):
    """Returns (score_fn or None, decoder). score_fn(cands) -> per-row probabilities."""
    if model_name == "nearest_arc":
        return None, decoder
    if model_name not in MODEL_CACHE:
        import joblib
        import models.baselines_linear as _bl
        # model.joblib was saved by running baselines_linear.py as a script, so the
        # pickle refers to __main__.<Class>; point those names at the real classes.
        _main = sys.modules["__main__"]
        for _name in ("ConditionalLogitModel", "Log1pGaussianNBModel", "NearestArcModel"):
            if not hasattr(_main, _name):
                setattr(_main, _name, getattr(_bl, _name))
        path = PROJECT_ROOT / "outputs" / "models" / model_name / "model.joblib"
        MODEL_CACHE[model_name] = joblib.load(path)
        logger.info(f"Loaded model {model_name} from {path}")
    model = MODEL_CACHE[model_name]

    def score_fn(cands: pd.DataFrame):
        df = cands.assign(trip_id="live")
        probs, _ = model.predict_probs(df)
        return probs

    return score_fn, model_decoder


def get_available_partitions() -> List[str]:
    """Finds all partitions present in data root."""
    parts = []
    for candidate in ["evaluator_test", "student_test", "train"]:
        if (DATA_ROOT / candidate).is_dir():
            parts.append(candidate)
    return parts


def get_available_models() -> List[str]:
    """Finds all trained models present in outputs/models."""
    models_dir = PROJECT_ROOT / "outputs" / "models"
    if not models_dir.is_dir():
        return ["conditional_logit", "gaussian_nb", "nearest_arc"]
    models = [d.name for d in sorted(models_dir.iterdir()) if d.is_dir()]
    if "conditional_logit" in models:
        models.remove("conditional_logit")
        models.insert(0, "conditional_logit")
    return models


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

        # Live pipeline: B's candidates + features -> model probabilities -> Viterbi.
        # Nothing here reads the reference route.
        score_fn, dec = get_scorer(model_name)
        decode_res = dec.decode_trip(trip, score_fn=score_fn)
        predicted_route = decode_res["predicted_route"]
        greedy_route = decode_res["greedy_route"]
        n_points = decode_res["num_points"]
        seg = decode_res["segment_id"]

        # Route continuity (steps across a >180 s gap are not counted)
        continuity = dec.compute_route_continuity(predicted_route, trip.arcs, seg)
        greedy_continuity = dec.compute_route_continuity(greedy_route, trip.arcs, seg)

        # Server-side ground-truth evaluation (never sent to the model; only numbers go to the UI).
        # Same Top-1 definition as the model comparison: usable points, chosen candidate has label 1.
        accuracy = None
        greedy_acc = None
        has_route = trip.has_route
        if has_route:
            from features.build_candidates import add_labels
            from hmm.benchmark_test_set import top1_on_usable
            labelled, _ = add_labels(decode_res["candidates"].copy(), decode_res["track_xy"],
                                     decode_res["geom"], trip.route, LABEL_CUT_M, trip.arcs)
            c, n = top1_on_usable(labelled, decode_res["point_idx"], predicted_route)
            accuracy = c / n if n else None
            c, n = top1_on_usable(labelled, decode_res["point_idx"], greedy_route)
            greedy_acc = c / n if n else None

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
            "greedy_route": greedy_route,
            "unique_predicted_arcs": len(set(predicted_route)),
            "route_continuity": round(continuity, 4),
            "greedy_continuity": round(greedy_continuity, 4),
            "top1_route_accuracy": round(accuracy, 4) if accuracy is not None else None,
            "greedy_accuracy": round(greedy_acc, 4) if greedy_acc is not None else None,
            "inference_time_ms": round(elapsed_ms, 2),
            "has_server_ground_truth": has_route,
            "num_segments": decode_res["num_segments"],
            "emission": "gaussian distance (sigma 6.68 m)" if score_fn is None
                        else f"{model_name} probabilities, weight {dec.emission_weight:g}",
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
            "models": get_available_models(),
            "endpoints": ["/trips", "/models", "/predict/{trip_id}", "/health"]
        }

    @app.get("/health")
    def health():
        return {"status": "healthy", "timestamp": time.time(), "partitions": get_available_partitions()}

    @app.get("/models")
    def get_models():
        return {"models": get_available_models()}

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
            "models": get_available_models(),
            "endpoints": ["/trips", "/models", "/predict/{trip_id}", "/health"]
        })

    async def health(request):
        return JSONResponse({"status": "healthy", "timestamp": time.time(), "partitions": get_available_partitions()})

    async def get_models(request):
        return JSONResponse({"models": get_available_models()})

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
        Route("/models", get_models, methods=["GET"]),
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
