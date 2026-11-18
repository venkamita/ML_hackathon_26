"""
PathSnap Evaluator User Interface (Person D).
Runs on Laptop 2 (Evaluator Client Interface).
Communicates with Laptop 1 (FastAPI backend) over Wi-Fi/LAN, with local fallback.
Provides trip selection, multi-model execution, live performance KPIs,
interactive Folium route mapping, comparative leaderboards, and viva preparation.
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import numpy as np

# Configure path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Laptop 2 normally needs only ui/requirements.txt (streamlit, requests, pandas, numpy):
# trips, predictions and maps all come from the Laptop 1 API. Local mode (running the
# pipeline in this process) is only a fallback when the full repo, its requirements
# and the raw data are present on this machine.
DEFAULT_API = os.environ.get("PATHSNAP_API", "http://localhost:8000")
_PARTS = ("evaluator_test", "student_test", "train")


def _find_data_root() -> Path:
    env = os.environ.get("PATHSNAP_DATA")
    if env:
        return Path(env).expanduser()
    for base in (PROJECT_ROOT.parent, PROJECT_ROOT):
        if any((base / p).is_dir() for p in _PARTS):
            return base
        for child in sorted(c for c in base.iterdir() if c.is_dir()):
            if child != PROJECT_ROOT and any((child / p).is_dir() for p in _PARTS):
                return child
    return PROJECT_ROOT.parent


DATA_ROOT = _find_data_root()
try:
    from common.data_loader import load_trip
    from viz.folium_map import make_trip_map
    LOCAL_OK = any((DATA_ROOT / p).is_dir() for p in _PARTS)
except Exception:
    LOCAL_OK = False

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 13px;
        text-transform: uppercase;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #212529;
    }
    .metric-delta {
        font-size: 13px;
        font-weight: 600;
        color: #28a745;
    }
    .main-title {
        color: #1a365d;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .subtitle {
        color: #4a5568;
        font-size: 14px;
        margin-bottom: 20px;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 9px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 12px;
        background-color: #e2e8f0;
        color: #2d3748;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>PES University | Dept of Computer Science & Engineering | Two-Laptop Distributed System (Laptop 2 Client)</div>", unsafe_allow_html=True)

# ==============================================================================
# Sidebar: Connection & Configuration
# ==============================================================================
st.sidebar.header("📡 Server Connection (Laptop 1)")
api_host = st.sidebar.text_input(
    "FastAPI Host URL",
    value=DEFAULT_API,
    help="Enter the IP address of Laptop 1, e.g., http://192.168.1.45:8000"
)

col_ping, col_status = st.sidebar.columns([1, 2])
with col_ping:
    ping_btn = st.button("Ping API")

api_connected = False
try:
    r = requests.get(f"{api_host}/health", timeout=1.5)
    if r.status_code == 200:
        api_connected = True
except Exception:
    api_connected = False

if ping_btn:
    if api_connected:
        st.sidebar.success("✅ Connected to Laptop 1!")
    else:
        st.sidebar.error(f"❌ Cannot connect to {api_host}. Check the IP, Wi-Fi and Laptop 1's firewall"
                         + (" (local fallback is available)." if LOCAL_OK else "."))

if api_connected:
    st.sidebar.caption("🟢 **Status:** Online (Communicating with Laptop 1)")
else:
    st.sidebar.caption("🟡 **Status:** Offline / Local Fallback" if LOCAL_OK else "🔴 **Status:** Offline (enter Laptop 1's URL)")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Evaluation Setup")

partition = st.sidebar.selectbox(
    "Dataset Partition",
    ["evaluator_test", "student_test", "train"],
    index=0,
    help="Select evaluator_test for the 10 final evaluation trips"
)

# Fetch trips
trips = []
if api_connected:
    try:
        resp = requests.get(f"{api_host}/trips?partition={partition}", timeout=3)
        if resp.status_code == 200:
            trips = resp.json().get("trips", [])
    except Exception:
        pass

if not trips and LOCAL_OK:
    part_dir = DATA_ROOT / partition
    if part_dir.is_dir():
        trips = sorted([d.name for d in part_dir.iterdir() if d.is_dir()])

if not trips:
    st.sidebar.warning(f"No trips found for '{partition}'.")
    selected_trip = None
else:
    selected_trip = st.sidebar.selectbox("Select Trip ID", trips, index=0)

# Discover models: the API lists only models it can run live
LIVE_MODELS = ["nearest_arc", "conditional_logit", "gaussian_nb"]
available_models = list(LIVE_MODELS)
if api_connected:
    try:
        m_resp = requests.get(f"{api_host}/models", timeout=2)
        if m_resp.status_code == 200:
            available_models = m_resp.json().get("models", available_models) or available_models
    except Exception:
        pass

# Every model with saved results in the repo (for the leaderboard / diagnostics tabs)
_models_dir = PROJECT_ROOT / "outputs" / "models"
ALL_RESULT_MODELS = sorted(d.name for d in _models_dir.iterdir()
                           if d.is_dir() and (d / "summary.json").exists()) if _models_dir.is_dir() else []

# Numbers from outputs/hmm/final_scores_*.csv and outputs/models/*/summary.json
MODEL_DISPLAY_NAMES = {
    "nearest_arc": "Distance emission + HMM (final pipeline, evaluator Top-1 94.3%)",
    "conditional_logit": "Conditional logit + HMM, w=0.05 (evaluator Top-1 93.4%)",
    "gaussian_nb": "Gaussian Naive Bayes + HMM",
    "lgbm_rank": "LightGBM LambdaRank (best classifier, CV Top-1 87.0%)",
    "lgbm_binary": "LightGBM binary (CV Top-1 86.6%)",
    "extra_trees": "Extra Trees (CV Top-1 85.6%)",
    "mlp": "MLP 64-32 (CV Top-1 86.3%)",
    "mlp_tuned": "MLP, Optuna-tuned (CV Top-1 86.3%)",
}

model_choice = st.sidebar.selectbox(
    "Classifier Emission Model",
    available_models,
    format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, f"{x.replace('_', ' ').title()}"),
    index=0
)

run_button = st.sidebar.button("🚀 Reconstruct Route", type="primary", use_container_width=True)

# ==============================================================================
# Helper: Run Inference (API or Local Fallback)
# ==============================================================================
def execute_route_reconstruction(trip_id: str, part: str, model: str) -> Optional[Dict[str, Any]]:
    # Attempt 1: Call Laptop 1 REST API
    if api_connected:
        try:
            endpoint = f"{api_host}/predict/{trip_id}?partition={part}&model={model}"
            res = requests.get(endpoint, timeout=40)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            st.warning(f"API request failed ({e}); switching to local inference.")

    # Attempt 2: Local fallback (needs the full repo, requirements and data here)
    if not LOCAL_OK:
        st.error("Laptop 1 API is not reachable and this machine has no local data. "
                 "Enter Laptop 1's URL in the sidebar and click Ping API.")
        return None
    try:
        from api.main import run_prediction_pipeline
        return run_prediction_pipeline(trip_id, partition=part, model_name=model)
    except Exception as e_local:
        st.error(f"Execution error: {e_local}")
        return None

# ==============================================================================
# Main Dashboard Tabs
# ==============================================================================
tab_reconstruct, tab_leaderboard, tab_models, tab_architecture = st.tabs([
    "🗺️ Route Reconstruction",
    "🏆 Benchmark Leaderboard",
    "🔬 Model Diagnostics",
    "📐 Architecture & Viva Guide"
])

# ------------------------------------------------------------------------------
# TAB 1: Route Reconstruction & Interactive Map
# ------------------------------------------------------------------------------
with tab_reconstruct:
    if selected_trip:
        st.markdown(f"### 📍 Trip: `{selected_trip}` &nbsp;|&nbsp; Partition: `{partition}` &nbsp;|&nbsp; Model: `{model_choice}`")

        # Session state caching for persistent map interaction
        state_key = f"result_{partition}_{selected_trip}_{model_choice}"
        
        if run_button or state_key in st.session_state:
            if run_button or state_key not in st.session_state:
                with st.spinner(f"Decoding route for Trip {selected_trip} via Viterbi HMM ..."):
                    data = execute_route_reconstruction(selected_trip, partition, model_choice)
                    if data and data.get("status") == "success":
                        st.session_state[state_key] = data
            else:
                data = st.session_state[state_key]

            if data and data.get("status") == "success":
                # --- KPI Metrics Row ---
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Observations", f"{data.get('num_gps_points', 0):,}")
                with col2:
                    cont = data.get("route_continuity", 0)
                    g_cont = data.get("greedy_continuity", 0)
                    delta_cont = f"{(cont - g_cont)*100:+.2f}% vs Greedy"
                    st.metric("Route Continuity (HMM)", f"{cont * 100:.2f}%", delta_cont)
                with col3:
                    acc = data.get("top1_route_accuracy")
                    if acc is not None:
                        g_acc = data.get("greedy_accuracy", 0)
                        delta_acc = f"{(acc - g_acc)*100:+.2f}% vs Greedy"
                        st.metric("Point Accuracy (HMM)", f"{acc * 100:.2f}%", delta_acc)
                    else:
                        st.metric("Point Accuracy", "Private Server-Side")
                with col4:
                    st.metric("Unique Road Arcs", data.get("unique_predicted_arcs", 0))
                with col5:
                    st.metric("Pipeline Latency", f"{data.get('inference_time_ms', 0):.1f} ms")

                st.markdown("---")

                # --- Comparative Analysis Box ---
                with st.expander("📊 Sequence Continuity & Accuracy Comparison (Greedy vs. HMM)", expanded=True):
                    col_greedy, col_hmm = st.columns(2)
                    with col_greedy:
                        st.markdown("##### ❌ Greedy Classification (No HMM)")
                        st.write(f"- **Route Continuity:** `{data.get('greedy_continuity', 0)*100:.2f}%`")
                        if data.get("greedy_accuracy") is not None:
                            st.write(f"- **Point-Level Accuracy:** `{data.get('greedy_accuracy', 0)*100:.2f}%`")
                        st.caption("Independent point scoring causes frequent jumps across parallel arcs and disconnected segments.")
                    with col_hmm:
                        st.markdown("##### ✅ Global Viterbi Decoding (HMM)")
                        st.write(f"- **Route Continuity:** `{data.get('route_continuity', 0)*100:.2f}%`")
                        if data.get("top1_route_accuracy") is not None:
                            st.write(f"- **Point-Level Accuracy:** `{data.get('top1_route_accuracy', 0)*100:.2f}%`")
                        st.caption("Topological transition matrix enforces valid road transitions, eliminating zigzagging.")

                # --- Interactive Folium Map ---
                st.markdown("### 🗺️ Interactive Geographical Map")
                map_html = None
                try:
                    if api_connected:
                        with st.spinner("Building map on Laptop 1 ..."):
                            m = requests.get(f"{api_host}/map/{selected_trip}",
                                             params={"partition": partition, "model": model_choice},
                                             timeout=90)
                        if m.status_code == 200:
                            map_html = m.text
                        else:
                            st.warning(f"Map request failed ({m.status_code}): {m.text[:200]}")
                    elif LOCAL_OK:
                        trip = load_trip(DATA_ROOT, partition, selected_trip)
                        temp_map_path = PROJECT_ROOT / "outputs" / "maps" / f"ui_map_{selected_trip}_{model_choice}.html"
                        temp_map_path.parent.mkdir(parents=True, exist_ok=True)
                        make_trip_map(trip=trip, out_html=temp_map_path,
                                      predicted_arcs=data.get("predicted_route", []),
                                      greedy_arcs=data.get("greedy_route", None))
                        map_html = temp_map_path.read_text(encoding="utf-8")
                    if map_html:
                        components.html(map_html, height=620, scrolling=True)
                        st.caption("💡 **Layer Controls (top-right of map):** Toggle Road Network (gray), GPS Points (blue), Reference Route (green), Predicted Viterbi Route (red dashed), and Greedy Route (orange dotted).")
                except Exception as map_err:
                    st.warning(f"Could not render map: {map_err}")

                # --- Tabular Sequence Inspector ---
                with st.expander("📄 Point-by-Point Arc Prediction Inspector & Export"):
                    pred_route = data.get("predicted_route", [])
                    greedy_route = data.get("greedy_route", [])
                    
                    pts = data.get("points") or {}
                    n = len(pred_route)
                    inspector_data = {
                        "Point Index": list(range(n)),
                        "Timestamp (s)": (pts.get("t") or [None] * n)[:n],
                        "Latitude": (pts.get("lat") or [None] * n)[:n],
                        "Longitude": (pts.get("lon") or [None] * n)[:n],
                        "HMM Predicted Arc": pred_route,
                        "Greedy Arc": greedy_route[:n] if greedy_route else [None] * n,
                    }
                    inspector_data["HMM = Greedy"] = [
                        "same" if g == p else "changed by HMM"
                        for p, g in zip(pred_route, inspector_data["Greedy Arc"])
                    ]
                    df_inspector = pd.DataFrame(inspector_data)
                    st.dataframe(df_inspector, use_container_width=True, height=280)

                    csv_bytes = df_inspector.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Sequence Predictions (CSV)",
                        data=csv_bytes,
                        file_name=f"predictions_{partition}_{selected_trip}_{model_choice}.csv",
                        mime="text/csv"
                    )

        else:
            st.info("👆 Click **'🚀 Reconstruct Route'** in the sidebar to run the HMM decoding pipeline for this trip.")

# ------------------------------------------------------------------------------
# TAB 2: Benchmark Leaderboard & Comparisons
# ------------------------------------------------------------------------------
with tab_leaderboard:
    st.markdown("### 🏆 Hackathon Benchmark Leaderboard")
    st.markdown("Evaluation metrics computed over all usable GPS points following Common Evaluation Rules.")

    bench_part = st.radio("Select Benchmark Dataset:", ["evaluator_test (10 Reserved Final Trips)", "student_test (10 Test Trips)"], horizontal=True)
    bench_file = "final_scores_evaluator_test.csv" if "evaluator" in bench_part else "final_scores_student_test.csv"
    bench_path = PROJECT_ROOT / "outputs" / "hmm" / bench_file

    if bench_path.exists():
        df_bench = pd.read_csv(bench_path)
        
        # Display formatted leaderboard table
        df_display = df_bench.copy()
        df_display["Top-1 Accuracy"] = (df_display["top1_acc"] * 100).map("{:.2f}%".format)
        df_display["Macro F1"] = df_display["macro_f1"].map("{:.4f}".format)
        df_display["Route Continuity"] = (df_display["route_continuity"] * 100).map("{:.2f}%".format)
        df_display["Total Points"] = df_display["n_usable_points"].map("{:,}".format)
        df_display = df_display.rename(columns={"pipeline": "Pipeline Architecture"})
        
        st.dataframe(
            df_display[["Pipeline Architecture", "Top-1 Accuracy", "Macro F1", "Route Continuity", "Total Points"]],
            use_container_width=True,
            hide_index=True
        )

        # Comparative Visualization Charts
        st.markdown("#### 📊 Accuracy & Continuity Comparison")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Top-1 Accuracy (%)**")
            chart_acc = df_bench.set_index("pipeline")[["top1_acc"]] * 100
            chart_acc.columns = ["Accuracy (%)"]
            st.bar_chart(chart_acc)

        with col_chart2:
            st.markdown("**Route Continuity (%)**")
            chart_cont = df_bench.set_index("pipeline")[["route_continuity"]] * 100
            chart_cont.columns = ["Continuity (%)"]
            st.bar_chart(chart_cont)

        st.caption("📌 **Key takeaway:** per-point classification alone reaches 49–85% Top-1 and 82–94% continuity; adding HMM/Viterbi lifts continuity to about 99%, and the distance emission + HMM is the best pipeline (evaluator Top-1 94.3%).")
    else:
        st.warning(f"Benchmark file {bench_file} not found.")

    # Model Cross-Validation Summary
    st.markdown("---")
    st.markdown("#### 🧪 Classifier Leaderboard: 5-Fold CV (train) and Student-Test Top-1, no HMM")
    cv_records = []
    models_dir = PROJECT_ROOT / "outputs" / "models"
    if models_dir.is_dir():
        for mdir in sorted(models_dir.iterdir()):
            if mdir.is_dir():
                sfile = mdir / "summary.json"
                if sfile.exists():
                    try:
                        sdata = json.loads(sfile.read_text(encoding="utf-8"))
                        m_name = mdir.name
                        cv_file = mdir / "cv_folds.csv"
                        cv_acc = float(pd.read_csv(cv_file)["top1_acc"].mean()) if cv_file.exists() else None
                        st_block = sdata.get("student_test_metrics") or sdata.get("student_test") or {}
                        test_acc = st_block.get("top1_acc")
                        cv_records.append({
                            "Model Name": MODEL_DISPLAY_NAMES.get(m_name, m_name),
                            "5-Fold CV Top-1 Acc": f"{cv_acc*100:.2f}%" if cv_acc else "N/A",
                            "Student-Test Top-1 Acc": f"{test_acc*100:.2f}%" if test_acc else "N/A",
                            "Model Category": "Linear Utility" if "logit" in m_name else ("Probabilistic" if "nb" in m_name else ("Spatial" if "nearest" in m_name else "Ensemble / NN"))
                        })
                    except Exception:
                        pass
    if cv_records:
        st.dataframe(pd.DataFrame(cv_records), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 3: Model Diagnostics & Feature Analysis
# ------------------------------------------------------------------------------
with tab_models:
    st.markdown("### 🔬 Model Diagnostics & Feature Inspection")
    selected_diag_model = st.selectbox(
        "Select Model to Inspect:",
        ALL_RESULT_MODELS or available_models,
        format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x),
        key="diag_model_select"
    )

    diag_dir = PROJECT_ROOT / "outputs" / "models" / selected_diag_model
    if diag_dir.is_dir():
        col_diag_left, col_diag_right = st.columns([1, 1])

        with col_diag_left:
            st.markdown("#### 🎯 Confusion Matrix")
            eval_cm = diag_dir / "evaluator_test_confusion_matrix.png"
            if not eval_cm.exists():
                eval_cm = diag_dir / "confusion_matrix.png"
            if eval_cm.exists():
                st.image(str(eval_cm), caption=f"Confusion Matrix — {selected_diag_model}", use_container_width=True)
            else:
                st.info("Confusion matrix image not found.")

        with col_diag_right:
            st.markdown("#### 📈 Optimization & Tuning")
            bayes_img = diag_dir / "bayes_opt_convergence.png"
            loss_img = diag_dir / "loss_curves.png"
            lcurve_img = diag_dir / "learning_curve_fold_0.png"

            if bayes_img.exists():
                st.image(str(bayes_img), caption="Bayesian Hyperparameter Optimization (Gaussian Process Regressor)", use_container_width=True)
            elif loss_img.exists():
                st.image(str(loss_img), caption="Training Epoch Loss Curve", use_container_width=True)
            elif lcurve_img.exists():
                st.image(str(lcurve_img), caption="Fold 0 Learning Curve", use_container_width=True)
            else:
                st.info("No tuning curve graphic available for this model.")

        # Coefficients / Feature Importances
        st.markdown("#### ⚖️ Feature Weights & Importances")
        coeff_file = diag_dir / "coefficients.csv"
        feat_file = next((f for f in (diag_dir / "feature_importance.csv", diag_dir / "feature_importances.csv")
                          if f.exists()), None)

        if coeff_file.exists():
            df_coef = pd.read_csv(coeff_file)
            st.markdown("**Learned Linear Utility Coefficients ($\beta$):**")
            st.dataframe(df_coef, use_container_width=True, height=250)
            st.caption("💡 **Interpretation:** Negative coefficients on `heading_diff_deg` and `dist_m` strongly penalize candidates misaligned with vehicle direction or far from GPS track.")
        elif feat_file and feat_file.exists():
            df_feat = pd.read_csv(feat_file)
            st.dataframe(df_feat, use_container_width=True, height=250)

# ------------------------------------------------------------------------------
# TAB 4: Architecture & Viva Guide
# ------------------------------------------------------------------------------
with tab_architecture:
    st.markdown("### 📐 System Architecture & Viva Defense Reference")

    col_arch1, col_arch2 = st.columns([1, 1])

    with col_arch1:
        st.markdown("""
        #### 🖥️ Two-Laptop Distributed System Architecture
        * **Laptop 1 (Model Server & Backend API):**
          * Framework: FastAPI / Starlette ASGI Server (`api/main.py`) running on `0.0.0.0:8000`.
          * Responsibility: Loads pre-fitted model weights (`joblib`), candidate projection, HMM Viterbi decoding, server-side private ground-truth evaluation, and live console audit logs.
          * Security: Ground-truth routes (`.route`) remain strictly on the server; only reconstructed paths and aggregate metrics are returned over the network.
        * **Laptop 2 (Evaluator Client Interface):**
          * Framework: Streamlit (`ui/app.py`) running on `localhost:8501`.
          * Responsibility: Interactive GUI for trip selection, model benchmarking, Folium geographic mapping, and sequence inspection.
        """)

    with col_arch2:
        st.markdown("""
        #### 🧮 HMM Mathematical Formulation
        * **Emission Probability:**
          $$P(z_t \\mid a_t) \\propto \\exp\\left(-\\frac{1}{2}\\left(\\frac{\\text{dist}(z_t, a_t)}{\\sigma_z}\\right)^2\\right)$$
          or scaled classifier score: $p_{\\text{model}}(a_t \\mid x_t)^w$ with tuned $w = 0.05$.
        * **Transition Probability:**
          $$P(a_t \\mid a_{t-1}) = \\begin{cases}
          e^{-0.43} & \\text{if } a_t = a_{t-1} \\text{ (same arc)} \\\\
          e^{-1.20} & \\text{if } a_t \\text{ is direct successor} \\\\
          e^{-3.22} & \\text{if } a_t \\text{ is 2-hop connected} \\\\
          e^{-12.0} & \\text{if disconnected (jump penalty)}
          \\end{cases}$$
        * **Viterbi Trellis:** Solves $\\arg\\max_{a_{1:T}} \\prod_{t=1}^T P(z_t \\mid a_t) P(a_t \\mid a_{t-1})$ via dynamic programming in $O(T \\cdot K^2)$ time.
        """)

    st.markdown("---")
    st.markdown("#### 🎓 Viva Defense: Key Questions & Answers")

    with st.expander("Q1: Why does Greedy Point Classification produce poor route reconstruction?"):
        st.write(r"""
        Greedy classification evaluates each GPS point in isolation ($\arg\max p(a_t \mid x_t)$).
        GPS noise ($6.68$ m std dev) and parallel service roads or overpasses frequently cause the point-level classifier to jump between adjacent roads. This yields low route continuity (~82% - 93%) and an un-drivable route.
        The HMM enforces physical vehicle dynamics via topological transition penalties.
        """)

    with st.expander("Q2: Why does Conditional Logit outperform standard Logistic Regression?"):
        st.write(r"""
        Standard Logistic Regression treats every candidate arc as an independent binary classification instance ($\sigma(z)$), ignoring that the candidates belong to the same GPS point.
        Conditional Logit explicitly models candidate competition within each point via a grouped softmax:
        $$p_i = \frac{\exp(\beta^T x_i)}{\sum_{j \in \text{cands}} \exp(\beta^T x_j)}$$
        Point-constant variables (e.g. speed, $\Delta t$) naturally cancel out, focusing optimization entirely on features that discriminate between rival road arcs.
        """)

    with st.expander("Q3: Why was the classifier emission weight tuned to $w = 0.05$?"):
        st.write(r"""
        Modern ML models (MLPs, boosted trees, logits) produce highly overconfident probabilities ($p > 0.99$), causing the emission term to dominate the HMM log-likelihood and drown out the transition penalties.
        Down-weighting the emission by $w = 0.05$ calibrates the emission scores against graph transition costs, achieving **99.36% - 99.74% route continuity** without sacrificing topological accuracy.
        """)
