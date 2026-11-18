"""
PathSnap Evaluator User Interface (Person D).
Runs on Laptop 2 (Evaluator Client Interface).
Communicates with Laptop 1 (FastAPI backend) over Wi-Fi/LAN.
Provides trip selection, model configuration, performance KPIs,
and rich interactive Folium route mapping.
"""
from __future__ import annotations

import os
import time
import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd

# Laptop 2 needs only streamlit, requests and pandas (ui/requirements.txt).
# Everything else (data, models, HMM, maps) runs on Laptop 1 behind the API.
DEFAULT_API = os.environ.get("PATHSNAP_API", "http://localhost:8000")

st.set_page_config(
    page_title="PathSnap Evaluator Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .main-title {
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction</h1>", unsafe_allow_html=True)
st.caption("PES University | Department of Computer Science & Engineering | Two-Laptop Distributed System (Laptop 2 Client)")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("📡 Server Connection (Laptop 1)")
api_host = st.sidebar.text_input(
    "Laptop 1 API URL",
    value=DEFAULT_API,
    help="Enter the IP address of Laptop 1, e.g., http://192.168.1.45:8000"
)

col_ping, col_status = st.sidebar.columns([1, 2])
with col_ping:
    ping_btn = st.button("Ping API")

api_connected = False
if ping_btn:
    try:
        t0 = time.time()
        r = requests.get(f"{api_host}/health", timeout=3)
        ms = (time.time() - t0) * 1000
        if r.status_code == 200:
            st.sidebar.success(f"Connected! ({ms:.1f} ms)")
            api_connected = True
        else:
            st.sidebar.warning(f"Status: {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"Cannot reach {api_host}. Check Wi-Fi / Windows Firewall.")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Evaluation Setup")

partition = st.sidebar.selectbox(
    "Dataset Partition",
    ["evaluator_test", "student_test", "train"],
    index=0,
    help="Select evaluator_test for the 10 final evaluation trips"
)

# Fetch trips from API
trips = []
try:
    resp = requests.get(f"{api_host}/trips?partition={partition}", timeout=4)
    if resp.status_code == 200:
        trips = resp.json().get("trips", [])
except Exception:
    pass  # API not reachable yet; the warning below explains what to check

if not trips:
    st.sidebar.warning(f"No trips for '{partition}'. Check the API URL above (Ping API) "
                       f"and that Laptop 1 has the '{partition}' folder.")
    selected_trip = None
else:
    selected_trip = st.sidebar.selectbox("Select Trip ID", trips, index=0)

# Models the API can run live (asked from Laptop 1)
available_models = ["nearest_arc", "conditional_logit", "gaussian_nb"]
try:
    m_resp = requests.get(f"{api_host}/models", timeout=2)
    if m_resp.status_code == 200:
        available_models = m_resp.json().get("models", available_models) or available_models
except Exception:
    pass

# Labels use results from outputs/hmm/final_scores_*.csv (HMM on top of each emission)
MODEL_DISPLAY_NAMES = {
    "nearest_arc": "Distance emission + HMM (final pipeline, evaluator Top-1 94.3%)",
    "conditional_logit": "Conditional logit + HMM (evaluator Top-1 93.4%)",
    "gaussian_nb": "Gaussian Naive Bayes + HMM",
}

model_choice = st.sidebar.selectbox(
    "Classifier Emission Model",
    available_models,
    format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, f"{x.replace('_', ' ').title()}"),
    index=0
)

run_button = st.sidebar.button("🚀 Reconstruct Route", type="primary", use_container_width=True)

# --- MAIN DASHBOARD AREA ---
if selected_trip:
    st.markdown(f"### 📍 Evaluation Trip: `{selected_trip}` &nbsp;|&nbsp; Partition: `{partition}`")

    if run_button:
        with st.spinner(f"Requesting prediction from Laptop 1 ({api_host}/predict/{selected_trip}) ..."):
            try:
                endpoint = f"{api_host}/predict/{selected_trip}?partition={partition}&model={model_choice}"
                res = requests.get(endpoint, timeout=120)
                if res.status_code != 200:
                    st.error(f"API Error ({res.status_code}): {res.text}")
                    data = None
                else:
                    data = res.json()
            except Exception as ex:
                st.error(f"Failed to communicate with Laptop 1 at {api_host}: {ex}")
                st.info("Tip: Verify both laptops are on the same Wi-Fi/hotspot and port 8000 is open in Windows Firewall.")
                data = None

        if data and data.get("status") == "success":
            # Metric KPIs
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total GPS Observations", data.get("num_gps_points", 0))

            with col2:
                acc = data.get("top1_route_accuracy")
                if acc is not None:
                    g_acc = data.get("greedy_accuracy", 0)
                    delta_acc = f"{(acc - g_acc)*100:+.2f}% vs greedy"
                    st.metric("Top-1 Route Accuracy", f"{acc * 100:.2f}%", delta_acc)
                else:
                    st.metric("Top-1 Route Accuracy", "Server-Side Ground Truth")

            with col3:
                cont = data.get("route_continuity", 0)
                g_cont = data.get("greedy_continuity", 0)
                delta_cont = f"{(cont - g_cont)*100:+.2f}% vs greedy"
                st.metric("Route Continuity", f"{cont * 100:.2f}%", delta_cont)

            with col4:
                st.metric("Unique Road Arcs", data.get("unique_predicted_arcs", 0))

            with col5:
                st.metric("Inference Latency", f"{data.get('inference_time_ms', 0):.1f} ms")

            # Comparative Analysis Expander
            with st.expander("📊 Sequence Continuity & Accuracy Comparison (Greedy vs. HMM)", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Greedy Point-by-Point Classification (No HMM)**")
                    st.write(f"- Route Continuity: **{data.get('greedy_continuity', 0)*100:.2f}%** (frequent jumps across parallel streets)")
                    if data.get("greedy_accuracy") is not None:
                        st.write(f"- Point Accuracy: **{data.get('greedy_accuracy', 0)*100:.2f}%**")
                with col_b:
                    st.markdown("**Global HMM & Viterbi Sequence Decoding**")
                    st.write(f"- Route Continuity: **{data.get('route_continuity', 0)*100:.2f}%** (topologically connected sequence)")
                    if data.get("top1_route_accuracy") is not None:
                        st.write(f"- Point Accuracy: **{data.get('top1_route_accuracy', 0)*100:.2f}%**")

            # Interactive Folium Map
            st.markdown("### 🗺️ Interactive Geographical Map (Folium)")
            try:
                with st.spinner("Building map on Laptop 1 ..."):
                    m = requests.get(f"{api_host}/map/{selected_trip}",
                                     params={"partition": partition, "model": model_choice}, timeout=60)
                if m.status_code == 200:
                    components.html(m.text, height=600, scrolling=True)
                    st.caption("💡 **Tip:** Use the Layer Control in the top-right corner of the map to toggle raw GPS points, road network, reference route, and predicted route.")
                else:
                    st.warning(f"Could not render map: API returned {m.status_code}: {m.text[:200]}")
            except Exception as map_err:
                st.warning(f"Could not render map: {map_err}")

            # Predicted Route Table
            with st.expander("📄 View & Export Predicted Sequence"):
                pred_route = data.get("predicted_route", [])
                df_pred = pd.DataFrame({
                    "GPS Point Index": list(range(len(pred_route))),
                    "Predicted Arc ID": pred_route
                })
                st.dataframe(df_pred, use_container_width=True)

                csv_data = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predicted Route CSV",
                    data=csv_data,
                    file_name=f"predicted_route_{selected_trip}_{model_choice}.csv",
                    mime="text/csv"
                )

    else:
        st.info("👈 Select an evaluation trip from the sidebar and click **'🚀 Reconstruct Route'** to run the prediction pipeline.")
