import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import os

from src import config
from src.models.architectures import get_model

# =========================================================================
# 1. CENTRAL CONFIGURATION
# =========================================================================
ACTIVE_MODEL_TYPE   = config.ACTIVE_MODEL_TYPE
ACTIVE_MODEL_WEIGHTS = config.ACTIVE_MODEL_WEIGHTS
WINDOW_SIZE         = config.WINDOW_SIZE
STEP_SIZE           = config.STEP_SIZE
CLASS_NAMES         = config.CLASS_NAMES
COLUMNS_TO_DROP     = config.COLUMNS_TO_DROP
SCALER_PATH         = config.SCALER_PATH

FAULT_COLORS = {
    0: {"name": "Normal",           "color": "rgba(34, 197, 94, 0.2)",  "solid": "#22C55E", "emoji": "✅"},
    1: {"name": "Engine Failure",   "color": "rgba(239, 68, 68, 0.4)",  "solid": "#EF4444", "emoji": "🔴"},
    2: {"name": "Elevator Failure", "color": "rgba(249, 115, 22, 0.4)", "solid": "#F97316", "emoji": "🟠"},
    3: {"name": "Rudder Failure",   "color": "rgba(234, 179, 8, 0.4)",  "solid": "#EAB308", "emoji": "🟡"},
    4: {"name": "Aileron Failure",  "color": "rgba(22, 163, 74, 0.4)",  "solid": "#16A34A", "emoji": "🟢"},
    5: {"name": "Multi-Fault",      "color": "rgba(147, 51, 234, 0.4)", "solid": "#9333EA", "emoji": "🟣"},
}

# ==========================================
# 2. MODEL & SCALER LOADING
# ==========================================
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(SCALER_PATH):
        st.error(f"❌ Scaler not found: {SCALER_PATH}")
        st.stop()
    scaler = joblib.load(SCALER_PATH)

    if not os.path.exists(ACTIVE_MODEL_WEIGHTS):
        st.error(f"❌ Model weights not found: {ACTIVE_MODEL_WEIGHTS}")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(ACTIVE_MODEL_TYPE, scaler.n_features_in_, config.NUM_CLASSES, device)
    model.load_state_dict(torch.load(ACTIVE_MODEL_WEIGHTS, map_location=device))
    model.eval()
    return scaler, model, device


# =========================================================================
# 3. TIMESTAMP AUTO-NORMALIZATION  (Fix #1 – "Big Number" problem)
# =========================================================================
def normalize_timestamps(raw: np.ndarray) -> np.ndarray:
    """
    Detect the timestamp scale and return flight-relative seconds (starts at 0).
    Handles:
      - ROS nanoseconds  (values > 1e15)
      - UNIX microseconds (values > 1e12)
      - UNIX milliseconds (values > 1e9)
      - Already seconds   (values <= 1e9)
    """
    max_val = float(np.max(raw))
    if max_val > 1e15:
        scale = 1e9        # nanoseconds → seconds
    elif max_val > 1e12:
        scale = 1e6        # microseconds → seconds
    elif max_val > 1e9:
        scale = 1e3        # milliseconds → seconds
    else:
        scale = 1.0        # already seconds

    t_sec = raw / scale
    return t_sec - t_sec[0]   # start at 0


# =========================================================================
# 4. SENSOR GROUPING  (Fix #2 – broken numeric check in original)
# =========================================================================
def get_sensor_groups(df: pd.DataFrame, feature_cols: list) -> dict:
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    groups = {
        "🎯 Model Features":        [c for c in all_numeric if c in feature_cols],
        "📐 Orientation (Attitude)":[c for c in all_numeric if 'orientation' in c.lower()],
        "🔄 Motion (IMU/Gyro)":     [c for c in all_numeric if 'angular_velocity' in c.lower()
                                                              or 'linear_acceleration' in c.lower()],
        "🌍 Position & Altitude":   [c for c in all_numeric if 'position' in c.lower() or 'alt' in c.lower()],
        "💨 Navigation & Control":  [c for c in all_numeric if any(k in c.lower()
                                     for k in ['nav_info', 'airspeed', 'pitch', 'roll', 'yaw'])],
        "🔋 Power System":          [c for c in all_numeric if 'battery' in c.lower()],
        "📦 All Numeric Columns":   all_numeric,
    }
    return {k: v for k, v in groups.items() if v}


# =========================================================================
# 5. DATA PROCESSING & PREDICTION
# =========================================================================
def preprocess_and_predict(uploaded_file, scaler, model, device):
    df = pd.read_csv(uploaded_file)

    # ── Find & clean time column ──────────────────────────────────────────
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    if not time_cols:
        st.error("❌ No column containing 'time' or '%time' found in the dataset.")
        st.stop()

    time_col = time_cols[0]
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')   # Fix #3 – force numeric
    df = df.dropna(subset=[time_col]).reset_index(drop=True)
    raw_timestamps = df[time_col].values

    # ── Feature columns ───────────────────────────────────────────────────
    RAW_COLS_TO_DROP = [
        "mavros-imu-data_raw.field.angular_velocity.x",
        "mavros-imu-data_raw.field.angular_velocity.y",
        "mavros-imu-data_raw.field.angular_velocity.z",
        "mavros-imu-data_raw.field.linear_acceleration.x",
        "mavros-imu-data_raw.field.linear_acceleration.y",
        "mavros-imu-data_raw.field.linear_acceleration.z",
    ]
    drop_set = set(
        [c for c in COLUMNS_TO_DROP if c in df.columns] +
        [c for c in RAW_COLS_TO_DROP if c in df.columns]
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in drop_set]

    X_df = df[feature_cols]
    if X_df.shape[1] != scaler.n_features_in_:
        st.error(
            f"❌ Feature count mismatch!  "
            f"Expected: **{scaler.n_features_in_}**  |  Got: **{X_df.shape[1]}**\n\n"
            f"Make sure you are uploading a CSV exported with the same ROS topics used during training."
        )
        st.stop()

    st.info("✅ Data validated — scaling features…")
    X_scaled = scaler.transform(X_df)

    # ── Sliding-window inference ──────────────────────────────────────────
    st.info("🤖 Running sliding-window inference…")
    predictions, pred_ts_raw, confidences = [], [], []
    indices      = range(0, len(X_scaled) - WINDOW_SIZE + 1, STEP_SIZE)
    progress_bar = st.progress(0)
    total_steps  = len(indices)

    for step_count, i in enumerate(indices):
        window   = X_scaled[i : i + WINDOW_SIZE]
        tensor   = torch.FloatTensor(window).unsqueeze(0).to(device)

        # store both start and end timestamps of the window
        t_start_raw = raw_timestamps[i]
        t_end_raw   = raw_timestamps[i + WINDOW_SIZE - 1]

        with torch.no_grad():
            logits  = model(tensor)
            probs   = torch.softmax(logits, dim=1)
            conf, pred_class = torch.max(probs, dim=1)

        predictions.append(pred_class.item())
        confidences.append(conf.item())
        pred_ts_raw.append((t_start_raw, t_end_raw))

        if (step_count + 1) % 50 == 0 or (step_count + 1) == total_steps:
            progress_bar.progress((step_count + 1) / total_steps)

    st.success("✅ Inference complete — building visualizations…")
    return df, raw_timestamps, pred_ts_raw, predictions, confidences, time_col, feature_cols


# =========================================================================
# 6. MAIN APP
# =========================================================================
st.set_page_config(page_title="PX4 AI Fault Classifier", page_icon="🚁", layout="wide")
st.title("🚁 PX4 AI Fault Classifier")
st.markdown(
    "Upload raw drone telemetry (ROS bag CSV). "
    "The AI model detects and **temporally visualizes** faults across the entire flight."
)

# Sidebar
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.info(
    f"**Model:** {ACTIVE_MODEL_TYPE.upper()}\n\n"
    f"**Window:** {WINDOW_SIZE} samples\n\n"
    f"**Step:** {STEP_SIZE} samples"
)
uploaded_file = st.sidebar.file_uploader("Upload a ROS Bag CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

st.success("✅ File uploaded!")

with st.spinner("Running AI analysis…"):
    scaler, model, device = load_model_and_scaler()
    (original_df, raw_timestamps, pred_ts_raw,
     predictions, confidences, time_col, feature_cols) = preprocess_and_predict(
        uploaded_file, scaler, model, device
    )

# ── Normalize all timestamps to T=0 seconds ──────────────────────────────
timestamps_sec = normalize_timestamps(raw_timestamps)

# Compute normalized window start/end for every prediction
# We find the index of each stored raw timestamp to map it to normalized time
raw_to_norm = dict(zip(raw_timestamps, timestamps_sec))   # fast lookup

window_data = []
for idx, ((t_start_raw, t_end_raw), pred, conf) in enumerate(
        zip(pred_ts_raw, predictions, confidences)):
    # Use closest available mapped value (handles float precision)
    t_s = raw_to_norm.get(t_start_raw, timestamps_sec[0])
    t_e = raw_to_norm.get(t_end_raw,   timestamps_sec[-1])
    window_data.append({
        "Window #":       idx + 1,
        "Start (s)":      round(t_s, 3),
        "End (s)":        round(t_e, 3),
        "prediction":     int(pred),
        "confidence_raw": float(conf),
        "Fault Class":    f"{FAULT_COLORS[int(pred)]['emoji']} {FAULT_COLORS[int(pred)]['name']}",
        "Confidence":     f"{conf * 100:.1f}%",
    })

pred_df = pd.DataFrame(window_data)

# =========================================================================
# 7. FILTERS
# =========================================================================
st.subheader("🔍 Analysis Controls")
flight_duration = float(timestamps_sec[-1])

col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    class_options = [
        f"{FAULT_COLORS[i]['emoji']} {FAULT_COLORS[i]['name']}"
        for i in range(config.NUM_CLASSES)
    ]
    selected_classes_str = st.multiselect(
        "🏷️ Fault Types", options=class_options, default=class_options
    )
    selected_class_ids = [
        i for i, name in enumerate(class_options) if name in selected_classes_str
    ]

with col_f2:
    max_time   = max(0.1, flight_duration)
    time_range = st.slider("⏱️ Time Range (s)", 0.0, max_time, (0.0, max_time), 0.1)

# Apply filters
mask = (
    pred_df["prediction"].isin(selected_class_ids) &
    (pred_df["End (s)"]   >= time_range[0]) &
    (pred_df["Start (s)"] <= time_range[1])
)
df_filtered = pred_df[mask].copy()

# =========================================================================
# 8. PREDICTION TABLE
# =========================================================================
st.subheader("📊 Window-Level Prediction Table")

if df_filtered.empty:
    st.warning("No predictions match the current filters.")
else:
    # Colour-map confidence values for visual feedback
    def style_confidence(val: str) -> str:
        pct = float(val.replace("%", ""))
        if pct >= 90:
            return "background-color: #dcfce7; color: #166534"    # green
        elif pct >= 70:
            return "background-color: #fef9c3; color: #713f12"    # yellow
        else:
            return "background-color: #fee2e2; color: #7f1d1d"    # red

    display_cols = ["Window #", "Start (s)", "End (s)", "Fault Class", "Confidence"]
    styled = (
        df_filtered[display_cols]
        .style
        .map(style_confidence, subset=["Confidence"])
        .set_properties(**{"text-align": "center"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    csv_bytes = df_filtered.drop(columns=["prediction", "confidence_raw"]).to_csv(
        index=False
    ).encode("utf-8-sig")
    st.download_button(
        "📥 Download Prediction Report (CSV)",
        data=csv_bytes,
        file_name="ai_fault_report.csv",
        mime="text/csv",
    )

# =========================================================================
# 9. RAW SIGNAL ANALYSIS (PX4 STYLE)
# =========================================================================
with st.expander("🔬 Raw Signal Analysis (PX4 Style - Spectral Analysis)", expanded=True):
    st.markdown("""
    This section visualizes the **vibration spectrum (FFT)** for the telemetry within the **selected time range**.
    """)

    # 1. Get temporal slice based on the slider above
    time_mask = (timestamps_sec >= time_range[0]) & (timestamps_sec <= time_range[1])
    sliced_df = original_df[time_mask]
    sliced_timestamps = timestamps_sec[time_mask]
    
    # 2. Get AI context for this slice
    # Filter predictions that fall within this time range
    slice_preds = df_filtered[(df_filtered["Start (s)"] >= time_range[0]) & (df_filtered["End (s)"] <= time_range[1])]
    
    if not slice_preds.empty:
        # Get dominant class and avg confidence for this slice
        main_class_id = slice_preds["prediction"].mode()[0]
        avg_conf = slice_preds["confidence_raw"].mean() * 100
        fault_info = FAULT_COLORS[main_class_id]
        
        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background-color:{fault_info['color']}; border:2px solid {fault_info['solid']}; margin-bottom:20px">
            <h3 style="margin:0; color:{fault_info['solid']}">{fault_info['emoji']} AI Diagnosis for this Slice: {fault_info['name']}</h3>
            <p style="margin:5px 0 0 0; font-weight:bold;">Average Confidence: {avg_conf:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No AI predictions available for this specific time slice.")

    # 3. Multi-Sensor FFT for the Slice
    raw_groups = get_sensor_groups(original_df, feature_cols)
    fft_group = st.selectbox("Select Sensor Group for FFT Analysis", options=list(raw_groups.keys()), key="fft_group_sel")
    
    fig_fft = go.Figure()
    fs = 100  # 100Hz
    
    # Calculate max amplitude to scale the fault rectangles
    max_amplitude = 0
    
    for s in raw_groups[fft_group]:
        if len(sliced_df) < 2:
            continue
            
        data = sliced_df[s].values
        data_clean = data - np.mean(data)
        
        n = len(data_clean)
        freq = np.fft.rfftfreq(n, d=1/fs)
        fft_values = np.abs(np.fft.rfft(data_clean))
        
        if len(fft_values) > 1:
            max_amplitude = max(max_amplitude, np.max(fft_values[1:]))
            
        fig_fft.add_trace(go.Scatter(
            x=freq[1:], y=fft_values[1:], 
            mode='lines', name=s.split('.')[-1]
        ))
    
    # 4. Overlay AI Fault Windows as background rectangles
    # Note: FFT x-axis is Frequency (Hz), NOT Time (s).
    # To overlay "windows", we must reconsider the logic. Since the user wants "windows over the fft signals",
    # and FFT x-axis is Hz, shading specific Hz ranges doesn't map directly to "Time Windows".
    # Instead, we will color the entire FFT background based on the dominant fault of the selected time slice
    # OR we can add a secondary temporal plot above it. 
    # Based on the prompt: "we can see the woindows over the fft signals... window will be relatied to the time range or the hz"
    # It seems the user wants the FFT background to reflect the fault. Let's color the background.
    
    if not slice_preds.empty and main_class_id != 0:
        fault_info = FAULT_COLORS[main_class_id]
        opacity = min(0.3, avg_conf / 100.0) # Scale opacity by confidence
        
        fig_fft.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=0, x1=1, y0=0, y1=1,
            fillcolor=fault_info['solid'],
            opacity=opacity,
            layer="below",
            line_width=0,
        )
        
        # Add a subtle annotation inside the chart
        fig_fft.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            text=f"Dominant Fault in this Time Slice: {fault_info['emoji']} {fault_info['name']} ({avg_conf:.1f}% Conf)",
            showarrow=False,
            font=dict(size=14, color=fault_info['solid']),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=fault_info['solid'],
            borderwidth=1,
            borderpad=4
        )

    fig_fft.update_layout(
        title=f"Spectral Analysis (FFT) for T={time_range[0]}s to T={time_range[1]}s",
        xaxis_title="Frequency (Hz)", yaxis_title="Amplitude",
        height=500, template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig_fft, use_container_width=True)


# =========================================================================
# 9. SENSOR VISUALIZATION  (Fix #4 – WebGL + normalized X-axis)
# =========================================================================
st.divider()
st.subheader("📈 Telemetry Signal Analysis")

if st.checkbox("🐞 Show Debug Info"):
    st.write(f"**Dataset shape:** {original_df.shape}")
    st.write(f"**Time range:** {timestamps_sec[0]:.3f} – {timestamps_sec[-1]:.3f} s")
    st.write(f"**Total windows:** {len(pred_df)}  |  Filtered: {len(df_filtered)}")
    st.write(f"**Feature columns ({len(feature_cols)}):** {feature_cols}")

    st.divider()
    st.info("Analysis and FFT visualization completed. Check the prediction table above for detailed results.")
