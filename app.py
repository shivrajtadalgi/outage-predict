import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==========================================================
st.set_page_config(
    page_title="Lloyd's Register AI Dashboard",
    layout="wide"
)

# ==========================================================
# DARK MARITIME THEME + WHITE TEXT
# ==========================================================
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0A1F44;
    color: white;
}

/* All Labels White */
label {
    color: white !important;
    font-weight: 500;
}

/* Slider Labels */
.stSlider label {
    color: white !important;
}

/* Metric Labels */
.stMetric label {
    color: white !important;
}

/* Metric Values */
[data-testid="stMetricValue"] {
    color: white !important;
}

[data-testid="stMetricLabel"] {
    color: white !important;
}

/* Remove extra spacing */
.block-container {
    padding-top: 2rem;
}

/* Button Styling */
div.stButton > button {
    background-color: #002B5B !important;
    color: white !important;
    border-radius: 8px !important;
    height: 3em;
    font-weight: bold;
    border: none;
    width: 100%;
}

div.stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}

div.stButton > button:active {
    background-color: #001B3A !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODELS
# ==========================================================
risk_model = joblib.load("risk_model.pkl")
severity_model = joblib.load("severity_model.pkl")
outage_model = joblib.load("outage_model.pkl")
scaler = joblib.load("scaler.pkl")
severity_encoder = joblib.load("severity_encoder.pkl")
outage_encoder = joblib.load("outage_encoder.pkl")

# ==========================================================
# HEADER SECTION
# ==========================================================
col_logo, col_title = st.columns([1,4])

with col_logo:
    st.image("lr_logo.png", width=120)

with col_title:
    st.title("AI-Based System Outage Intelligence")
    st.markdown("Maritime | Offshore | Gas Pipeline Predictive Risk Dashboard")

# ==========================================================
# KPI INPUT SECTION
# ==========================================================
st.markdown("## Operational KPI Controls")

cols = st.columns(3)

with cols[0]:
    pressure = st.slider("Pressure (bar)", 40.0, 120.0, 80.0)
    temperature = st.slider("Temperature (°C)", -20.0, 80.0, 30.0)
    flow_rate = st.slider("Flow Rate (m³/hr)", 100.0, 1500.0, 600.0)
    gas_leak = st.slider("Gas Leak (ppm)", 0.0, 500.0, 50.0)
    valve = st.slider("Valve Position (%)", 0.0, 100.0, 50.0)

with cols[1]:
    vibration = st.slider("Vibration (mm/s)", 0.0, 12.0, 3.0)
    rpm = st.slider("Compressor RPM", 1500.0, 5000.0, 3000.0)
    bearing_temp = st.slider("Bearing Temperature (°C)", 40.0, 120.0, 70.0)
    oil_pressure = st.slider("Lubrication Oil Pressure (bar)", 2.0, 8.0, 5.0)
    corrosion = st.slider("Corrosion Rate (mm/year)", 0.0, 2.5, 0.5)

with cols[2]:
    age = st.slider("Equipment Age (years)", 1, 30, 10)
    maintenance = st.slider("Maintenance Overdue (days)", 0, 180, 10)
    inspection = st.slider("Inspection Risk Score", 0.0, 100.0, 20.0)
    energy = st.slider("Energy Consumption (kWh)", 1000.0, 10000.0, 4000.0)
    voltage = st.slider("Voltage Fluctuation (%)", 0.0, 15.0, 3.0)

# ==========================================================
# CENTERED PREDICTION BUTTON
# ==========================================================
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict_btn = st.button("Run AI Prediction")

# ==========================================================
# PREDICTION LOGIC
# ==========================================================
if predict_btn:

    input_data = np.array([[pressure, temperature, flow_rate, gas_leak, valve,
                            vibration, rpm, bearing_temp, oil_pressure,
                            corrosion, age, maintenance, inspection,
                            energy, voltage]])

    scaled_input = scaler.transform(input_data)

    risk_score = risk_model.predict(scaled_input)[0]
    severity_pred = severity_model.predict(scaled_input)[0]
    outage_pred = outage_model.predict(scaled_input)[0]

    severity_label = severity_encoder.inverse_transform([severity_pred])[0]
    outage_label = outage_encoder.inverse_transform([outage_pred])[0]

    st.markdown("## Prediction Results")

    col1, col2, col3 = st.columns(3)

    # --------------------------
    # RISK GAUGE
    # --------------------------
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 25], 'color': "green"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ]
            }))
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # SEVERITY BOX
    # --------------------------
    with col2:
        if severity_label == "HIGH":
            bg = "#FF4B4B"
            icon = "🚨"
        elif severity_label == "Medium":
            bg = "#FFA500"
            icon = "⚠"
        elif severity_label == "Normal":
            bg = "#2E8B57"
            icon = "✅"
        else:
            bg = "#1E90FF"
            icon = "ℹ"

        st.markdown(f"""
        <div style="
            background-color:{bg};
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:20px;
            font-weight:bold;
            color:white;">
            {icon} SEVERITY: {severity_label}
        </div>
        """, unsafe_allow_html=True)

    # --------------------------
    # OUTAGE TYPE BOX
    # --------------------------
    with col3:
        st.markdown(f"""
        <div style="
            background-color:#001F3F;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:18px;
            font-weight:bold;
            color:white;">
            Predicted Outage Type<br><br>
            {outage_label}
        </div>
        """, unsafe_allow_html=True)

    st.success("AI Prediction Completed Successfully")
