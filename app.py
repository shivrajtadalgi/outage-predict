import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Predictive Enterprise Asset Intelligence", layout="wide")

# ==========================================================
# GLOBAL STYLING
# ==========================================================
st.markdown("""
<style>

.stApp {
    background-color:#0A1F44;
    color:white;
}

.main-header {
    background: linear-gradient(90deg, #001F3F, #004080, #0066CC);
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:26px;
    font-weight:700;
    color:white;
    margin-bottom:5px;
}

.sub-header {
    text-align:center;
    font-size:15px;
    color:#B3E5FC;
    margin-bottom:20px;
}

.kpi-banner {
    background: linear-gradient(90deg, #004080, #0066CC, #0099FF);
    padding:12px;
    border-radius:10px;
    margin-bottom:12px;
    text-align:center;
    font-size:18px;
    font-weight:700;
    color:white;
}

div[data-testid="stSlider"] {
    margin-bottom:4px;
}

div.stButton > button {
    background-color:#FF6F00 !important;
    color:white !important;
    border-radius:6px !important;
    height:2.4em;
    font-weight:bold;
    border:none;
}

.status-text {
    color:#FFFACD;
    font-weight:bold;
    font-size:15px;
    text-align:center;
}

.compact-box {
    background:white;
    padding:16px;
    border-radius:12px;
    display:inline-block;
    min-width:260px;
}

.severity-title {
    text-align:center;
    font-weight:700;
    font-size:16px;
    color:#FF6F00;
    margin-bottom:8px;
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
# HEADER
# ==========================================================
st.markdown("<div class='main-header'>Predictive Enterprise Asset Management Intelligence using AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Maritime | Offshore | Gas Pipeline Predictive Risk Dashboard</div>", unsafe_allow_html=True)

# ==========================================================
# KPI SECTION
# ==========================================================
st.markdown("<div class='kpi-banner'>Operational KPI Controls</div>", unsafe_allow_html=True)

kpi_colors = [
"#FF1744","#FF9100","#FFD600","#00E676","#00E5FF",
"#2979FF","#D500F9","#F50057","#FF3D00","#00BFA5",
"#64DD17","#FFAB00","#00B0FF","#FF4081","#7C4DFF"
]

kpi_names = [
"Pressure","Temperature","Flow Rate","Gas Leak","Valve Position",
"Vibration","Compressor RPM","Bearing Temp","Oil Pressure","Corrosion Rate",
"Inspection Risk","Energy Use","Voltage Fluctuation",
"Maintenance Overdue","Equipment Age"
]

ranges = [
(40.0,120.0,80.0),(-20.0,80.0,30.0),(100.0,1500.0,600.0),
(0.0,500.0,50.0),(0.0,100.0,50.0),
(0.0,12.0,3.0),(1500.0,5000.0,3000.0),(40.0,120.0,70.0),
(2.0,8.0,5.0),(0.0,2.5,0.5),
(0.0,100.0,20.0),(1000.0,10000.0,4000.0),
(0.0,15.0,3.0),(0,180,10),(1,30,10)
]

values = []
cols = st.columns(5)

for i, name in enumerate(kpi_names):
    col = cols[i % 5]
    min_v, max_v, def_v = ranges[i]
    col.markdown(
        f"<span style='color:{kpi_colors[i]}; font-weight:bold; font-size:13px'>{name}</span>",
        unsafe_allow_html=True
    )
    values.append(col.slider("", min_v, max_v, def_v))

predict_btn = st.button("Run AI Prediction")

# ==========================================================
# SINGLE PREDICTION
# ==========================================================
if predict_btn:

    status_placeholder = st.empty()
    status_placeholder.markdown(
        "<div class='status-text'>Smart AI predicting System Outage Severity and Type...</div>",
        unsafe_allow_html=True
    )

    time.sleep(6)

    status_placeholder.markdown(
        "<div class='status-text'>AI Smart Predictions</div>",
        unsafe_allow_html=True
    )

    maintenance_index = kpi_names.index("Maintenance Overdue")
    age_index = kpi_names.index("Equipment Age")
    values[maintenance_index] *= 2.3
    values[age_index] *= 2.0

    input_array = np.array([values])
    scaled_input = scaler.transform(input_array)

    risk_score = risk_model.predict(scaled_input)[0]
    severity_pred = severity_model.predict(scaled_input)[0]
    outage_pred = outage_model.predict(scaled_input)[0]

    severity_label = severity_encoder.inverse_transform([severity_pred])[0]
    outage_label = outage_encoder.inverse_transform([outage_pred])[0]

    # ✅ FIX: normalize case
    severity_upper = severity_label.upper()

    severity_colors = {
        "HIGH":"red",
        "MEDIUM":"orange",
        "LOW":"yellow",
        "NORMAL":"green"
    }

    sev_color = severity_colors.get(severity_upper,"white")

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("<div class='compact-box'>", unsafe_allow_html=True)
        st.markdown("<div class='severity-title'>Severity Score</div>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            gauge={
                'axis':{'range':[0,100]},
                'steps':[
                    {'range':[0,25],'color':'green'},
                    {'range':[25,50],'color':'yellow'},
                    {'range':[50,75],'color':'orange'},
                    {'range':[75,100],'color':'red'}
                ]
            }))
        fig.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<h3 style='color:{sev_color}'>Severity: {severity_label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:white'>Outage Type: {outage_label}</h3>", unsafe_allow_html=True)

# ==========================================================
# UPLOAD SECTION
# ==========================================================
st.markdown("---")
st.markdown("### Upload Incidents to Smart AI Prediction of System Outage")

uploaded_file = st.file_uploader("Upload CSV (10–15 rows)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    status_placeholder = st.empty()
    status_placeholder.markdown(
        "<div class='status-text'>Smart AI predicting System Outage Severity and Type...</div>",
        unsafe_allow_html=True
    )

    time.sleep(8)

    scaled = scaler.transform(df)
    df["Predicted_Risk_Score"] = risk_model.predict(scaled)
    df["Severity_Class"] = severity_encoder.inverse_transform(
        severity_model.predict(scaled))
    df["Outage_Type"] = outage_encoder.inverse_transform(
        outage_model.predict(scaled))

    # ✅ FIX color mapping for upload table
    def color_severity(val):
        val = str(val).upper()
        if val == "HIGH":
            return "background-color:white; color:red;"
        elif val == "MEDIUM":
            return "background-color:white; color:orange;"
        elif val == "LOW":
            return "background-color:white; color:yellow;"
        elif val == "NORMAL":
            return "background-color:white; color:green;"
        return ""

    styled_df = df.style \
        .applymap(color_severity, subset=["Severity_Class"]) \
        .set_properties(subset=["Predicted_Risk_Score","Severity_Class","Outage_Type"],
                        **{'background-color':'#E3F2FD'})

    status_placeholder.markdown(
        "<div class='status-text'>AI Smart Predictions</div>",
        unsafe_allow_html=True
    )

    st.dataframe(styled_df, use_container_width=True)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================
st.markdown("### Top Features Impacting System Outage")

importance = risk_model.feature_importances_
df_imp = pd.DataFrame({"Feature":kpi_names,"Importance":importance})
df_imp = df_imp.sort_values(by="Importance", ascending=False).head(8)

fig2 = px.bar(df_imp, x="Importance", y="Feature", orientation='h')
fig2.update_layout(height=300)
st.plotly_chart(fig2, use_container_width=True)
