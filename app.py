import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

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

.title-box {
background: linear-gradient(90deg,#004080,#0066CC,#0099FF);
padding:12px;
border-radius:10px;
text-align:center;
font-size:26px;
font-weight:700;
color:white;
margin-bottom:15px;
}

.upload-title {
background: linear-gradient(90deg,#004080,#0066CC,#0099FF);
padding:10px;
border-radius:8px;
text-align:center;
font-size:20px;
font-weight:700;
color:white;
margin-bottom:12px;
}

.sub-header {
text-align:center;
font-size:15px;
color:#B3E5FC;
margin-bottom:20px;
}

.kpi-banner {
background: linear-gradient(90deg,#004080,#0066CC,#0099FF);
padding:10px;
border-radius:8px;
text-align:center;
font-size:18px;
font-weight:700;
color:white;
margin-bottom:10px;
}

.status-text {
color:#FFFACD;
font-weight:bold;
text-align:center;
}

div.stButton > button {
background-color:#FF6F00 !important;
color:white !important;
font-weight:bold !important;
border-radius:8px !important;
}

.feature-title {
background:rgba(255,255,255,0.15);
padding:8px;
border-radius:6px;
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
st.markdown("<div class='title-box'>Interactive Asset Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Maritime | Offshore | Gas Pipeline Predictive Risk Dashboard</div>", unsafe_allow_html=True)

# ==========================================================
# KPI CONTROLS
# ==========================================================
st.markdown("<div class='kpi-banner'>Operational KPI Controls</div>", unsafe_allow_html=True)

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

values=[]
cols=st.columns(5)

for i,name in enumerate(kpi_names):
    col=cols[i%5]
    min_v,max_v,def_v=ranges[i]
    col.markdown(f"**{name}**")
    values.append(col.slider("",min_v,max_v,def_v))

predict_btn=st.button("Run AI Prediction")

# ==========================================================
# PREDICTION
# ==========================================================
if predict_btn:

    status=st.empty()
    status.markdown("<div class='status-text'>Smart AI predicting System Outage Severity and Type...</div>",unsafe_allow_html=True)

    time.sleep(6)

    status.markdown("<div class='status-text'>AI Smart Predictions</div>",unsafe_allow_html=True)

    maintenance_index=kpi_names.index("Maintenance Overdue")
    age_index=kpi_names.index("Equipment Age")

    values[maintenance_index]*=2.3
    values[age_index]*=2.0

    input_array=np.array([values])
    scaled=scaler.transform(input_array)

    risk_score=risk_model.predict(scaled)[0]
    severity_pred=severity_model.predict(scaled)[0]
    outage_pred=outage_model.predict(scaled)[0]

    severity_label=severity_encoder.inverse_transform([severity_pred])[0]
    outage_label=outage_encoder.inverse_transform([outage_pred])[0]

    severity_colors = {
        "HIGH":"red",
        "MEDIUM":"orange",
        "LOW":"#C9A000",
        "NORMAL":"green"
    }

    sev_color = severity_colors.get(severity_label.upper(),"white")

    col1,col2=st.columns([1,1])

# ==========================================================
# SEVERITY SPEEDOMETER
# ==========================================================
    with col1:

        st.markdown("### Severity Score")

        fig=go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        number={'font':{'size':30,'color': "white"}},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"#69A1F6"},
            'steps':[
                {'range':[0,25],'color':'green'},
                {'range':[25,50],'color':'yellow'},
                {'range':[50,75],'color':'orange'},
                {'range':[75,100],'color':'red'}
            ],
            'threshold':{
                'line':{'color':"#051C3D",'width':6},
                'thickness':0.50,
                'value':risk_score
            }
        }
        ))

        fig.update_layout(
        height=320,
        margin=dict(l=20,r=20,t=30,b=20),
        paper_bgcolor="#0A1F44",
        font={'color':"white"}
        )

        st.plotly_chart(fig,use_container_width=True)

# ==========================================================
# SEVERITY + OUTAGE
# ==========================================================
    with col2:

        st.markdown(f"<h3 style='color:{sev_color}'>Severity: {severity_label}</h3>",unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#66CFFF'>Outage Type: {outage_label}</h3>",unsafe_allow_html=True)

# ==========================================================
# UPLOAD SECTION
# ==========================================================
st.markdown("<br><br>",unsafe_allow_html=True)

st.markdown("<div class='upload-title'>Upload Incidents to Smart AI Prediction of System Outage</div>",unsafe_allow_html=True)

file=st.file_uploader("Upload CSV",type=["csv"])

if file:

    df=pd.read_csv(file)

    status=st.empty()
    status.markdown("<div class='status-text'>Smart AI predicting System Outage Severity and Type...</div>",unsafe_allow_html=True)

    time.sleep(8)

    scaled=scaler.transform(df)

    df["Predicted_Risk_Score"]=risk_model.predict(scaled)
    df["Severity_Class"]=severity_encoder.inverse_transform(severity_model.predict(scaled))
    df["Outage_Type"]=outage_encoder.inverse_transform(outage_model.predict(scaled))

    def color_severity(val):

        val=str(val).upper()

        if val=="HIGH":
            return "color:red"
        if val=="MEDIUM":
            return "color:orange"
        if val=="LOW":
            return "color:#C9A000"
        if val=="NORMAL":
            return "color:green"

        return ""

    styled=df.style.applymap(color_severity,subset=["Severity_Class"])

    styled=styled.set_table_styles([
        {'selector':'th','props':[('background-color','#E3F2FD'),('color','black')]}
    ])

    styled=styled.set_properties(
        subset=["Predicted_Risk_Score","Severity_Class","Outage_Type"],
        **{'background-color':'#E3F2FD'}
    )

    st.write(styled)

# ==========================================================
# DASHBOARD LINK
# ==========================================================
st.markdown("<br><br>",unsafe_allow_html=True)

st.markdown("<div class='upload-title'>Interactive Asset Intelligence Dashboard</div>",unsafe_allow_html=True)

st.markdown(
'<a href="https://app.powerbi.com/groups/me/reports/6a27f5e8-83e3-4fd4-854d-84a21f79764b/64c3e471dab511a4d704?experience=power-bi" target="_blank" style="color:red;font-weight:bold;font-size:18px;">Open Power BI Asset Intelligence Dashboard</a>',
unsafe_allow_html=True
)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================
st.markdown("<br><br><br>",unsafe_allow_html=True)

st.markdown('<div class="feature-title"><h3>Feature Intelligence Console</h3></div>',unsafe_allow_html=True)

importance=risk_model.feature_importances_

df_imp=pd.DataFrame({
"Feature":kpi_names,
"Importance":importance
}).sort_values(by="Importance",ascending=False).head(8)

colors=["#FF1744","#FF9100","#FFD600","#00E676","#00E5FF","#2979FF","#D500F9","#FF4081"]

fig2=px.bar(
df_imp,
x="Importance",
y="Feature",
orientation="h",
color="Feature",
color_discrete_sequence=colors
)

fig2.update_layout(height=320,showlegend=False)

st.plotly_chart(fig2,use_container_width=True)

