
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import base64

# --- Display Logo ---
logo = Image.open("ttu_logo.png")
st.image(logo, width=900)

# --- Header Info ---
st.markdown("""
<h2 style='text-align: center; color: navy;'>Graduation Project II</h2>
<h3 style='text-align: center; color: darkgreen;'>College of Engineering / Natural Resources and Chemical Engineering Department</h3>
<h4 style='text-align: center;'>Tafila Technical University</h4>
<h5 style='text-align: center; color: gray;'>Designed and implemented by students:</h5>
<ul style='text-align: center; list-style: none; padding-left: 0;'>
    <li>1 - Duaa</li>
    <li>2 - Shahed</li>
    <li>3 - Rahaf</li>
</ul>
""", unsafe_allow_html=True)

# --- Load ML Model and Scalers ---
scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
model = load_model('ann_water_model.h5')

output_vars = [
    'Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L',
    'Zn_final_mg_L', 'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
    'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%','Coagulant_dose_mg_L',
    'Flocculant_dose_mg_L', 'Mixing_speed_rpm',
    'Rapid_mix_time_min', 'Slow_mix_time_min', 'Settling_time_min'
]

limits = {
    'Turbidity_final_NTU': (0.0, 5.0),
    'Fe_final_mg_L': (0.0, 0.3),
    'Mn_final_mg_L': (0.0, 0.1),
    'Cu_final_mg_L': (0.0, 1.0),
    'Zn_final_mg_L': (0.0, 5.0),
    'Suspended_solids_final_mg_L': (0.0, 50.0),
    'TDS_final_mg_L': (0.0, 1000.0)
}

# --- Title & Form ---
st.title("Water Treatment Quality Predictor (ANN-based)")
st.markdown("Enter experimental values below to predict treated water quality and assess reuse suitability.")

with st.form("input_form"):
    pH_raw = st.slider("pH of Raw Water", 3.0, 11.0, 7.0)
    turbidity_raw = st.slider("Turbidity (NTU)", 0.1, 500.0, 50.0)
    temperature = st.slider("Temperature (°C)", 5.0, 40.0, 25.0)
    fe_initial = st.slider("Initial Fe (mg/L)", 0.0, 10.0, 1.0)
    mn_initial = st.slider("Initial Mn (mg/L)", 0.0, 5.0, 0.3)
    cu_initial = st.slider("Initial Cu (mg/L)", 0.0, 2.0, 0.05)
    zn_initial = st.slider("Initial Zn (mg/L)", 0.0, 5.0, 0.1)
    ss = st.slider("Suspended Solids (mg/L)", 0.0, 1000.0, 150.0)
    tds = st.slider("TDS (mg/L)", 0.0, 5000.0, 1000.0)

    submitted = st.form_submit_button("Test Water Quality")

# --- Prediction Logic ---
if submitted:
    input_array = np.array([[pH_raw, turbidity_raw, temperature, coagulant_dose, flocculant_dose,
                             fe_initial, mn_initial, cu_initial, zn_initial, ss, tds,
                             mixing_speed, rapid_mix, slow_mix, settling_time]])

    X_scaled = scaler_X.transform(input_array)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

    # --- Build Result Table ---
    results = pd.DataFrame(columns=["Parameter", "Predicted Value", "Standard Limit", "Unit", "Assessment"])

    units = ['NTU', 'mg/L', 'mg/L', 'mg/L', 'mg/L', 'mg/L', 'mg/L', '%', '%', '%']
    reuse_safe = True

    for i, var in enumerate(output_vars):
        value = y_pred[i]
        if var in limits:
            std_text, limit_val = limits[var]
            assessment = "✅ OK" if value <= limit_val else "❌ Exceeds Limit"
            if value > limit_val:
                reuse_safe = False
        else:
            std_text = "--"
            assessment = "--"

        results.loc[i] = [var, round(value, 3), std_text, units[i], assessment]

    # --- Display Table ---
    st.subheader("Predicted Treated Water Quality")
    st.dataframe(results)

    # --- Decision Message ---
    st.subheader("Reuse Decision")
    if reuse_safe:
        st.success("✅ Water is safe for reuse or discharge.")
    else:
        st.error("❌ Water does NOT meet quality standards for reuse.")

    # --- PDF Report Generation ---
    def create_pdf(df, reuse_safe):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Water Quality Prediction Report", ln=True, align='C')

        pdf.ln(10)
        for index, row in df.iterrows():
            pdf.cell(200, 10, f"{row['Parameter']}: {row['Predicted Value']} {row['Unit']} (Standard: {row['Standard Limit']}) --> {row['Assessment']}", ln=True)

        pdf.ln(10)
        decision = "Water is safe for reuse or discharge." if reuse_safe else "Water does NOT meet reuse standards."
        pdf.multi_cell(0, 10, f"Final Decision:\n{decision}")

        return pdf

    # Generate and download PDF
    pdf = create_pdf(results, reuse_safe)
    pdf.output("report.pdf")

    with open("report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="water_quality_report.pdf">📥 Download Report as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
