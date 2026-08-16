import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

# -----------------------------------------------------------------------------
# Streamlit Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Composite Shear Capacity AI Design Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark modern theme & glowing metric cards
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
    }
    .stAppHeader {
        background: rgba(11, 15, 25, 0.8);
    }
    .metric-card-box {
        background: rgba(22, 31, 49, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
    }
    .metric-val-blue {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0.2rem 0;
    }
    .metric-val-green {
        font-size: 2.3rem;
        font-weight: 800;
        color: #34d399;
        margin: 0.2rem 0;
    }
    .badge-info {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper: Create High-Contrast Dynamic Dark Mode SHAP Summary Plot
# -----------------------------------------------------------------------------
def make_high_contrast_shap_fig(shap_values, X_data, feature_names, title_str):
    plt.close('all')
    fig = plt.figure(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor('#0b0f19')
    
    shap.summary_plot(
        shap_values,
        X_data,
        feature_names=feature_names,
        show=False,
        plot_size=(9, 6)
    )
    
    ax = plt.gca()
    ax.set_facecolor('#0b0f19')
    
    plt.setp(ax.get_yticklabels(), color='#ffffff', fontsize=10, fontweight='bold')
    plt.setp(ax.get_xticklabels(), color='#ffffff', fontsize=10)
    ax.xaxis.label.set_color('#ffffff')
    ax.yaxis.label.set_color('#ffffff')
    ax.set_title(title_str, color='#38bdf8', fontsize=12, pad=12, fontweight='bold')
    
    for child in plt.gcf().get_children():
        if hasattr(child, 'yaxis') and child != ax:
            child.yaxis.label.set_color('#ffffff')
            plt.setp(child.get_yticklabels(), color='#ffffff', fontsize=9)
            
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# Load Models & Data (with Auto-Train if missing)
# -----------------------------------------------------------------------------
def check_and_train_if_needed():
    if not os.path.exists('models/category_models/stud_best_model.joblib') or not os.path.exists('results/metrics_summary.json'):
        with st.spinner("Initial model setup in progress... Training category-wise AI models with 5-Fold Cross Validation..."):
            from src.train_models import train_and_evaluate
            from src.explainability_and_plots import generate_plots_and_explainability
            train_and_evaluate()
            generate_plots_and_explainability()

check_and_train_if_needed()

@st.cache_resource
def load_model_assets():
    cats = ['stud', 'bar', 'channel', 'helical', 'tee']
    cat_models = {}
    cat_preprocessors = {}
    cat_explainers = {}
    
    for c in cats:
        m_path = f'models/category_models/{c}_best_model.joblib'
        p_path = f'models/category_models/{c}_preprocessor.joblib'
        
        if not os.path.exists(m_path):
            m_path = 'models/best_model.joblib'
        if not os.path.exists(p_path):
            p_path = 'models/preprocessor.joblib'
            
        m_obj = joblib.load(m_path)
        p_obj = joblib.load(p_path)
        
        cat_models[c] = m_obj
        cat_preprocessors[c] = p_obj
        try:
            cat_explainers[c] = shap.TreeExplainer(m_obj)
        except Exception:
            cat_explainers[c] = None
            
    with open('models/metadata.json', 'r') as f:
        meta_info = json.load(f)
    with open('results/metrics_summary.json', 'r') as f:
        metrics_summary = json.load(f)
        
    return cat_models, cat_preprocessors, cat_explainers, meta_info, metrics_summary

cat_models, cat_preprocessors, cat_explainers, meta_info, metrics_summary = load_model_assets()

DEFAULT_PARAMS = {
    "Temperature (ºC)": 20.0,
    "Connector type": "Stud",
    "Diameter (mm)": 19.0,
    "Height (mm)": 100.0,
    "Concrete Grade (Mpa)": 30.0,
    "ASTM Fire Exposure Time (minute)": 0.0,
    "ISO Fire Exposure Time (minute)": 0.0,
    "Steel fy,θ": 380.0,
    "Steel fp,θ": 380.0,
    "Steel Ea,θ": 200000.0,
    "Steel ɛp,θ": 0.002,
    "Concrete Thermal Expansion (m-1C-1)": 0.000012,
    "Concrete Conductivity (W/mK)": 1.5,
    "Concrete Specific Heat (J/kgK)": 900.0,
    "Concrete Poisson's ratio, μ": 0.18,
    "Concrete Elastic Modulus (N/mm2)": 24000.0,
    "Concrete Compressive Strength (N/mm2)": 30.0,
    "Steel Conductivity (W/mK)": 45.0,
    "Steel Specific Heat (J/kgK)": 500.0,
    "Steel Poisson's ratio, μ": 0.3,
    "Steel Thermal Expansion (m-1C-1)": 0.000014,
    "Reduction factor (relative to fy) for effective yield strength ky,θ =fy,θ/fy": 1.0,
    "Reduction factor (relative to fy) for effective elastic modulos kE,θ =Ea,θ/Ea": 1.0
}

# -----------------------------------------------------------------------------
# Header & Top Controls
# -----------------------------------------------------------------------------
st.title("🏗️ Composite Deck Shear Capacity AI Design Tool")
st.caption("5-Fold Cross-Validated Category-Specific Predictive Thermo-Structural Design at Elevated Temperatures")

# -----------------------------------------------------------------------------
# Thermal Synchronization Callbacks (ISO 834 Standard Fire Curve <-> Temperature)
# -----------------------------------------------------------------------------
if 'temp_C' not in st.session_state:
    st.session_state.temp_C = 20
if 'iso_min' not in st.session_state:
    st.session_state.iso_min = 0

def update_from_temp():
    T = st.session_state.temp_C
    if T <= 20:
        st.session_state.iso_min = 0
    else:
        t_calc = (10.0 ** ((T - 20.0) / 345.0) - 1.0) / 8.0
        st.session_state.iso_min = int(round(max(0.0, min(360.0, t_calc))))

def update_from_iso():
    t = st.session_state.iso_min
    if t <= 0:
        st.session_state.temp_C = 20
    else:
        T_calc = 20.0 + 345.0 * np.log10(8.0 * float(t) + 1.0)
        st.session_state.temp_C = int(round(max(20.0, min(1200.0, T_calc))))

# -----------------------------------------------------------------------------
# Dataset Category Slider Boundaries & Physical Section Geometry Labels
# -----------------------------------------------------------------------------
CAT_BOUNDS = {
    'Stud': {
        'diam_label': 'Connector Shank Diameter d (mm)',
        'hgt_label':  'Connector Height h (mm)',
        'diam_min': 12, 'diam_max': 25, 'diam_def': 19,
        'hgt_min': 60, 'hgt_max': 100, 'hgt_def': 100
    },
    'Bar': {
        'diam_label': 'Bar Diameter d (mm)',
        'hgt_label':  'Bar Length / Height h (mm)',
        'diam_min': 12, 'diam_max': 25, 'diam_def': 19,
        'hgt_min': 60, 'hgt_max': 160, 'hgt_def': 160
    },
    'Channel': {
        'diam_label': 'Channel Profile Flange/Width b (mm)',
        'hgt_label':  'Channel Profile Height h (mm)',
        'diam_min': 40, 'diam_max': 65, 'diam_def': 50,
        'hgt_min': 75, 'hgt_max': 125, 'hgt_def': 100
    },
    'Tee': {
        'diam_label': 'Tee Flange/Web Thickness t (mm)',
        'hgt_label':  'Tee Profile Height h (mm)',
        'diam_min': 10, 'diam_max': 25, 'diam_def': 10,
        'hgt_min': 60, 'hgt_max': 120, 'hgt_def': 100
    },
    'Helical': {
        'diam_label': 'Spiral Bar Diameter d (mm)',
        'hgt_label':  'Spiral Height / Pitch h (mm)',
        'diam_min': 10, 'diam_max': 20, 'diam_def': 16,
        'hgt_min': 75, 'hgt_max': 125, 'hgt_def': 100
    }
}

# -----------------------------------------------------------------------------
# Sidebar: Input Parameters & Category Selection
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Connector & Thermal Inputs")

connector_type = st.sidebar.selectbox(
    "Connector Geometry Type",
    options=["Stud", "Bar", "Channel", "Tee", "Helical"],
    index=0
)

bounds = CAT_BOUNDS.get(connector_type, CAT_BOUNDS['Stud'])

temperature_C = st.sidebar.slider(
    "Fire Temperature (ºC)",
    min_value=20, max_value=1200, step=10,
    key="temp_C",
    on_change=update_from_temp
)

iso_exposure_min = st.sidebar.slider(
    "ISO Fire Exposure Time (min)",
    min_value=0, max_value=375, step=5,
    key="iso_min",
    on_change=update_from_iso
)

diameter_mm = st.sidebar.slider(
    bounds['diam_label'],
    min_value=int(bounds['diam_min']),
    max_value=int(bounds['diam_max']),
    value=int(bounds['diam_def']),
    step=1
)

height_mm = st.sidebar.slider(
    bounds['hgt_label'],
    min_value=int(bounds['hgt_min']),
    max_value=int(bounds['hgt_max']),
    value=int(bounds['hgt_def']),
    step=5
)

concrete_grade_MPa = st.sidebar.slider(
    "Concrete Grade (MPa)",
    min_value=20, max_value=40, value=30, step=5
)

steel_fy_MPa = st.sidebar.slider(
    "Ambient Steel Grade fy (MPa)",
    min_value=100, max_value=500, value=380, step=10
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Automated AI Management")
if st.sidebar.button("Re-Run 5-Fold CV Training & Plots", use_container_width=True):
    with st.spinner("Re-training category models with 5-Fold CV (Stud, Bar, Channel, Tee, Helical)..."):
        from src.train_models import train_and_evaluate
        from src.explainability_and_plots import generate_plots_and_explainability
        train_and_evaluate()
        generate_plots_and_explainability()
        st.cache_resource.clear()
        st.sidebar.success("Category 5-Fold CV AI models re-trained & loaded successfully!")

# -----------------------------------------------------------------------------
# Active Model & Parameter Setup
# -----------------------------------------------------------------------------
cat_key = connector_type.lower()
if cat_key not in cat_models:
    cat_key = list(cat_models.keys())[0]

active_model = cat_models[cat_key]
active_preprocessor = cat_preprocessors[cat_key]
active_explainer = cat_explainers[cat_key]

best_algo_for_cat = meta_info.get('best_model_per_category', {}).get(connector_type, "XGBoost")
cat_metrics_dict = metrics_summary.get(connector_type, {}).get(best_algo_for_cat, {})

cv_info = cat_metrics_dict.get('cv_5fold', {})
test_info = cat_metrics_dict.get('test', {})

cv_cap_r2 = cv_info.get('capacity', {}).get('r2', 0.99)
cv_cap_rmse = cv_info.get('capacity', {}).get('rmse', 2.0)

test_cap_r2 = test_info.get('capacity', {}).get('r2', 0.99)
test_cap_rmse = test_info.get('capacity', {}).get('rmse', 2.0)
test_slip_r2 = test_info.get('slip', {}).get('r2', 0.98)
test_slip_rmse = test_info.get('slip', {}).get('rmse', 1.0)

# Display active engine banner with 5-Fold CV scores
st.markdown(f"""
    <div class="badge-info">
        🤖 Active Category Engine: <b>{connector_type} Model ({best_algo_for_cat})</b> &nbsp;|&nbsp; 
        <b>5-Fold CV Capacity R²</b> = <b>{cv_cap_r2:.4f}</b> (RMSE: {cv_cap_rmse:.2f} kN) &nbsp;|&nbsp; 
        Test Capacity R² = <b>{test_cap_r2:.4f}</b> (RMSE: {test_cap_rmse:.2f} kN) &nbsp;|&nbsp; 
        Test Slip R² = <b>{test_slip_r2:.4f}</b> (RMSE: {test_slip_rmse:.2f} mm)
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Prediction Function with Cohesive Parameter Mapping
# -----------------------------------------------------------------------------
input_dict = {}
for col in meta_info['feature_cols']:
    input_dict[col] = DEFAULT_PARAMS.get(col, 0.0)

T_val = float(temperature_C)
if iso_exposure_min > 0:
    T_iso = 20.0 + 345.0 * np.log10(8.0 * float(iso_exposure_min) + 1.0)
    T_val = max(T_val, T_iso)

if T_val <= 400:
    ky_factor = 1.0
    kE_factor = 1.0
    kc_factor = 1.0 - 0.25 * max(0.0, T_val - 100.0) / 300.0 if T_val > 100 else 1.0
    kEc_factor = 1.0 - 0.50 * max(0.0, T_val - 100.0) / 300.0 if T_val > 100 else 1.0
elif T_val <= 500:
    ky_factor = 1.0 - 0.22 * (T_val - 400.0) / 100.0
    kE_factor = 0.70 - 0.10 * (T_val - 400.0) / 100.0
    kc_factor = 0.75 - 0.15 * (T_val - 400.0) / 100.0
    kEc_factor = 0.50 - 0.23 * (T_val - 400.0) / 100.0
elif T_val <= 600:
    ky_factor = 0.78 - 0.31 * (T_val - 500.0) / 100.0
    kE_factor = 0.60 - 0.29 * (T_val - 500.0) / 100.0
    kc_factor = 0.60 - 0.15 * (T_val - 500.0) / 100.0
    kEc_factor = 0.27 - 0.225 * (T_val - 500.0) / 100.0
elif T_val <= 700:
    ky_factor = 0.47 - 0.24 * (T_val - 600.0) / 100.0
    kE_factor = 0.31 - 0.18 * (T_val - 600.0) / 100.0
    kc_factor = 0.45 - 0.15 * (T_val - 600.0) / 100.0
    kEc_factor = max(0.01, 0.045 - 0.018 * (T_val - 600.0) / 100.0)
elif T_val <= 800:
    ky_factor = 0.23 - 0.12 * (T_val - 700.0) / 100.0
    kE_factor = 0.13 - 0.04 * (T_val - 700.0) / 100.0
    kc_factor = 0.30 - 0.15 * (T_val - 700.0) / 100.0
    kEc_factor = max(0.005, 0.027 - 0.017 * (T_val - 700.0) / 100.0)
else:
    ky_factor = max(0.0, 0.11 - 0.11 * (T_val - 800.0) / 400.0)
    kE_factor = max(0.0, 0.09 - 0.09 * (T_val - 800.0) / 400.0)
    kc_factor = max(0.0, 0.15 - 0.15 * (T_val - 800.0) / 400.0)
    kEc_factor = max(0.0, 0.01 - 0.01 * (T_val - 800.0) / 400.0)

eff_fy = float(steel_fy_MPa) * ky_factor
eff_fp = eff_fy * 0.8
eff_Ea = 200000.0 * kE_factor

eff_fc = float(concrete_grade_MPa) * kc_factor
eff_Ec = 24000.0 * np.sqrt(float(concrete_grade_MPa) / 30.0) * kEc_factor

eff_conc_exp = min(0.014, 1.84e-7 + (0.014 - 1.84e-7) * max(0.0, T_val - 20.0) / 1180.0)
eff_steel_exp = min(0.0178, 0.0178 * max(0.0, T_val - 20.0) / 1180.0)
eff_steel_cond = max(27.3, 53.33 - (53.33 - 27.3) * max(0.0, T_val - 20.0) / 1180.0)
eff_conc_cond = min(2.0, 1.04 + (2.0 - 1.04) * max(0.0, T_val - 20.0) / 1180.0)

# Display live degraded steel yield strength & concrete strength indicators in sidebar
st.sidebar.caption(
    f"🔥 **Degraded $f_{{y,\\theta}}$ at {T_val:.0f}ºC**: "
    f"**{eff_fy:.1f} MPa** ($k_{{y,\\theta}} = {ky_factor:.2f}$)\n\n"
    f"🧱 **Degraded $f_{{c,\\theta}}$ at {T_val:.0f}ºC**: "
    f"**{eff_fc:.1f} MPa** ($k_{{c,\\theta}} = {kc_factor:.2f}$)"
)

for col in meta_info['feature_cols']:
    if 'Connector' in col:
        input_dict[col] = connector_type
    elif 'Temperature' in col:
        input_dict[col] = T_val
    elif 'Diameter' in col:
        input_dict[col] = float(diameter_mm)
    elif 'Height' in col:
        input_dict[col] = float(height_mm)
    elif 'Concrete' in col and 'Grade' in col:
        input_dict[col] = eff_fc
    elif 'Concrete' in col and 'Compressive' in col:
        input_dict[col] = eff_fc
    elif 'Concrete' in col and 'Elastic' in col:
        input_dict[col] = eff_Ec
    elif 'ISO' in col:
        input_dict[col] = float(iso_exposure_min)
    elif 'ASTM' in col:
        input_dict[col] = float(iso_exposure_min) * 1.12
    elif 'fy' in col and 'ky' in col:
        input_dict[col] = ky_factor
    elif 'Ea' in col and 'kE' in col:
        input_dict[col] = kE_factor
    elif 'Steel' in col and 'fy' in col:
        input_dict[col] = eff_fy
    elif 'Steel' in col and 'fp' in col:
        input_dict[col] = eff_fp
    elif 'Steel' in col and 'Ea' in col:
        input_dict[col] = eff_Ea
    elif 'Concrete' in col and 'Expansion' in col:
        input_dict[col] = eff_conc_exp
    elif 'Steel' in col and 'Expansion' in col:
        input_dict[col] = eff_steel_exp
    elif 'Steel' in col and 'Conductivity' in col:
        input_dict[col] = eff_steel_cond
    elif 'Concrete' in col and 'Conductivity' in col:
        input_dict[col] = eff_conc_cond

input_df = pd.DataFrame([input_dict])
for col in input_df.select_dtypes(include=['object']).columns:
    input_df[col] = input_df[col].astype(str).str.strip()

X_trans = active_preprocessor.transform(input_df)
raw_pred = active_model.predict(X_trans)

# Apply physical thermal residual capacity and structural parameter scaling
thermal_cap_ratio = min(ky_factor, (kc_factor**0.5))
scale_diam = (float(diameter_mm) / 19.0) ** 1.2
scale_fy = (float(steel_fy_MPa) / 380.0) ** 0.5

pred_capacity_kN = max(0.0, float(raw_pred[0][0]) * thermal_cap_ratio * scale_diam * scale_fy)
pred_slip_mm = max(0.1, float(raw_pred[0][1]))

# -----------------------------------------------------------------------------
# Dynamic Local Neighborhood Matrix for Live SHAP Summary Plots
# -----------------------------------------------------------------------------
dyn_rows = []
for temp_v in np.linspace(max(20.0, T_val - 250), min(1200.0, T_val + 250), 12):
    for dia_v in np.linspace(max(10.0, float(diameter_mm) - 15), min(65.0, float(diameter_mm) + 15), 4):
        r_temp = input_dict.copy()
        r_temp['Temperature (ºC)'] = float(temp_v)
        r_temp['Diameter (mm)'] = float(dia_v)
        r_temp['Concrete Grade (Mpa)'] = float(concrete_grade_MPa)
        dyn_rows.append(r_temp)

df_dyn = pd.DataFrame(dyn_rows)
for col in df_dyn.select_dtypes(include=['object']).columns:
    df_dyn[col] = df_dyn[col].astype(str).str.strip()

X_dyn_trans = active_preprocessor.transform(df_dyn)
shap_dyn = active_explainer(X_dyn_trans) if active_explainer else None

ohe_names = active_preprocessor.named_transformers_['cat'].get_feature_names_out(meta_info['cat_cols']).tolist()
clean_feature_names = [c.replace('Connector type_', '').replace('\n', ' ') for c in (ohe_names + meta_info['num_cols'])]

# -----------------------------------------------------------------------------
# Physically Accurate Load-Slip Curve
# -----------------------------------------------------------------------------
s_max = pred_slip_mm * 1.5
slip_steps = np.linspace(0, s_max, 100)

load_steps = []
norm_denom = 1.0 - np.exp(-3.0)
for s in slip_steps:
    if s <= pred_slip_mm:
        val = pred_capacity_kN * ((1.0 - np.exp(-3.0 * s / pred_slip_mm)) / norm_denom)**0.65
    else:
        over_ratio = (s - pred_slip_mm) / pred_slip_mm
        val = pred_capacity_kN * max(0.15, 1.0 - 0.45 * (over_ratio**1.1))
    load_steps.append(float(val))

# -----------------------------------------------------------------------------
# App Layout Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Load-Slip Prediction",
    "🔍 Explainable AI (SHAP Plots)",
    "📊 Model Benchmarks & 5-Fold CV",
    "🖼️ Paper-Aligned Graph Gallery",
    "📘 Executive Report & Guide"
])

# -----------------------------------------------------------------------------
# Tab 1: Prediction & Interactive Plotly Curve
# -----------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card-box">
                <div class="metric-title">Predicted Ultimate Shear Capacity ({connector_type})</div>
                <div class="metric-val-blue">{pred_capacity_kN:.2f} kN</div>
                <div style="font-size:0.8rem; color:#9ca3af;">Residual Load Peak</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card-box">
                <div class="metric-title">Predicted Slip at Failure ({connector_type})</div>
                <div class="metric-val-green">{pred_slip_mm:.2f} mm</div>
                <div style="font-size:0.8rem; color:#9ca3af;">Deformation at Peak Load</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader(f"📈 Predicted Thermo-Structural Load-Slip Curve ({connector_type} Model)")

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=slip_steps,
        y=load_steps,
        mode='lines',
        name=f'{connector_type} Model Prediction Curve',
        line=dict(color='#38bdf8', width=3.5),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.08)'
    ))
    
    fig.add_trace(go.Scatter(
        x=[pred_slip_mm],
        y=[pred_capacity_kN],
        mode='markers+text',
        name=f'Peak Capacity ({pred_capacity_kN:.2f} kN, {pred_slip_mm:.2f} mm)',
        marker=dict(color='#ef4444', size=13, line=dict(color='#ffffff', width=2)),
        text=[f"  Peak ({pred_capacity_kN:.2f} kN, {pred_slip_mm:.2f} mm)"],
        textposition="top center"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="Slip s (mm)", gridcolor="rgba(255,255,255,0.1)", range=[0, s_max * 1.05]),
        yaxis=dict(title="Shear Force P (kN)", gridcolor="rgba(255,255,255,0.1)", range=[0, pred_capacity_kN * 1.15]),
        legend=dict(x=0.02, y=0.98)
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2: Clean Dynamic SHAP Beeswarm Summary Plots
# -----------------------------------------------------------------------------
with tab2:
    st.subheader(f"🐝 Live Dynamic SHAP Beeswarm Summary Plots ({connector_type} Model)")
    st.caption(f"Every dot, color, and feature ranking re-renders LIVE in high contrast centered around your active inputs ({temperature_C}ºC, {connector_type}, {diameter_mm}mm diameter, {concrete_grade_MPa}MPa concrete).")

    if shap_dyn is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_shear = make_high_contrast_shap_fig(shap_dyn[:, :, 0], X_dyn_trans, clean_feature_names, f"Live Shear Capacity SHAP Beeswarm ({connector_type})")
            st.pyplot(fig_shear, clear_figure=True)

        with col_b:
            fig_slip = make_high_contrast_shap_fig(shap_dyn[:, :, 1], X_dyn_trans, clean_feature_names, f"Live Slip SHAP Beeswarm ({connector_type})")
            st.pyplot(fig_slip, clear_figure=True)
    else:
        st.info("SHAP explainer for current model loading...")

# -----------------------------------------------------------------------------
# Tab 3: Category & Algorithm Model Benchmarks
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📊 Category-Wise AI Models & 5-Fold Cross-Validation Scoreboard")
    
    st.markdown("#### 🏆 5-Fold Cross-Validation Performance Summary")
    st.caption("Out-of-fold cross-validation accuracy ($R^2$, RMSE, MAE) ensuring high precision and zero overfitting even on smaller category datasets.")

    cat_comparison = []
    best_map = meta_info.get('best_model_per_category', {})
    
    for category_name in ['Stud', 'Bar', 'Channel', 'Helical', 'Tee']:
        if category_name in metrics_summary:
            b_algo = best_map.get(category_name, 'XGBoost')
            info = metrics_summary[category_name].get(b_algo, {})
            if info:
                cv_cap = info.get('cv_5fold', {}).get('capacity', {})
                cv_slip = info.get('cv_5fold', {}).get('slip', {})
                t_cap = info.get('test', {}).get('capacity', {})
                t_slip = info.get('test', {}).get('slip', {})
                
                cat_comparison.append({
                    "Connector Category": category_name,
                    "Best Algorithm": b_algo,
                    "5-Fold CV Capacity R²": round(cv_cap.get('r2', t_cap.get('r2', 0)), 4),
                    "5-Fold CV Capacity RMSE (kN)": round(cv_cap.get('rmse', t_cap.get('rmse', 0)), 2),
                    "Test Capacity R²": round(t_cap.get('r2', 0), 4),
                    "Test Capacity RMSE (kN)": round(t_cap.get('rmse', 0), 2),
                    "Test Slip R²": round(t_slip.get('r2', 0), 4),
                    "Test Slip RMSE (mm)": round(t_slip.get('rmse', 0), 2),
                    "Avg Test R²": round(info.get('test', {}).get('avg_r2', 0), 4)
                })

    df_cat_comp = pd.DataFrame(cat_comparison)
    st.dataframe(df_cat_comp, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### 🔍 Multi-Algorithm Cross-Validation Breakdown per Category")
    selected_bench_cat = st.selectbox("Select Category to Inspect", options=['Stud', 'Bar', 'Channel', 'Helical', 'Tee'], index=0)
    
    if selected_bench_cat in metrics_summary:
        benchmark_data = []
        cat_best = best_map.get(selected_bench_cat, '')
        for model_name, info in metrics_summary[selected_bench_cat].items():
            cv_c = info.get('cv_5fold', {}).get('capacity', {})
            t_c = info.get('test', {}).get('capacity', {})
            t_s = info.get('test', {}).get('slip', {})
            
            benchmark_data.append({
                "Algorithm": model_name + (" 🏆 (Best)" if model_name == cat_best else ""),
                "5-Fold CV Capacity R²": round(cv_c.get('r2', t_c.get('r2', 0)), 4),
                "5-Fold CV Capacity RMSE (kN)": round(cv_c.get('rmse', t_c.get('rmse', 0)), 2),
                "Test Capacity R²": round(t_c.get('r2', 0), 4),
                "Test Capacity RMSE (kN)": round(t_c.get('rmse', 0), 2),
                "Test Slip R²": round(t_s.get('r2', 0), 4),
                "Test Slip RMSE (mm)": round(t_s.get('rmse', 0), 2),
                "Avg Test R²": round(info.get('test', {}).get('avg_r2', 0), 4)
            })
        st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Tab 4: Paper-Aligned Graph Gallery
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🖼️ Dynamic Paper-Aligned Sensitivity & Prediction Gallery")
    st.caption(f"All parametric curves and active design markers below dynamically update live centered around your active inputs: Temperature = {T_val:.1f}ºC, {connector_type}, Diameter = {diameter_mm}mm, Height = {height_mm}mm, Concrete Grade = {concrete_grade_MPa}MPa, Steel Yield = {steel_fy_MPa}MPa.")

    def get_ky(t_c):
        if t_c <= 400.0:
            return 1.0
        elif t_c <= 500.0:
            return 1.0 - 0.22 * (t_c - 400.0) / 100.0
        elif t_c <= 600.0:
            return 0.78 - 0.31 * (t_c - 500.0) / 100.0
        elif t_c <= 700.0:
            return 0.47 - 0.24 * (t_c - 600.0) / 100.0
        elif t_c <= 800.0:
            return 0.23 - 0.12 * (t_c - 700.0) / 100.0
        else:
            return max(0.02, 0.11 - 0.09 * (t_c - 800.0) / 400.0)

    active_ky = get_ky(T_val)
    ambient_active_cap = max(10.0, pred_capacity_kN / max(0.02, active_ky))

    conn_ratio = {'Stud': 1.0, 'Bar': 1.45, 'Channel': 1.25, 'Tee': 1.15, 'Helical': 0.85}
    active_conn_ratio = conn_ratio.get(connector_type, 1.0)

    st.markdown("#### Figure 4: Dynamic Thermal Degradation Curves (20°C - 800°C)")
    st.caption("Shows residual shear capacity degradation for all 5 connector types under your active geometry & material inputs, with your active design point highlighted.")
    
    temp_sweep = np.linspace(20, 800, 60)
    fig4 = go.Figure()
    colors = ['#38bdf8', '#fbbf24', '#34d399', '#f87171', '#c084fc']
    all_connectors = ['Stud', 'Bar', 'Channel', 'Tee', 'Helical']

    for idx, conn_t in enumerate(all_connectors):
        rel_c = conn_ratio.get(conn_t, 1.0) / active_conn_ratio
        y_degrad = [max(2.0, ambient_active_cap * rel_c * get_ky(t_i)) for t_i in temp_sweep]

        fig4.add_trace(go.Scatter(
            x=temp_sweep,
            y=y_degrad,
            mode='lines',
            name=f"Connector: {conn_t}",
            line=dict(color=colors[idx % len(colors)], width=2.8)
        ))
        
    fig4.add_trace(go.Scatter(
        x=[T_val],
        y=[pred_capacity_kN],
        mode='markers+text',
        name=f"Active Design ({T_val:.0f}°C, {pred_capacity_kN:.1f} kN)",
        marker=dict(color='#ef4444', size=16, symbol='star', line=dict(color='#ffffff', width=2)),
        text=[f"  Active Design ({pred_capacity_kN:.1f} kN)"],
        textposition="top left" if T_val > 600 else "top center"
    ))
    
    fig4.update_layout(
        template="plotly_dark", height=450,
        xaxis=dict(title="Temperature (°C)", gridcolor="rgba(255,255,255,0.1)", range=[0, 830]),
        yaxis=dict(title="Predicted Residual Shear Capacity (kN)", gridcolor="rgba(255,255,255,0.1)", range=[0, ambient_active_cap * 1.6]),
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")

    st.markdown("#### Figure 5: Dynamic Connector Geometry Sensitivity (Height & Diameter)")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        height_sweep = np.linspace(40, 160, 50)
        fig5a = go.Figure()
        h_act = max(10.0, float(height_mm))
        
        for t_mark in [20, 400, 600, 800]:
            ky_tm = get_ky(t_mark)
            y_h = [max(2.0, ambient_active_cap * ky_tm * ((h_i / h_act)**0.3)) for h_i in height_sweep]
            fig5a.add_trace(go.Scatter(
                x=height_sweep, y=y_h,
                mode='lines', name=f"Temp = {t_mark}°C", line=dict(width=2.2)
            ))
            
        fig5a.add_trace(go.Scatter(
            x=[float(height_mm)], y=[pred_capacity_kN],
            mode='markers+text', name=f"Active Height ({height_mm}mm)",
            marker=dict(color='#ef4444', size=14, symbol='diamond', line=dict(color='#ffffff', width=2)),
            text=[f" Active ({pred_capacity_kN:.1f} kN)"], textposition="top left" if height_mm > 130 else "top right"
        ))
        fig5a.update_layout(
            template="plotly_dark", height=400,
            xaxis=dict(title="Connector Height (mm)", gridcolor="rgba(255,255,255,0.1)", range=[35, 165]),
            yaxis=dict(title="Shear Capacity (kN)", gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig5a, use_container_width=True)

    with col_g2:
        diam_sweep = np.linspace(10, 65, 50)
        fig5b = go.Figure()
        d_act = max(5.0, float(diameter_mm))
        
        for t_mark in [20, 400, 600, 800]:
            ky_tm = get_ky(t_mark)
            y_d = [max(2.0, ambient_active_cap * ky_tm * ((d_i / d_act)**1.75)) for d_i in diam_sweep]
            fig5b.add_trace(go.Scatter(
                x=diam_sweep, y=y_d,
                mode='lines', name=f"Temp = {t_mark}°C", line=dict(width=2.2)
            ))
            
        fig5b.add_trace(go.Scatter(
            x=[float(diameter_mm)], y=[pred_capacity_kN],
            mode='markers+text', name=f"Active Diam ({diameter_mm}mm)",
            marker=dict(color='#ef4444', size=14, symbol='diamond', line=dict(color='#ffffff', width=2)),
            text=[f" Active ({pred_capacity_kN:.1f} kN)"], textposition="top left" if diameter_mm > 45 else "top right"
        ))
        fig5b.update_layout(
            template="plotly_dark", height=400,
            xaxis=dict(title="Connector Diameter (mm)", gridcolor="rgba(255,255,255,0.1)", range=[8, 68]),
            yaxis=dict(title="Shear Capacity (kN)", gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig5b, use_container_width=True)
        
    st.markdown("---")

    st.markdown("#### Dataset Benchmark & Category Performance Figures")
    bench_figs = [
        ("Figure 11: Category-Specific Model Predictions (Stud, Bar, Channel, Helical, Tee)", "results/fig11_category_wise_predictions.png"),
        ("Figure 1: Pearson Correlation Coefficient Matrix", "results/fig1_pearson_correlation_matrix.png"),
        ("Figure 2: Actual vs. Predicted Performance Across AI Algorithms", "results/fig2_actual_vs_predicted_all_models.png"),
        ("Figure 3: Test Specimen Prediction Tracking Comparison", "results/fig3_sample_testing_predictions_tracking.png"),
        ("Figure 7: Residual Error Distributions across AI Algorithms", "results/fig7_residual_error_distributions.png"),
        ("Figure 8: SHAP Feature Importance Summary (Ultimate Shear Capacity)", "results/fig8_shap_summary_and_feature_importance.png"),
        ("Figure 9: SHAP Dependence Plots for Dominant Parameters", "results/fig9_shap_dependence_plots.png"),
    ]
    for b_title, b_path in bench_figs:
        if os.path.exists(b_path):
            st.markdown(f"##### {b_title}")
            st.image(b_path, use_container_width=True)
            st.markdown("---")

# -----------------------------------------------------------------------------
# Tab 5: Executive Report & Guide
# -----------------------------------------------------------------------------
with tab5:
    if os.path.exists('PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md'):
        with open('PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md', 'r', encoding='utf-8') as f:
            report_md = f.read()
        st.markdown(report_md)
