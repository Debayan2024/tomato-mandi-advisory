import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Tomato Saathi - Mandi Advisory",
    page_icon="🍅",
    layout="wide"
)

# 1. Load Pretrained Artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load('tomato_artifacts.joblib')

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    feature_cols = artifacts['features']
    mandi_means = artifacts['mandi_means']
    state_means = artifacts['state_means']
    mandi_vol = artifacts['mandi_vol']
    train_mean = artifacts['train_mean']
    geo = artifacts['geo']
except Exception as e:
    st.error(f"Error loading artifacts: {e}. Please ensure 'tomato_artifacts.joblib' is placed in this folder.")
    st.stop()

# 2. Header
st.title("🍅 Tomato Saathi: Mandi Advisory & Price Forecasting")
st.markdown("Empowering Indian tomato farmers with AI-driven harvest timing, destination market selection, and liquidity guardrails.")
st.divider()

# 3. Sidebar: Farmer Inputs
st.sidebar.header("📍 Farmer Location & Timing")

# State Selection
states = sorted(geo['state_name'].unique().tolist())
default_state = "Maharashtra" if "Maharashtra" in states else states[0]
selected_state = st.sidebar.selectbox("Select State", states, index=states.index(default_state))

# District Selection
districts = sorted(geo[geo['state_name'] == selected_state]['district_name'].unique().tolist())
default_district = "Nashik" if "Nashik" in districts else districts[0]
selected_district = st.sidebar.selectbox("Select District", districts, index=districts.index(default_district))

# Simple Everyday Language Selector
planning_mode = st.sidebar.radio(
    "Choose What You Want To Do:",
    [
        "🌱 I am Planting (When will I harvest?)", 
        "🧺 My Crop is Ready (Where should I sell?)"
    ]
)

growth_days = {'Deshi': 70, 'Other': 75, 'Hybrid': 80}

if "Planting" in planning_mode:
    plant_date = st.sidebar.date_input("When are you planting?", datetime.today())
    st.sidebar.markdown("""
    **⏱️ Days to First Picking:**
    * **Deshi:** 70 Days
    * **Other:** 75 Days
    * **Hybrid:** 80 Days
    """)
    mode_type = "forward"
else:
    target_harvest_date = st.sidebar.date_input("When will you harvest?", datetime.today() + timedelta(days=60))
    mode_type = "backward"

truck_load = st.sidebar.number_input("Your Expected Harvest Load (Tonnes)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)

# 4. Recommendation Engine
def recommend(state, district, mode_type, base_date):
    mandis = geo[(geo['state_name'] == state) & (geo['district_name'] == district)]['market_center_name'].unique()
    
    if len(mandis) == 0:
        st.warning("No specific mandis found for this district.")
        return pd.DataFrame(), "", None

    varieties = ['Hybrid', 'Deshi', 'Other']
    rows = []
    
    date_col_name = "Expected Harvest Date" if mode_type == "forward" else "Recommended Sowing Date"
    
    for m in mandis:
        vol = mandi_vol.get(m, 2.0)
        log_vol = np.log1p(vol)
        m_enc = mandi_means.get(m, train_mean)
        s_enc = state_means.get(state, train_mean)
        
        for v in varieties:
            if mode_type == "forward":
                harvest_dt = pd.to_datetime(base_date) + timedelta(days=growth_days[v])
                display_date = harvest_dt.strftime('%d-%b-%Y')
                model_date = harvest_dt
            else:
                harvest_dt = pd.to_datetime(base_date)
                plant_dt = harvest_dt - timedelta(days=growth_days[v])
                display_date = plant_dt.strftime('%d-%b-%Y')
                model_date = harvest_dt
                
            week = int(model_date.isocalendar().week)
            month = model_date.month
            
            rows.append({
                'Mandi': m,
                'Variety': v,
                date_col_name: display_date,
                'Daily Vol (Tonnes)': round(vol, 1),
                'mandi_encoded': m_enc,
                'state_encoded': s_enc,
                'log_arrival': log_vol,
                'mandi_arrival_interact': m_enc * log_vol,
                'sin_week': np.sin(2 * np.pi * week / 52),
                'cos_week': np.cos(2 * np.pi * week / 52),
                'sin_month': np.sin(2 * np.pi * month / 12),
                'cos_month': np.cos(2 * np.pi * month / 12),
                'variety_Deshi': int(v == 'Deshi'),
                'variety_Hybrid': int(v == 'Hybrid'),
                'variety_Other': int(v == 'Other')
            })
            
    res = pd.DataFrame(rows)
    res['Expected Price (₹/q)'] = np.round(model.predict(res[feature_cols]), 2)
    res['Expected Price (₹/kg)'] = np.round(res['Expected Price (₹/q)'] / 100, 2)
    
    # Safe Capacity Check
    res['is_safe_capacity'] = res['Daily Vol (Tonnes)'] >= truck_load
    
    # Liquidity Badge
    res['Liquidity Status'] = np.where(res['Daily Vol (Tonnes)'] >= 10.0, '🟢 High (Hub)',
                              np.where(res['Daily Vol (Tonnes)'] >= 2.0, '🟡 Medium', '🔴 Low (Risk)'))
    
    # Prioritize Safe Markets First
    safe_candidates = res[res['is_safe_capacity']]
    if not safe_candidates.empty:
        top_pick = safe_candidates.sort_values('Expected Price (₹/q)', ascending=False).iloc[0]
    else:
        top_pick = res.sort_values('Daily Vol (Tonnes)', ascending=False).iloc[0]

    # Sort table: Safe first, then highest price
    res_sorted = res.sort_values(by=['is_safe_capacity', 'Expected Price (₹/q)'], ascending=[False, False]).reset_index(drop=True)
    
    cols = ['Mandi', 'Variety', 'Expected Price (₹/q)', 'Expected Price (₹/kg)', 'Daily Vol (Tonnes)', 'Liquidity Status', date_col_name]
    return res_sorted[cols], date_col_name, top_pick

# 5. Output UI
active_date = plant_date if mode_type == "forward" else target_harvest_date
results, date_col_name, top_pick = recommend(selected_state, selected_district, mode_type, active_date)

if not results.empty and top_pick is not None:
    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**🥇 Recommended Mandi**  \n### {top_pick['Mandi']}")
    with col2:
        st.markdown(f"**🍅 Best Variety**  \n### {top_pick['Variety']}")
    with col3:
        st.markdown(f"**💰 Expected Price**  \n### ₹{top_pick['Expected Price (₹/q)']:,} / q  \n*(₹{top_pick['Expected Price (₹/kg)']}/kg)*")
    with col4:
        st.markdown(f"**📅 {date_col_name}**  \n### {top_pick[date_col_name]}")
    
    st.divider()

    # Tabs: Table and Visual Chart (No extra banner boxes!)
    tab1, tab2 = st.tabs(["📋 Mandi & Variety Comparison Table", "📊 Visual Price Chart"])
    with tab1:
        st.dataframe(results, use_container_width=True)
    with tab2:
        st.bar_chart(data=results, x="Mandi", y="Expected Price (₹/q)", color="Variety")
