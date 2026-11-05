import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Tomato Mandi Advisory System",
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
st.title("🍅 Tomato Mandi Advisory & Price Forecasting System")
st.markdown("Empowering Indian tomato farmers with AI-driven harvest timing, destination market selection, and liquidity guardrails.")
st.divider()

# 3. Sidebar: Cascading Dropdowns & Inputs
st.sidebar.header("📍 Farmer Location & Timing")

# State Selection
states = sorted(geo['state_name'].unique().tolist())
default_state = "Maharashtra" if "Maharashtra" in states else states[0]
selected_state = st.sidebar.selectbox("Select State", states, index=states.index(default_state))

# District Selection (Filtered by State)
districts = sorted(geo[geo['state_name'] == selected_state]['district_name'].unique().tolist())
default_district = "Nashik" if "Nashik" in districts else districts[0]
selected_district = st.sidebar.selectbox("Select District", districts, index=districts.index(default_district))

planning_mode = st.sidebar.radio(
    "Choose Planning Mode:",
    ["📅 Target Harvest Date (Where to sell?)", "🌱 Planting Today (When to harvest?)"]
)

if "Harvest Date" in planning_mode:
    target_harvest_date = st.sidebar.date_input("Target Harvest Date", datetime.today() + timedelta(days=30))
else:
    plant_date = st.sidebar.date_input("Transplanting Date (Today)", datetime.today())
    target_harvest_date = plant_date + timedelta(days=75)
    st.sidebar.info(f"Estimated Harvest Window: **{target_harvest_date.strftime('%d-%b-%Y')}**")

truck_load = st.sidebar.number_input("Your Expected Harvest Load (Tonnes)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)

# 4. Recommendation Engine
def recommend(state, district, target_dt):
    dt = pd.to_datetime(target_dt)
    mandis = geo[(geo['state_name'] == state) & (geo['district_name'] == district)]['market_center_name'].unique()
    
    if len(mandis) == 0:
        st.warning("No specific mandis found for this district.")
        return pd.DataFrame()

    growth_days = {'Hybrid': 80, 'Deshi': 70, 'Other': 75}
    varieties = ['Hybrid', 'Deshi', 'Other']
    
    rows = []
    for m in mandis:
        vol = mandi_vol.get(m, 2.0)
        log_vol = np.log1p(vol)
        m_enc = mandi_means.get(m, train_mean)
        s_enc = state_means.get(state, train_mean)
        
        for v in varieties:
            p_date = (dt - timedelta(days=growth_days[v])).date()
            rows.append({
                'Mandi': m,
                'Variety': v,
                'Plant Date': p_date.strftime('%d-%b-%Y'),
                'Daily Vol (Tonnes)': round(vol, 1),
                'mandi_encoded': m_enc,
                'state_encoded': s_enc,
                'log_arrival': log_vol,
                'mandi_arrival_interact': m_enc * log_vol,
                'sin_week': np.sin(2 * np.pi * dt.isocalendar().week / 52),
                'cos_week': np.cos(2 * np.pi * dt.isocalendar().week / 52),
                'sin_month': np.sin(2 * np.pi * dt.month / 12),
                'cos_month': np.cos(2 * np.pi * dt.month / 12),
                'variety_Deshi': int(v == 'Deshi'),
                'variety_Hybrid': int(v == 'Hybrid'),
                'variety_Other': int(v == 'Other')
            })
            
    res = pd.DataFrame(rows)
    res['Expected Price (₹/q)'] = np.round(model.predict(res[feature_cols]), 2)
    res['Expected Price (₹/kg)'] = np.round(res['Expected Price (₹/q)'] / 100, 2)
    
    # Liquidity Badge
    res['Liquidity Status'] = np.where(res['Daily Vol (Tonnes)'] >= 10.0, '🟢 High (Hub)',
                              np.where(res['Daily Vol (Tonnes)'] >= 2.0, '🟡 Medium', '🔴 Low (Risk)'))
    
    cols = ['Mandi', 'Variety', 'Expected Price (₹/q)', 'Expected Price (₹/kg)', 'Daily Vol (Tonnes)', 'Liquidity Status', 'Plant Date']
    return res[cols].sort_values('Expected Price (₹/q)', ascending=False).reset_index(drop=True)

# 5. Output UI
results = recommend(selected_state, selected_district, target_harvest_date)

if not results.empty:
    top_pick = results.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🥇 Top Recommended Mandi", top_pick['Mandi'])
    col2.metric("🍅 Best Variety", top_pick['Variety'])
    col3.metric("💰 Expected Gross Price", f"₹{top_pick['Expected Price (₹/q)']:,} / q", f"₹{top_pick['Expected Price (₹/kg)']}/kg")
    col4.metric("🌱 Sowing Date", top_pick['Plant Date'])
    
    st.divider()
    
    if top_pick['Daily Vol (Tonnes)'] < truck_load:
        st.warning(f"⚠️ High Risk Warning: Top mandi ({top_pick['Mandi']}) only trades {top_pick['Daily Vol (Tonnes)']} Tonnes/day. A {truck_load}-Tonne load might crash local prices! Consider a High Liquidity hub.")
    else:
        st.success(f"✅ Safe Trading Volume: {top_pick['Mandi']} trades {top_pick['Daily Vol (Tonnes)']} Tonnes daily and can easily absorb your {truck_load}-Tonne harvest!")
        
    tab1, tab2 = st.tabs(["📋 Mandi & Variety Comparison Table", "📊 Visual Price Chart"])
    with tab1:
        st.dataframe(results, use_container_width=True)
    with tab2:
        st.bar_chart(data=results, x="Mandi", y="Expected Price (₹/q)", color="Variety")