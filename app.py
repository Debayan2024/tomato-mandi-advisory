import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Tomato Saathi - AI Mandi Advisory",
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
    st.error(f"Error loading artifacts: {e}. Please ensure 'tomato_artifacts.joblib' is in the root directory.")
    st.stop()

# 2. API Key from Streamlit Secrets
api_key = st.secrets.get("DATA_GOV_API_KEY", "")

# Live APMC Data Fetcher via data.gov.in API
@st.cache_data(ttl=3600)
def fetch_live_mandi_prices(key, state, district):
    if not key:
        return None
    try:
        url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
            "api-key": key,
            "format": "json",
            "filters[state.keyword]": state,
            "filters[district]": district,
            "filters[commodity]": "Tomato",
            "limit": 10
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            records = resp.json().get('records', [])
            if records:
                live_df = pd.DataFrame(records)[['market', 'variety', 'modal_price', 'arrival_date']]
                live_df.columns = ['Mandi', 'Variety', 'Today Live Price (₹/q)', 'Auction Date']
                return live_df
    except Exception:
        return None
    return None

# 3. Header
st.title("🍅 Tomato Saathi: Mandi Advisory & Price Forecasting")
st.markdown("Empowering Indian tomato farmers with AI-driven harvest timing, destination market selection, and liquidity guardrails.")
st.divider()

# 4. Sidebar: Farmer Inputs
st.sidebar.header("📍 Farmer Location & Inputs")

# Live API Status Badge in Sidebar
if api_key:
    st.sidebar.success("🟢 Government Live API Connected")
else:
    st.sidebar.info("⚪ Running on Offline AI Model")

# State & District Selection
states = sorted(geo['state_name'].unique().tolist())
default_state = "Maharashtra" if "Maharashtra" in states else states[0]
selected_state = st.sidebar.selectbox("Select State", states, index=states.index(default_state))

districts = sorted(geo[geo['state_name'] == selected_state]['district_name'].unique().tolist())
default_district = "Nashik" if "Nashik" in districts else districts[0]
selected_district = st.sidebar.selectbox("Select District", districts, index=districts.index(default_district))

st.sidebar.divider()

# Simple Mode Selector
planning_mode = st.sidebar.radio(
    "Choose What You Want To Do:",
    [
        "🌱 I am Planting (Tell me when & where to sell)",
        "🧺 My Crop is Ready (Where should I sell?)"
    ]
)

growth_days = {'Deshi': 70, 'Other': 75, 'Hybrid': 80}

if "Planting" in planning_mode:
    plant_date = st.sidebar.date_input("When will you plant?", datetime.today())
    
    # Farmer enters Land and Seeds (Clean Grams input)
    acres = st.sidebar.number_input("Your Farm Land Area (in Acres)", min_value=0.25, max_value=50.0, value=1.0, step=0.25)
    seed_grams = st.sidebar.number_input("Seeds Planted (in Grams)", min_value=10, max_value=2000, value=50, step=10, help="Standard: 40-50 grams seeds per acre (approx 4-5 packets of 10g each)")
    
    # Expected Harvest Yield Calculation (ICAR Standard: ~22 Tonnes/acre, picked across 5-6 trips)
    total_expected_harvest = round(acres * 22.0, 1)
    single_trip_load = round(total_expected_harvest / 5.5, 1) # per-trip load to mandi
    
    st.sidebar.caption(f"🌾 **Expected Total Harvest:** ~{total_expected_harvest} Tonnes  \n*(~{single_trip_load} Tonnes per mandi trip)*")
    
    mode_type = "forward"
    active_date = plant_date
    load_to_check = single_trip_load

else:
    target_harvest_date = st.sidebar.date_input("When will you harvest?", datetime.today() + timedelta(days=60))
    single_trip_load = st.sidebar.number_input("Your Harvest Load for this trip (Tonnes)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
    
    total_expected_harvest = single_trip_load
    mode_type = "backward"
    active_date = target_harvest_date
    load_to_check = single_trip_load

# 5. Core Recommendation Engine (All Features Restored)
def recommend(state, district, mode_type, ref_date, trip_load):
    mandis = geo[(geo['state_name'] == state) & (geo['district_name'] == district)]['market_center_name'].unique()
    if len(mandis) == 0:
        return pd.DataFrame(), "", None
        
    date_col = "Expected Harvest Date" if mode_type == "forward" else "Recommended Sowing Date"
    rows = []
    
    for m in mandis:
        vol = mandi_vol.get(m, 10.0)
        log_vol = np.log1p(vol)
        m_enc = mandi_means.get(m, train_mean)
        s_enc = state_means.get(state, train_mean)
        
        for v in ['Hybrid', 'Deshi', 'Other']:
            days = growth_days[v]
            if mode_type == "forward":
                harvest_dt = pd.to_datetime(ref_date) + timedelta(days=days)
                shown_date_str = harvest_dt.strftime('%d-%b-%Y')
                model_date = harvest_dt
            else:
                harvest_dt = pd.to_datetime(ref_date)
                sow_dt = harvest_dt - timedelta(days=days)
                shown_date_str = sow_dt.strftime('%d-%b-%Y')
                model_date = harvest_dt
                
            week = int(model_date.isocalendar().week)
            month = model_date.month
            
            rows.append({
                'Mandi': m,
                'Variety': v,
                date_col: shown_date_str,
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
    
    # Check if Mandi can absorb this trip load
    res['is_safe_capacity'] = res['Daily Vol (Tonnes)'] >= trip_load
    res['Liquidity Status'] = np.where(res['Daily Vol (Tonnes)'] >= 10.0, '🟢 High (Hub)',
                              np.where(res['Daily Vol (Tonnes)'] >= 2.0, '🟡 Medium', '🔴 Low (Risk)'))
    
    # Pick Top Safe Mandi with Highest Price
    safe_candidates = res[res['is_safe_capacity']]
    top_pick = safe_candidates.sort_values('Expected Price (₹/q)', ascending=False).iloc[0] if not safe_candidates.empty else res.sort_values('Daily Vol (Tonnes)', ascending=False).iloc[0]
    
    res_sorted = res.sort_values(by=['is_safe_capacity', 'Expected Price (₹/q)'], ascending=[False, False]).reset_index(drop=True)
    cols = ['Mandi', 'Variety', 'Expected Price (₹/q)', 'Expected Price (₹/kg)', 'Daily Vol (Tonnes)', 'Liquidity Status', date_col]
    return res_sorted[cols], date_col, top_pick

# 6. Render UI
results, date_col_name, top_pick = recommend(selected_state, selected_district, mode_type, active_date, load_to_check)

if not results.empty and top_pick is not None:
    
    # --- PROMINENT DIRECT STORY CARD ---
    if mode_type == "forward":
        st.markdown(f"""
        <div style="background-color: #1e3a2f; padding: 18px; border-radius: 10px; border-left: 6px solid #28a745; margin-bottom: 25px;">
            <h3 style="margin: 0; color: #a3e635;">🏆 Your Optimal Harvest & Profit Recommendation</h3>
            <p style="margin: 8px 0 0 0; font-size: 17px; color: #ffffff; line-height: 1.6;">
                For your <b>{acres} Acre</b> land (with <b>{seed_grams}g seeds</b>), your expected tomato harvest will be <b>~{total_expected_harvest} Tonnes</b>.<br>
                If you plant <b>{top_pick['Variety']}</b> on <b>{pd.to_datetime(plant_date).strftime('%d-%b-%Y')}</b>, you will harvest around <b>{top_pick[date_col_name]}</b>.<br>
                If you sell at <b>{top_pick['Mandi']}</b> (trades <b>{top_pick['Daily Vol (Tonnes)']} Tonnes/day</b>, safe for your volume), you will get the maximum expected gross price of: <br>
                <span style="font-size: 22px; color: #facc15; font-weight: bold;">₹{top_pick['Expected Price (₹/q)']:,} / quintal (₹{top_pick['Expected Price (₹/kg)']}/kg)</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #1e3a2f; padding: 18px; border-radius: 10px; border-left: 6px solid #28a745; margin-bottom: 25px;">
            <h3 style="margin: 0; color: #a3e635;">🏆 Best Mandi for Your Harvest</h3>
            <p style="margin: 8px 0 0 0; font-size: 17px; color: #ffffff; line-height: 1.6;">
                For your harvest around <b>{pd.to_datetime(target_harvest_date).strftime('%d-%b-%Y')}</b> (load: <b>{single_trip_load} Tonnes</b>):<br>
                Sell at <b>{top_pick['Mandi']}</b> for variety <b>{top_pick['Variety']}</b> to earn the highest price of: <br>
                <span style="font-size: 22px; color: #facc15; font-weight: bold;">₹{top_pick['Expected Price (₹/q)']:,} / quintal (₹{top_pick['Expected Price (₹/kg)']}/kg)</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Top KPI Metrics
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

    # Live APMC Data if Available
    if api_key:
        live_data = fetch_live_mandi_prices(api_key, selected_state, selected_district)
        if live_data is not None and not live_data.empty:
            st.success("🟢 Connected to Live Agmarknet Government Feed (Today's Real Mandi Auctions)")
            st.dataframe(live_data, use_container_width=True)
            st.divider()

    # Comparison Table & Chart
    tab1, tab2 = st.tabs(["📋 Mandi & Variety Comparison Table", "📊 Visual Price Chart"])
    with tab1:
        st.dataframe(results, use_container_width=True)
    with tab2:
        st.bar_chart(data=results, x="Mandi", y="Expected Price (₹/q)", color="Variety")
