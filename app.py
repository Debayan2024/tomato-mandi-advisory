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

# 2. Live APMC Data Fetcher via data.gov.in API
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_live_mandi_prices(api_key, state, district):
    if not api_key:
        return None
    try:
        url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
            "api-key": api_key,
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
                return live_df
    except Exception:
        return None
    return None

# 3. Header
st.title("🍅 Tomato Saathi: Mandi Advisory & Price Forecasting")
st.markdown("Empowering Indian tomato farmers with AI-driven harvest timing, destination market selection, and liquidity guardrails.")
st.divider()

# 4. Sidebar: Farmer Inputs
st.sidebar.header("📍 Farmer Location & Farm Size")

# State & District Selection
states = sorted(geo['state_name'].unique().tolist())
default_state = "Maharashtra" if "Maharashtra" in states else states[0]
selected_state = st.sidebar.selectbox("Select State", states, index=states.index(default_state))

districts = sorted(geo[geo['state_name'] == selected_state]['district_name'].unique().tolist())
default_district = "Nashik" if "Nashik" in districts else districts[0]
selected_district = st.sidebar.selectbox("Select District", districts, index=districts.index(default_district))

# --- FEATURE 3: Yield Calculator (Land Area / Plants -> Truckload Tonnes) ---
st.sidebar.subheader("🚜 Your Farm Size & Harvest Calculator")
calc_mode = st.sidebar.radio(
    "How do you want to enter your farm size?",
    ["📐 Land Area (Acres)", "🌱 Number of Saplings/Plants", "⚖️ Direct Tonnes (Experienced)"]
)

if "Land Area" in calc_mode:
    acres = st.sidebar.number_input("Enter Land Area (in Acres)", min_value=0.25, max_value=20.0, value=1.0, step=0.25)
    # 1 Acre produces ~20 Tonnes total, picked across 6 rounds (~3.3 Tonnes per truckload)
    truck_load = round(acres * 3.3, 1)
    st.sidebar.caption(f"💡 Estimated per-picking truckload: **~{truck_load} Tonnes** (Total crop ~{round(acres * 20, 1)}T across 6 pickings)")

elif "Saplings" in calc_mode:
    plants = st.sidebar.number_input("Enter Total Plant Count", min_value=500, max_value=100000, value=5000, step=500)
    # Average hybrid yields ~4.5 kg total across 6 pickings
    truck_load = round((plants * 4.5) / (1000 * 6), 1)
    truck_load = max(0.5, truck_load)
    st.sidebar.caption(f"💡 Estimated per-picking truckload: **~{truck_load} Tonnes**")

else:
    truck_load = st.sidebar.number_input("Your Expected Harvest Load per Truck (Tonnes)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)

st.sidebar.divider()

# Planning Mode
planning_mode = st.sidebar.radio(
    "Choose What You Want To Do:",
    ["🌱 I am Planting (When will I harvest?)", "🧺 My Crop is Ready (Where should I sell?)"]
)

growth_days = {'Deshi': 70, 'Other': 75, 'Hybrid': 80}

if "Planting" in planning_mode:
    base_date = st.sidebar.date_input("When are you planting?", datetime.today())
    st.sidebar.markdown("""
    **⏱️ Days to First Picking:**
    * **Deshi:** 70 Days | **Other:** 75 Days | **Hybrid:** 80 Days
    """)
    mode_type = "forward"
else:
    base_date = st.sidebar.date_input("When will you harvest?", datetime.today() + timedelta(days=60))
    mode_type = "backward"

# Optional API Key input in sidebar
api_key = st.sidebar.text_input("data.gov.in API Key (Optional)", type="password", help="Enter free API key to see live today APMC rates.")

# 5. Core Recommendation Function
def recommend(state, district, mode_type, ref_date):
    mandis = geo[(geo['state_name'] == state) & (geo['district_name'] == district)]['market_center_name'].unique()
    if len(mandis) == 0:
        return pd.DataFrame(), "", None

    varieties = ['Hybrid', 'Deshi', 'Other']
    rows = []
    date_col = "Expected Harvest Date" if mode_type == "forward" else "Recommended Sowing Date"
    
    for m in mandis:
        vol = mandi_vol.get(m, 2.0)
        log_vol = np.log1p(vol)
        m_enc = mandi_means.get(m, train_mean)
        s_enc = state_means.get(state, train_mean)
        
        for v in varieties:
            if mode_type == "forward":
                harvest_dt = pd.to_datetime(ref_date) + timedelta(days=growth_days[v])
                display_date = harvest_dt.strftime('%d-%b-%Y')
                model_date = harvest_dt
            else:
                harvest_dt = pd.to_datetime(ref_date)
                plant_dt = harvest_dt - timedelta(days=growth_days[v])
                display_date = plant_dt.strftime('%d-%b-%Y')
                model_date = harvest_dt
                
            week = int(model_date.isocalendar().week)
            month = model_date.month
            
            rows.append({
                'Mandi': m,
                'Variety': v,
                date_col: display_date,
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
    res['is_safe_capacity'] = res['Daily Vol (Tonnes)'] >= truck_load
    res['Liquidity Status'] = np.where(res['Daily Vol (Tonnes)'] >= 10.0, '🟢 High (Hub)',
                              np.where(res['Daily Vol (Tonnes)'] >= 2.0, '🟡 Medium', '🔴 Low (Risk)'))
    
    safe_candidates = res[res['is_safe_capacity']]
    top_pick = safe_candidates.sort_values('Expected Price (₹/q)', ascending=False).iloc[0] if not safe_candidates.empty else res.sort_values('Daily Vol (Tonnes)', ascending=False).iloc[0]
    
    res_sorted = res.sort_values(by=['is_safe_capacity', 'Expected Price (₹/q)'], ascending=[False, False]).reset_index(drop=True)
    cols = ['Mandi', 'Variety', 'Expected Price (₹/q)', 'Expected Price (₹/kg)', 'Daily Vol (Tonnes)', 'Liquidity Status', date_col]
    return res_sorted[cols], date_col, top_pick

# --- FEATURE 2: Optimal Planting Date Scanner (Maximum Profit Finder) ---
def find_optimal_sowing_window(state, district):
    today = datetime.today()
    candidate_dates = [today + timedelta(days=7 * i) for i in range(1, 13)] # Scan next 12 weeks
    best_option = None
    max_price = -1
    
    for c_date in candidate_dates:
        df_scan, _, top = recommend(state, district, "forward", c_date)
        if top is not None and top['Expected Price (₹/q)'] > max_price:
            max_price = top['Expected Price (₹/q)']
            best_option = {
                'plant_date': c_date.strftime('%d-%b-%Y'),
                'harvest_date': top['Expected Harvest Date'],
                'mandi': top['Mandi'],
                'variety': top['Variety'],
                'price_q': top['Expected Price (₹/q)'],
                'price_kg': top['Expected Price (₹/kg)']
            }
    return best_option

# 6. Render UI
# Banner: Feature 2 - Optimal Profit Finder
optimal_window = find_optimal_sowing_window(selected_state, selected_district)
if optimal_window:
    st.markdown(f"""
    <div style="background-color: #1e3a2f; padding: 15px; border-radius: 10px; border-left: 6px solid #28a745; margin-bottom: 20px;">
        <h4 style="margin: 0; color: #a3e635;">🏆 Best Planting Window for Maximum Profit (Next 3 Months)</h4>
        <p style="margin: 5px 0 0 0; font-size: 16px; color: #ffffff;">
            If you transplant saplings on <b>{optimal_window['plant_date']}</b>, you will harvest around <b>{optimal_window['harvest_date']}</b> to catch the peak price at <b>{optimal_window['mandi']}</b> for <b>{optimal_window['variety']}</b> at an estimated <b>₹{optimal_window['price_q']:,} / quintal (₹{optimal_window['price_kg']}/kg)</b>!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main Recommendation Results
results, date_col_name, top_pick = recommend(selected_state, selected_district, mode_type, base_date)

if not results.empty and top_pick is not None:
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

    # Optional Live APMC Price Table
    live_data = fetch_live_mandi_prices(api_key, selected_state, selected_district)
    if live_data is not None and not live_data.empty:
        st.subheader("🔴 Today's Live Mandi Trading Rates (Agmarknet Live API)")
        st.dataframe(live_data, use_container_width=True)

    tab1, tab2 = st.tabs(["📋 Mandi & Variety Comparison Table", "📊 Visual Price Chart"])
    with tab1:
        st.dataframe(results, use_container_width=True)
    with tab2:
        st.bar_chart(data=results, x="Mandi", y="Expected Price (₹/q)", color="Variety")
