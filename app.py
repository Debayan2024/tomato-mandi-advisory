import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

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
try:
    api_key = st.secrets.get("DATA_GOV_API_KEY", "")
except Exception:
    api_key = ""

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

# Today's date (India time, fetched automatically)
def get_today():
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Kolkata")).date()
        except Exception:
            pass
    return datetime.utcnow().date() + timedelta(hours=5, minutes=30) if False else (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()

today = get_today()

# Constants
growth_days = {'Deshi': 70, 'Other': 75, 'Hybrid': 80}
VARIETIES = ['Hybrid', 'Deshi', 'Other']
SCAN_DAYS = 180            # how many days from today we search for the best planting date
YIELD_QUINTAL_PER_ACRE = 220.0   # 22 Tonnes/acre = 220 quintal/acre
TRIPS = 5.5                # harvest is picked across ~5-6 trips

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

st.sidebar.markdown(f"📅 **Today:** {today.strftime('%d-%b-%Y')}")

# Farmer enters Land (Acres) and Seeds (Grams)
acres = st.sidebar.number_input("Your Farm Land Area (in Acres)", min_value=0.25, max_value=50.0, value=1.0, step=0.25)
seed_grams = st.sidebar.number_input("Seeds Planted (in Grams)", min_value=10, max_value=2000, value=50, step=10, help="Standard: 40-50 grams seeds per acre (approx 4-5 packets of 10g each)")

# Expected Harvest Yield (in Quintal): ~22 Tonnes/acre = 220 Quintal/acre, picked across 5-6 trips
total_expected_quintal = round(acres * YIELD_QUINTAL_PER_ACRE, 1)
single_trip_quintal = round(total_expected_quintal / TRIPS, 1)
single_trip_load_tonnes = single_trip_quintal / 10.0   # mandi volumes in the model are in Tonnes

st.sidebar.caption(f"🌾 **Expected Total Harvest:** ~{total_expected_quintal} Quintal  \n*(~{single_trip_quintal} Quintal per mandi trip)*")

# 5. Core Recommendation Engine: scans every planting date from today
def scan_plant_dates(state, district, start_date, trip_load_tonnes):
    mandis = geo[(geo['state_name'] == state) & (geo['district_name'] == district)]['market_center_name'].unique()
    if len(mandis) == 0:
        return pd.DataFrame()

    plant_dates = pd.date_range(start=pd.Timestamp(start_date), periods=SCAN_DAYS, freq='D')
    s_enc = state_means.get(state, train_mean)
    frames = []

    for m in mandis:
        vol = mandi_vol.get(m, 10.0)
        log_vol = np.log1p(vol)
        m_enc = mandi_means.get(m, train_mean)

        for v in VARIETIES:
            harvest_dates = plant_dates + pd.to_timedelta(growth_days[v], unit='D')
            week = harvest_dates.isocalendar().week.to_numpy().astype(int)
            month = harvest_dates.month.to_numpy()

            frames.append(pd.DataFrame({
                'Mandi': m,
                'Variety': v,
                'Plant Date': plant_dates,
                'Harvest Date': harvest_dates,
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
            }))

    res = pd.concat(frames, ignore_index=True)
    res['Expected Price (₹/q)'] = np.round(model.predict(res[feature_cols]), 2)
    res['Expected Price (₹/kg)'] = np.round(res['Expected Price (₹/q)'] / 100, 2)

    # Check if Mandi can absorb one trip load
    res['is_safe_capacity'] = res['Daily Vol (Tonnes)'] >= trip_load_tonnes
    res['Liquidity Status'] = np.where(res['Daily Vol (Tonnes)'] >= 10.0, '🟢 High (Hub)',
                              np.where(res['Daily Vol (Tonnes)'] >= 2.0, '🟡 Medium', '🔴 Low (Risk)'))
    return res

def best_per_variety(res):
    """For each variety: best (plant date, mandi) among safe mandis; if none is safe, use the biggest-volume mandi."""
    picks = {}
    for v in VARIETIES:
        pool = res[res['Variety'] == v]
        safe = pool[pool['is_safe_capacity']]
        fallback = safe.empty
        if fallback:
            pool = pool[pool['Daily Vol (Tonnes)'] == pool['Daily Vol (Tonnes)'].max()]
        else:
            pool = safe
        row = pool.loc[pool['Expected Price (₹/q)'].idxmax()]
        picks[v] = (row, fallback)
    return picks

# 6. Render UI
res_all = scan_plant_dates(selected_state, selected_district, today, single_trip_load_tonnes)

if res_all.empty:
    st.warning("No mandi data available for this district.")
else:
    picks = best_per_variety(res_all)
    fmt = lambda d: pd.to_datetime(d).strftime('%d-%b-%Y')
    icons = {'Hybrid': '🍅', 'Deshi': '🌿', 'Other': '🌱'}

    # --- PROMINENT DIRECT STORY CARD ---
    points_html = ""
    for v in VARIETIES:
        row, fallback = picks[v]
        warn = " <i>(⚠️ no mandi here safely absorbs a full trip, showing the biggest market)</i>" if fallback else ""
        points_html += (
            f"<p style='margin: 14px 0 0 0; font-size: 17px; color: #ffffff; line-height: 1.6;'>"
            f"{icons[v]} <b>{v}:</b> Plant <b>{v}</b> on <b style='color:#a3e635;'>{fmt(row['Plant Date'])}</b> "
            f"and sell at <b>{row['Mandi']}</b> on <b style='color:#a3e635;'>{fmt(row['Harvest Date'])}</b> "
            f"(expected harvest date, mandi trades <b>{row['Daily Vol (Tonnes)']} Tonnes/day</b>) so that you earn the most.{warn}<br>"
            f"Expected maximum price: "
            f"<span style='font-size: 22px; color: #facc15; font-weight: bold;'>₹{row['Expected Price (₹/q)']:,} / quintal (₹{row['Expected Price (₹/kg)']}/kg)</span>"
            f"</p>"
        )

    st.markdown(
        f"<div style='background-color: #1e3a2f; padding: 18px; border-radius: 10px; border-left: 6px solid #28a745; margin-bottom: 25px;'>"
        f"<h3 style='margin: 0; color: #a3e635;'>🏆 Your Optimal Planting & Profit Recommendation</h3>"
        f"<p style='margin: 8px 0 0 0; font-size: 17px; color: #ffffff; line-height: 1.6;'>"
        f"For your <b>{acres} Acre</b> land (with <b>{seed_grams}g seeds</b>), your expected tomato harvest will be "
        f"<b>~{total_expected_quintal} Quintal</b>. Looking at the next {SCAN_DAYS} days from today (<b>{today.strftime('%d-%b-%Y')}</b>):"
        f"</p>{points_html}</div>",
        unsafe_allow_html=True
    )

    # Top KPI Metrics: one column per variety
    cols = st.columns(3)
    for col, v in zip(cols, VARIETIES):
        row, _ = picks[v]
        with col:
            st.markdown(
                f"**{icons[v]} {v} - Best Price**  \n"
                f"### ₹{row['Expected Price (₹/q)']:,} / q  \n"
                f"*(₹{row['Expected Price (₹/kg)']}/kg)*  \n"
                f"Plant: **{fmt(row['Plant Date'])}**  \n"
                f"Harvest/Sell: **{fmt(row['Harvest Date'])}**  \n"
                f"Mandi: **{row['Mandi']}**"
            )

    st.divider()

    # Live APMC Data if Available
    if api_key:
        live_data = fetch_live_mandi_prices(api_key, selected_state, selected_district)
        if live_data is not None and not live_data.empty:
            st.success("🟢 Connected to Live Agmarknet Government Feed (Today's Real Mandi Auctions)")
            st.dataframe(live_data, use_container_width=True)
            st.divider()

    # Comparison Table & Chart: best planting date for every mandi x variety
    best_rows = res_all.loc[res_all.groupby(['Mandi', 'Variety'])['Expected Price (₹/q)'].idxmax()].copy()
    best_rows = best_rows.sort_values(by=['is_safe_capacity', 'Expected Price (₹/q)'], ascending=[False, False]).reset_index(drop=True)
    best_rows['Best Plant Date'] = best_rows['Plant Date'].dt.strftime('%d-%b-%Y')
    best_rows['Expected Harvest Date'] = best_rows['Harvest Date'].dt.strftime('%d-%b-%Y')
    table_cols = ['Mandi', 'Variety', 'Best Plant Date', 'Expected Harvest Date',
                  'Expected Price (₹/q)', 'Expected Price (₹/kg)', 'Daily Vol (Tonnes)', 'Liquidity Status']

    tab1, tab2 = st.tabs(["📋 Mandi & Variety Comparison Table", "📊 Visual Price Chart"])
    with tab1:
        st.dataframe(best_rows[table_cols], use_container_width=True)
    with tab2:
        st.bar_chart(data=best_rows, x="Mandi", y="Expected Price (₹/q)", color="Variety")
