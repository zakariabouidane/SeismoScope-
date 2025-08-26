import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Page setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="SeismoScope",
    layout="wide",
    page_icon="favicon.svg",
    initial_sidebar_state="expanded",
)

# Title / subtitle
try:
    st.image("seismoscope_logo.svg", width=200)  # Adjust width as needed
except FileNotFoundError:
    # Fallback to text title if logo file is not found
    st.title("SeismoScope")
st.caption("Professional UX • Interactive Maps • Forecasting • Advanced Geophysics Tools")

# -------------------------------------------------------------
# Session defaults for predictive parameters
# -------------------------------------------------------------
if "pred_min_mag" not in st.session_state:
    st.session_state.pred_min_mag = 5.0
if "pred_time_window" not in st.session_state:
    st.session_state.pred_time_window = 10
if "pred_risk_threshold" not in st.session_state:
    st.session_state.pred_risk_threshold = 0.3

# -------------------------------------------------------------
# Constants & helpers
# -------------------------------------------------------------
FEEDS = {
    "Past Hour": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
    "Past Day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
    "Past Week": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson",
    "Past Month": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson",
}

EXTENDED_TIMEFRAMES = {
    "Custom Range": "custom",
    "Past 3 Months": "3months",
    "Past Year": "year",
    "Past 5 Years": "5years",
    "Past 10 Years": "10years",
    "Past 20 Years": "20years",
    "Past 50 Years": "50years",
    "20th Century (1900-1999)": "20th_century",
    "21st Century (2000-Present)": "21st_century",
}

TIMEFRAMES_NEEDING_MIN_MAG = [
    "Past 5 Years",
    "Past 10 Years",
    "Past 20 Years",
    "Past 50 Years",
    "20th Century (1900-1999)",
    "21st Century (2000-Present)",
]

# OpenStreetMap tile styles (free, with attribution)
OSM_STYLES = {
    "OSM Standard": {
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "subdomains": ["a", "b", "c"],
        "attribution": "© OpenStreetMap contributors"
    },
    "OSM Humanitarian": {
        "url": "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        "subdomains": ["a", "b", "c"],
        "attribution": "© OpenStreetMap contributors, © Humanitarian OSM Team"
    },
    "OpenTopoMap": {
        "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "subdomains": ["a", "b", "c"],
        "attribution": "© OpenStreetMap contributors, © OpenTopoMap (CC-BY-SA)"
    },
}

def make_osm_tile_layer(style_name: str) -> pdk.Layer:
    style = OSM_STYLES.get(style_name, OSM_STYLES["OSM Standard"])
    # deck.gl TileLayer supports a single URL template; we replace {s} with 'a' (or rotate if desired)
    url_template = style["url"].replace("{s}", style["subdomains"][0])
    return pdk.Layer(
        "TileLayer",
        data=url_template,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0,
        # Keep attribution available for compliance (shown by Streamlit footer note too)
        # Note: pydeck doesn't show this string itself; we display it below in the UI.
        # Passing it here documents the source in the layer spec.
        attribution=style["attribution"],
    )

# ------------------- Utility Functions -----------------------
def extract_country(place: str) -> str:
    if place and "," in place:
        country = place.split(",")[-1].strip()
        country = country.split()[-1]
    else:
        country = place or "Unknown"
    return country

@st.cache_data(show_spinner=False, ttl=600)
def fetch_geojson(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=600)
def fetch_usgs_range(start_date: datetime, end_date: datetime, min_mag: float) -> dict:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?"
        f"format=geojson&starttime={start_str}&endtime={end_str}"
        f"&minmagnitude={min_mag}&limit=20000"
    )
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.json()

# ------------------- Geophysics Functions --------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_seismic_gaps(df, min_mag=6.0, time_window_years=10):
    if len(df) == 0:
        return pd.DataFrame()
    sig = df[df["magnitude"] >= min_mag].copy()
    if len(sig) < 2:
        return pd.DataFrame()
    sig["time"] = pd.to_datetime(sig["time"])
    sig = sig.sort_values("time")
    sig["lat_bin"] = np.round(sig["lat"] / 2.0) * 2.0
    sig["lon_bin"] = np.round(sig["lon"] / 2.0) * 2.0
    gaps = []
    for (lat_bin, lon_bin), group in sig.groupby(["lat_bin", "lon_bin"]):
        last_eq = group.iloc[-1]
        time_since_last = (datetime.now() - last_eq["time"]).days / 365.25
        if time_since_last > time_window_years:
            gaps.append(
                {
                    "latitude": lat_bin,
                    "longitude": lon_bin,
                    "time_since_last_eq": time_since_last,
                    "last_magnitude": last_eq["magnitude"],
                    "last_date": last_eq["time"],
                    "region_earthquakes": len(group),
                }
            )
    return pd.DataFrame(gaps)

def identify_foreshock_sequences(df, time_window_hours=72, distance_km=100):
    if len(df) < 2:
        return pd.DataFrame()
    tmp = df.sort_values("time").copy()
    tmp["time"] = pd.to_datetime(tmp["time"])
    sequences = []
    for i in range(len(tmp)):
        mainshock = tmp.iloc[i]
        if mainshock["magnitude"] < 4.0:
            continue
        time_mask = (mainshock["time"] - tmp["time"]).dt.total_seconds() <= time_window_hours * 3600
        time_mask &= (mainshock["time"] > tmp["time"])
        distances = haversine_distance(mainshock["lat"], mainshock["lon"], tmp["lat"], tmp["lon"])
        distance_mask = distances <= distance_km
        foreshocks = tmp[time_mask & distance_mask]
        if len(foreshocks) > 0:
            sequences.append(
                {
                    "mainshock_time": mainshock["time"],
                    "mainshock_magnitude": mainshock["magnitude"],
                    "mainshock_location": mainshock.get("place", ""),
                    "foreshock_count": len(foreshocks),
                    "largest_foreshock": foreshocks["magnitude"].max(),
                    "time_before_mainshock_hours": (
                        (mainshock["time"] - foreshocks["time"].min()).total_seconds() / 3600
                    ),
                }
            )
    return pd.DataFrame(sequences)

def estimate_ground_motion(magnitude, distance_km, site_class="B"):
    if magnitude < 3.0:
        return 0.0
    pga = np.exp(-2.0 + 0.6 * magnitude - np.log(distance_km + 10) - 0.003 * distance_km)
    site_factors = {"A": 0.8, "B": 1.0, "C": 1.2, "D": 1.4}
    pga *= site_factors.get(site_class, 1.0)
    return round(float(pga), 3)

def assess_liquefaction_potential(pga, groundwater_depth_m=2, fines_content=15):
    if pga < 0.1:
        return "Very Low"
    gw_factor = max(0.5, min(1.5, 2.0 - groundwater_depth_m / 5.0))
    fines_factor = max(0.7, min(1.3, 1.0 + (fines_content - 15) / 50))
    liquefaction_index = pga * gw_factor * fines_factor
    if liquefaction_index < 0.15:
        return "Low"
    elif liquefaction_index < 0.25:
        return "Moderate"
    elif liquefaction_index < 0.4:
        return "High"
    else:
        return "Very High"

def generate_predictive_data(df, min_mag=5.0, time_window_years=10):
    if len(df) == 0:
        return pd.DataFrame()
    min_mag = getattr(st.session_state, "pred_min_mag", min_mag)
    time_window_years = getattr(st.session_state, "pred_time_window", time_window_years)
    sig = df[df["magnitude"] >= min_mag].copy()
    if len(sig) < 2:
        return pd.DataFrame()
    sig["time"] = pd.to_datetime(sig["time"])
    sig = sig.sort_values("time")
    sig["lat_bin"] = np.round(sig["lat"] / 1.0) * 1.0
    sig["lon_bin"] = np.round(sig["lon"] / 1.0) * 1.0
    pts = []
    for (lat_bin, lon_bin), group in sig.groupby(["lat_bin", "lon_bin"]):
        last_eq = group.iloc[-1]
        time_since_last = (datetime.now() - last_eq["time"]).days / 365.25
        time_risk = min(1.0, time_since_last / time_window_years)
        mag_risk = min(1.0, (last_eq["magnitude"] - min_mag) / (8.0 - min_mag))
        freq_risk = min(1.0, len(group) / 20.0)
        risk_score = time_risk * 0.5 + mag_risk * 0.3 + freq_risk * 0.2
        threshold = getattr(st.session_state, "pred_risk_threshold", 0.3)
        if risk_score >= threshold:
            pts.append(
                {
                    "lat": lat_bin,
                    "lon": lon_bin,
                    "risk_score": risk_score,
                    "time_since_last": time_since_last,
                    "last_magnitude": last_eq["magnitude"],
                    "last_major_date": last_eq["time"].strftime("%Y-%m-%d"),
                    "event_count": len(group),
                    "region": f"{lat_bin:.1f}°N, {lon_bin:.1f}°E",
                }
            )
    return pd.DataFrame(pts)

# ------------------- Sidebar: Global Controls ----------------
with st.sidebar:
    st.header("🔎 Filters & Settings")

    # Navigation
    page = st.radio(
        "Navigation",
        ["🗺️ Map", "📊 Analysis", "📈 Stats", "🔍 Advanced", "🔭 Forecasting", "📖 Documentation"],
        index=0,
    )

    # Timeframe selector
    timeframe_options = list(FEEDS.keys()) + list(EXTENDED_TIMEFRAMES.keys())
    time_range = st.selectbox("Select timeframe:", timeframe_options, index=1)

    # Minimum magnitude fetch (for heavy ranges)
    min_mag_fetch = 0.1
    if time_range in TIMEFRAMES_NEEDING_MIN_MAG:
        min_mag_fetch = st.slider(
            "Minimum magnitude to fetch:", 0.0, 8.0, 4.5, 0.1, key="min_mag_fetch",
            help="Higher values improve performance on long timeframes",
        )

    # Custom date range
    custom_start = None
    custom_end = None
    if time_range == "Custom Range":
        c1, c2 = st.columns(2)
        with c1:
            default_start = datetime.now() - timedelta(days=30 * 365)
            custom_start = st.date_input(
                "Start date", value=default_start, max_value=datetime.now()
            )
        with c2:
            custom_end = st.date_input(
                "End date", value=datetime.now(), max_value=datetime.now()
            )
        if custom_start and custom_end:
            if (custom_end - custom_start).days > 365:
                min_mag_fetch = st.slider(
                    "Minimum magnitude to fetch:", 0.0, 8.0, 4.5, 0.1,
                    key="custom_min_mag_fetch",
                )

    # Depth filter
    depth_min, depth_max = st.slider(
        "Depth range (km):", 0.0, 700.0, (0.0, 300.0), help="Filter earthquakes by depth"
    )

    # Basemap & rendering options (OSM)
    basemap_name = st.selectbox("Basemap (OpenStreetMap):", list(OSM_STYLES.keys()), index=0)
    render_mode = st.radio(
        "Map render mode:",
        ["Dots", "Dots + AOI Circles", "Heatmap", "Hexagon"],
        index=1,
        help="Choose how earthquakes are visualized on the map."
    )
    heat_radius = 60 if render_mode == "Heatmap" else st.session_state.get("heat_radius", 60)
    if render_mode == "Heatmap":
        heat_radius = st.slider("Heatmap radius (pixels)", 10, 120, 60, 5)
        st.session_state.heat_radius = heat_radius
    if render_mode == "Hexagon":
        hex_radius_km = st.slider("Hexagon radius (km)", 10, 200, 50, 5)

# ------------------- Data Fetch & Prep -----------------------
with st.spinner("Fetching earthquake data..."):
    try:
        if time_range in FEEDS:
            payload = fetch_geojson(FEEDS[time_range])
        else:
            now = datetime.now()
            if time_range == "Custom Range" and custom_start and custom_end:
                start_date = datetime.combine(custom_start, datetime.min.time())
                end_date = datetime.combine(custom_end, datetime.min.time())
            elif time_range == "Past 3 Months":
                start_date, end_date = now - timedelta(days=90), now
            elif time_range == "Past Year":
                start_date, end_date = now - timedelta(days=365), now
            elif time_range == "Past 5 Years":
                start_date, end_date = now - timedelta(days=5 * 365), now
            elif time_range == "Past 10 Years":
                start_date, end_date = now - timedelta(days=10 * 365), now
            elif time_range == "Past 20 Years":
                start_date, end_date = now - timedelta(days=20 * 365), now
            elif time_range == "Past 50 Years":
                start_date, end_date = now - timedelta(days=50 * 365), now
            elif time_range == "20th Century (1900-1999)":
                start_date, end_date = datetime(1900, 1, 1), datetime(1999, 12, 31)
            elif time_range == "21st Century (2000-Present)":
                start_date, end_date = datetime(2000, 1, 1), now
            else:
                start_date, end_date = now - timedelta(days=365), now
            payload = fetch_usgs_range(start_date, end_date, float(min_mag_fetch))
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

features = payload.get("features", [])
rows = []
countries_set = set()
for q in features:
    coords = (q.get("geometry") or {}).get("coordinates") or [None, None, None]
    props = q.get("properties") or {}
    place = props.get("place") or ""
    country = extract_country(place)
    if country:
        countries_set.add(country)
    if coords[0] is not None and coords[1] is not None and props.get("mag") is not None:
        rows.append(
            {
                "lon": coords[0],
                "lat": coords[1],
                "depth_km": coords[2],
                "magnitude": props.get("mag"),
                "place": place,
                "country": country,
                "time": pd.to_datetime(props.get("time"), unit="ms", errors="coerce"),
                "url": props.get("url"),
                "id": q.get("id"),
                "significance": props.get("sig", 0),
                "tsunami": props.get("tsunami", 0),
                "alert": props.get("alert", ""),
            }
        )

df = pd.DataFrame(rows)
required_columns = ["lat", "lon", "magnitude"]
if not all(col in df.columns for col in required_columns):
    st.error("Missing required data columns. Please try a different timeframe.")
    st.stop()

# Depth filter
if "depth_km" in df.columns:
    df = df[(df["depth_km"] >= depth_min) & (df["depth_km"] <= depth_max)]

# Country filter
countries = sorted(list(countries_set))
countries.insert(0, "Worldwide")
if len(countries) > 10:
    selected_country = st.sidebar.selectbox(
        "Filter by country (searchable):", options=countries, index=0, help="Type to search"
    )
else:
    selected_country = st.sidebar.selectbox("Filter by country:", options=countries, index=0)

if selected_country != "Worldwide" and "country" in df.columns:
    df = df[df["country"].str.contains(selected_country, case=False, na=False)]

# Magnitude display slider (post-filter)
if not df.empty and "magnitude" in df.columns:
    mag_floor = float(max(df["magnitude"].min(), 0.0))
    min_mag_display = st.sidebar.slider(
        "Minimum magnitude to display:", mag_floor, 8.0, float(max(min_mag_fetch, mag_floor)), 0.1
    )
    df = df[df["magnitude"] >= min_mag_display].copy()
else:
    min_mag_display = float(min_mag_fetch)

# Timeframe info
if time_range in EXTENDED_TIMEFRAMES and time_range != "Custom Range":
    st.info(f"Displaying earthquakes from {time_range} with magnitude ≥ {min_mag_display}")

# -------------------------------------------------------------
# Reusable UI bits
# -------------------------------------------------------------
legend_html = """
<div style="position: relative;">
  <div style="position:absolute; right:16px; top:16px; background:rgba(0,0,0,0.65); padding:10px 12px; border-radius:10px; color:#fff; font-size:12px; z-index: 999;">
    <b>Legend</b><br>
    🔴 Shallow (0–70 km)<br>
    🟠 Intermediate (70–300 km)<br>
    🔵 Deep (>300 km)<br>
    ⭕ Effect area ∝ magnitude
  </div>
</div>
"""

def top_quakes_cards(dataframe: pd.DataFrame, k: int = 5):
    if dataframe.empty:
        st.info("No earthquake data available with current filters.")
        return
    st.subheader("🔥 Recent Major Earthquakes")
    cols = st.columns(k)
    topk = dataframe.sort_values("magnitude", ascending=False).head(k).copy()
    for i, (_, row) in enumerate(topk.iterrows()):
        with cols[i]:
            st.metric(
                label=f"🌍 {row['place'][:40]}…" if len(str(row['place'])) > 40 else f"🌍 {row['place']}",
                value=f"M {row['magnitude']:.1f}",
                delta=f"Depth {row['depth_km']:.0f} km • {row['time'].strftime('%Y-%m-%d')}",
            )

def depth_to_color(depth):
    if depth < 70:
        return [255, 0, 0, 160]  # Red for shallow
    elif depth < 300:
        return [255, 165, 0, 160]  # Orange for intermediate
    else:
        return [0, 0, 255, 160]  # Blue for deep

# -------------------------------------------------------------
# PAGES
# -------------------------------------------------------------

# ===================== 🗺️ MAP ===============================
if page == "🗺️ Map":
    st.subheader("🗺️ Interactive Map")
    top_quakes_cards(df, k=5)

    if not df.empty and {"lat", "lon"}.issubset(df.columns):
        avg_lat, avg_lon = df["lat"].mean(), df["lon"].mean()
        zoom_level = 2 if selected_country == "Worldwide" else 4
        df_map = df.copy()
        df_map["color"] = df_map["depth_km"].apply(depth_to_color)

        # Basemap (OpenStreetMap)
        basemap_layer = make_osm_tile_layer(basemap_name)

        # Data layers
        layers = [basemap_layer]

        if render_mode in ["Dots", "Dots + AOI Circles"]:
            # Precise location dots
            dot_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position=["lon", "lat"],
                get_radius=1000,
                get_fill_color="color",
                pickable=True,
                opacity=0.8,
            )
            layers.append(dot_layer)

            if render_mode == "Dots + AOI Circles":
                circle_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position=["lon", "lat"],
                    get_radius="magnitude * 30000",
                    get_fill_color="color",
                    pickable=False,
                    opacity=0.1,
                    stroked=True,
                    get_line_color="color",
                    get_line_width=100,
                )
                # Show circles beneath the dots
                layers = [basemap_layer, circle_layer, dot_layer]

        elif render_mode == "Heatmap":
            heat_layer = pdk.Layer(
                "HeatmapLayer",
                data=df_map,
                get_position=["lon", "lat"],
                get_weight="magnitude",
                radius_pixels=heat_radius,
                aggregation="MEAN",
            )
            layers.append(heat_layer)

        elif render_mode == "Hexagon":
            hex_layer = pdk.Layer(
                "HexagonLayer",
                data=df_map,
                get_position=["lon", "lat"],
                elevation_scale=20,
                extruded=True,
                coverage=0.9,
                radius=int(hex_radius_km * 1000),
                pickable=True,
            )
            layers.append(hex_layer)

        view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=zoom_level)
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": (
                    "<b>Magnitude:</b> {magnitude}<br/>"
                    "<b>Place:</b> {place}<br/>"
                    "<b>Time:</b> {time}<br/>"
                    "<b>Depth:</b> {depth_km} km<br/>"
                    "<b>Significance:</b> {significance}"
                )
            },
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.markdown(legend_html, unsafe_allow_html=True)

        # OSM attribution for compliance
        st.caption(f"Basemap: {OSM_STYLES[basemap_name]['attribution']}")

        st.write(
            f"Showing **{len(df)}** earthquakes (M≥{min_mag_display}) in **{selected_country}**"
        )

        # Table
        st.subheader("📋 Earthquake List")
        try:
            from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode  # type: ignore

            df_display = df.copy()
            if "time" in df_display.columns:
                df_display["time"] = pd.to_datetime(df_display["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            gb = GridOptionsBuilder.from_dataframe(
                df_display[["time", "magnitude", "depth_km", "place", "country"]]
            )
            gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=15)
            gb.configure_default_column(filter=True, sortable=True, resizable=True)
            grid_options = gb.build()
            AgGrid(
                df_display,
                gridOptions=grid_options,
                height=450,
                update_mode=GridUpdateMode.NO_UPDATE,
                enable_enterprise_modules=False,
                theme="streamlit",
            )
        except Exception:
            st.dataframe(
                df[["time", "magnitude", "depth_km", "place", "country"]]
                .sort_values(by="magnitude", ascending=False),
                use_container_width=True,
                height=450,
            )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "earthquakes.csv", "text/csv")
    else:
        st.info("No earthquake data available for mapping with current filters.")

# ===================== 📊 ANALYSIS ============================
elif page == "📊 Analysis":
    st.subheader("📊 Seismic Analysis Tools")
    if df.empty:
        st.info("No data available for analysis.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if "magnitude" in df.columns:
                fig_mag = px.histogram(df, x="magnitude", nbins=20, title="Magnitude Distribution")
                fig_mag.update_layout(xaxis_title="Magnitude", yaxis_title="Count")
                st.plotly_chart(fig_mag, use_container_width=True)
            if {"magnitude", "depth_km"}.issubset(df.columns):
                fig_sc = px.scatter(
                    df,
                    x="magnitude",
                    y="depth_km",
                    color="depth_km",
                    title="Depth vs Magnitude",
                    labels={"magnitude": "Magnitude", "depth_km": "Depth (km)"},
                )
                st.plotly_chart(fig_sc, use_container_width=True)
        with c2:
            if "time" in df.columns:
                dft = df.copy()
                dft["date"] = dft["time"].dt.date
                ts = dft.groupby("date").size().reset_index(name="count")
                fig_time = px.line(ts, x="date", y="count", title="Earthquakes Over Time")
                st.plotly_chart(fig_time, use_container_width=True)
            if "depth_km" in df.columns:
                fig_depth = px.histogram(df, x="depth_km", nbins=20, title="Depth Distribution")
                fig_depth.update_layout(xaxis_title="Depth (km)", yaxis_title="Count")
                st.plotly_chart(fig_depth, use_container_width=True)

# ===================== 📈 STATS ===============================
elif page == "📈 Stats":
    st.subheader("📈 Seismic Statistics")
    if df.empty:
        st.info("No data available for statistics.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Earthquakes", len(df))
            if "magnitude" in df.columns:
                st.metric("Average Magnitude", f"{df['magnitude'].mean():.2f}")
                st.metric("Magnitude Std Dev", f"{df['magnitude'].std():.2f}")
        with c2:
            if "magnitude" in df.columns:
                st.metric("Maximum Magnitude", f"{df['magnitude'].max():.1f}")
            if "depth_km" in df.columns:
                st.metric("Average Depth", f"{df['depth_km'].mean():.1f} km")
                st.metric("Depth Std Dev", f"{df['depth_km'].std():.1f} km")
        with c3:
            if "depth_km" in df.columns:
                st.metric("Minimum Depth", f"{df['depth_km'].min():.1f} km")
                st.metric("Maximum Depth", f"{df['depth_km'].max():.1f} km")
            if "tsunami" in df.columns:
                st.metric("Tsunami Events", int(df["tsunami"].sum()))

        if "time" in df.columns:
            df_year = df.copy()
            df_year["year"] = df_year["time"].dt.year
            year_counts = df_year["year"].value_counts().sort_index()
            st.bar_chart(year_counts)

        if {"magnitude", "depth_km"}.issubset(df.columns):
            corr = df["magnitude"].corr(df["depth_km"])
            st.metric("Magnitude-Depth Correlation", f"{corr:.3f}")

# ===================== 🔍 ADVANCED ============================
elif page == "🔍 Advanced":
    st.subheader("🔍 Advanced Geophysical Features")
    if df.empty:
        st.info("No data available for advanced analysis.")
    else:
        with st.expander("🧭 Seismic Gap Analysis", expanded=False):
            st.info(
                "Identifies regions lacking recent significant activity compared to historical patterns."
            )
            if st.button("Run Seismic Gap Analysis", key="gap_analysis"):
                gaps = calculate_seismic_gaps(df, min_mag=6.0, time_window_years=10)
                if len(gaps) > 0:
                    st.success(f"Found {len(gaps)} potential seismic gaps")
                    st.dataframe(gaps)
                else:
                    st.info("No significant seismic gaps identified with current parameters")

        with st.expander("🧩 Foreshock–Mainshock Sequences", expanded=False):
            st.info("Detect potential foreshock–mainshock clustering in space and time.")
            if st.button("Identify Earthquake Sequences", key="foreshock_analysis"):
                sequences = identify_foreshock_sequences(df, time_window_hours=72, distance_km=100)
                if len(sequences) > 0:
                    st.success(f"Found {len(sequences)} potential earthquake sequences")
                    st.dataframe(sequences)
                else:
                    st.info("No significant earthquake sequences identified")

        with st.expander("🌐 Ground Motion & Liquefaction", expanded=False):
            st.info("Estimate shaking intensity and basic liquefaction potential.")
            gc1, gc2 = st.columns(2)
            with gc1:
                gm_magnitude = st.number_input("Magnitude", min_value=3.0, max_value=9.0, value=6.0, step=0.1)
                gm_distance = st.number_input("Distance (km)", min_value=1, max_value=500, value=50)
            with gc2:
                gm_site_class = st.selectbox("Site Class", options=["A", "B", "C", "D"], index=1)
            pga = estimate_ground_motion(gm_magnitude, gm_distance, gm_site_class)
            st.metric("Estimated Peak Ground Acceleration", f"{pga} g")

            lc1, lc2 = st.columns(2)
            with lc1:
                liq_pga = st.number_input("PGA (g)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
                liq_groundwater = st.number_input("Groundwater Depth (m)", min_value=0, max_value=10, value=2)
            with lc2:
                liq_fines = st.number_input("Fines Content (%)", min_value=0, max_value=50, value=15)
            liquefaction_risk = assess_liquefaction_potential(liq_pga, liq_groundwater, liq_fines)
            st.metric("Liquefaction Potential", liquefaction_risk)

# ===================== 🔮 FORECASTING =========================
elif page == "🔭 Forecasting":
    st.subheader("🔭 Seismic Forecasting & Prediction")
    if len(df) < 50:
        st.warning("Insufficient data for reliable forecasting. Please select a longer timeframe with more earthquakes.")
    else:
        # Summary cards
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Dataset Days", f"{(df['time'].max() - df['time'].min()).days}")
        with s2:
            st.metric("Mean Daily Count", f"{df.set_index('time').resample('1D').size().mean():.2f}")
        with s3:
            st.metric("Max Magnitude", f"{df['magnitude'].max():.1f}")

        st.subheader("📈 Activity Forecast")
        dff = df.copy()
        dff["date"] = dff["time"].dt.date
        daily_counts = dff.groupby("date").size().reset_index(name="count")
        daily_counts["date"] = pd.to_datetime(daily_counts["date"])
        daily_counts = daily_counts.set_index("date")
        forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30, 7)
        ma_7 = daily_counts["count"].rolling(window=7).mean()
        ma_30 = daily_counts["count"].rolling(window=30).mean()
        fig_forecast = go.Figure()
        fig_forecast.add_trace(
            go.Scatter(x=daily_counts.index, y=daily_counts["count"], name="Daily Count", mode="markers", marker=dict(size=4, opacity=0.6))
        )
        fig_forecast.add_trace(go.Scatter(x=ma_7.index, y=ma_7, name="7-Day MA"))
        fig_forecast.add_trace(go.Scatter(x=ma_30.index, y=ma_30, name="30-Day MA"))
        last_date = daily_counts.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq="D")[1:]
        recent_avg = daily_counts["count"].tail(30).mean()
        forecast_values = [recent_avg] * forecast_days
        fig_forecast.add_trace(
            go.Scatter(x=forecast_dates, y=forecast_values, name=f"{forecast_days}-Day Forecast", line=dict(dash="dash"), fill="tonexty")
        )
        fig_forecast.update_layout(title=f"Earthquake Activity Forecast ({forecast_days} days)", xaxis_title="Date", yaxis_title="Daily Earthquakes", hovermode="x unified")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("🗺️ Predictive Risk Map")
        pred = generate_predictive_data(df)
        if not pred.empty:
            avg_lat, avg_lon = pred["lat"].mean(), pred["lon"].mean()

            def risk_to_color(r):
                if r < 0.3:
                    return [128, 0, 128, 100]
                elif r < 0.6:
                    return [255, 0, 255, 150]
                else:
                    return [255, 105, 180, 200]

            pred = pred.copy()
            pred["color"] = pred["risk_score"].apply(risk_to_color)

            # Basemap (OpenStreetMap)
            basemap_layer = make_osm_tile_layer(basemap_name)

            predictive_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pred,
                get_position=["lon", "lat"],
                get_radius="risk_score * 50000",
                get_fill_color="color",
                get_line_color=[255, 255, 255, 200],
                get_line_width=50,
                pickable=True,
                opacity=0.7,
                stroked=True,
                filled=True,
                transitions={
                    "get_radius": {"type": "interpolation", "duration": 2000},
                    "get_fill_color": {"type": "interpolation", "duration": 2000},
                },
            )
            historical_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["lon", "lat"],
                get_radius=20000,
                get_fill_color=[200, 200, 200, 50],
                pickable=False,
                opacity=0.3,
            )
            view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=2)
            deck = pdk.Deck(
                layers=[basemap_layer, historical_layer, predictive_layer],
                initial_view_state=view_state,
                tooltip={
                    "html": (
                        "<b>Risk Score:</b> {risk_score:.2f}<br/>"
                        "<b>Region:</b> {region}<br/>"
                        "<b>Last Major EQ:</b> {last_major_date}<br/>"
                        "<b>Time Since Last:</b> {time_since_last:.1f} years"
                    )
                },
            )
            st.pydeck_chart(deck, use_container_width=True)
            st.caption(f"Basemap: {OSM_STYLES[basemap_name]['attribution']}")
            st.markdown(
                """
                **Predictive Map Legend**<br>
                <span style="color:#800080;">&#9679;</span> Low (0.0–0.3) • 
                <span style="color:#FF00FF;">&#9679;</span> Medium (0.3–0.6) • 
                <span style="color:#FF69B4;">&#9679;</span> High (0.6–1.0) • 
                <span style="color:#D3D3D3;">&#9679;</span> Historical earthquake locations
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No predictive points above threshold. Adjust parameters in the sidebar.")

        st.subheader("📊 Magnitude Probability (Gutenberg–Richter)")
        if "magnitude" in df.columns:
            mags = df["magnitude"].dropna()
            if len(mags) > 0:
                min_mag = mags.min()
                max_mag = mags.max()
                mag_bins = np.arange(math.floor(min_mag), math.ceil(max_mag) + 0.5, 0.5)
                hist, bin_edges = np.histogram(mags, bins=mag_bins)
                non_zero = hist > 0
                if np.any(non_zero):
                    log_counts = np.log10(hist[non_zero])
                    mag_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    mag_centers = mag_centers[non_zero]
                    slope, intercept = np.polyfit(mag_centers, log_counts, 1)
                    b_value = -slope
                    a_value = 10 ** intercept
                    st.metric("Estimated b-value", f"{b_value:.2f}")
                    st.metric("a-value", f"{a_value:.2f}")
                    prob_m6 = 10 ** (a_value - b_value * 6.0) / len(mags) * 100
                    prob_m7 = 10 ** (a_value - b_value * 7.0) / len(mags) * 100
                    st.metric("Probability of M6+ in dataset", f"{min(prob_m6, 100):.1f}%")
                    st.metric("Probability of M7+ in dataset", f"{min(prob_m7, 100):.1f}%")

        st.divider()
        st.subheader("⚙️ Forecast Parameters")
        min_mag_pred = st.slider("Minimum magnitude for prediction", 4.0, 7.0, 5.0, 0.1)
        time_window = st.slider("Seismic gap time window (years)", 5, 50, 10, 1)
        risk_threshold = st.slider("Risk threshold for display", 0.1, 0.9, 0.3, 0.1)
        if st.button("🔄 Update Predictive Parameters", key="generate_prediction"):
            st.session_state.pred_min_mag = min_mag_pred
            st.session_state.pred_time_window = time_window
            st.session_state.pred_risk_threshold = risk_threshold
            st.success("Predictive parameters updated — refresh the Predictive Risk Map above.")

        st.warning(
            "**Disclaimer**: Forecasts are statistical estimates and must not be used for emergency planning. Refer to official seismic hazard assessments."
        )

# ===================== 📖 DOCUMENTATION =========================
elif page == "📖 Documentation":
    st.subheader("📖 Documentation & User Guide")

    with st.expander("🌍 Overview & Purpose", expanded=True):
        st.markdown("""
**SeismoScope** helps you **monitor, explore, and analyze** global seismicity using **USGS** feeds and custom time ranges.  
It offers **interactive mapping**, **analytical charts**, **geophysical utilities**, and **lightweight forecasting** intended for **education, research triage, and situational awareness** (not for emergency planning).
        """)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Primary Users**  \nResearchers, students, urban planners, educators")
        with c2:
            st.markdown("**Data Source**  \nUSGS Earthquake API (GeoJSON) + OpenStreetMap basemaps")
        with c3:
            st.markdown("**Key Pages**  \n🗺️ Map • 📊 Analysis • 📈 Stats • 🔍 Advanced • 🔭 Forecasting")

    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
1. **Pick a timeframe** in the sidebar (e.g., *Past Day*, *Past 5 Years*, or **Custom Range**).  
2. **Fine-tune filters**: depth range, country, and **Minimum magnitude to display**.  
3. In **🗺️ Map**, choose **render mode**: *Dots*, *Dots + AOI Circles*, *Heatmap*, or *Hexagon*.  
4. Review **top events** cards, hover points for details, sort the table, and **Download CSV**.  
5. Explore **📊 Analysis** for distributions/trends and **📈 Stats** for summary metrics.  
6. In **🔍 Advanced**, run **Seismic Gaps**, **Foreshock Sequences**, and **Ground Motion/Liquefaction**.  
7. In **🔭 Forecasting**, view the **activity forecast**, **Predictive Risk Map**, and **Gutenberg–Richter** estimates.
        """)

    with st.expander("🗺️ Interpreting the Map", expanded=False):
        st.markdown("""
**Layers & Modes**
- **Dots**: each point = an event; color reflects depth:  
  • 🔴 **Shallow** (0–70 km) • 🟠 **Intermediate** (70–300 km) • 🔵 **Deep** (>300 km)  
- **Dots + AOI Circles**: adds a soft circle per event with radius proportional to magnitude  
  (`radius ≈ magnitude × 30 km` for visualization).
- **Heatmap**: density + magnitude weighting (`radius_pixels` configurable).
- **Hexagon**: spatial aggregation (3D columns) over a hex grid (`radius` in km).

**Basemaps (OSM)**  
- **Standard**: general purpose.  
- **Humanitarian (HOT)**: emphasizes populated/critical features.  
- **OpenTopoMap**: terrain/topography context.

**Tooltips**  
Hover to see **Magnitude, Place, Time, Depth, Significance**.  
Use filters to refine; the **caption** shows the current count and country filter.

**Table**  
Sort/filter columns (time, magnitude, depth, place, country). Use the **CSV download** for offline work.
        """)

    with st.expander("🧪 Data Collection & Filters", expanded=False):
        st.markdown("""
**Feeds**  
- Live summaries: **Past Hour/Day/Week/Month** (USGS GeoJSON).  
- Extended ranges (e.g., **Past 5–50 Years**, **20th/21st Century**, **Custom Range**) query the USGS FDSN service.

**Performance**  
- For very long ranges, a **Minimum magnitude to fetch** slider appears to limit results (cap ~20,000 events/query).
- **Caching** (`st.cache_data`) reduces repeated fetches.

**Filters**  
- **Depth (km)**: restricts crustal vs intermediate vs deep-focus events.  
- **Country**: simple heuristic from the `place` string (may be approximate).  
- **Minimum magnitude to display**: post-filter visibility threshold.
        """)

    with st.expander("🔍 Advanced Analysis — Methods", expanded=False):
        st.markdown("#### 1) Seismic Gap Analysis")
        st.markdown("""
Groups events into ~**2° × 2°** bins using significant quakes (default **M ≥ 6.0**).  
For each bin, compute **time since last significant event** and flag **gaps** where  
`time_since_last > time_window_years` (default **10 years**).  
**Use cases**: identify regions with historically active seismicity lacking recent large events.  
**Caveats**: binning choice, catalog completeness, and reporting thresholds affect results.
        """)

        st.markdown("#### 2) Foreshock–Mainshock Sequences")
        st.markdown("""
For each potential **mainshock** (default **M ≥ 4.0**), look **backward** **time_window_hours** (e.g., **72 h**)  
and within **distance_km** (e.g., **100 km**) to count potential **foreshocks**.  
Outputs: **foreshock count**, **largest foreshock**, **time span** before mainshock.  
**Caveats**: purely empirical; true foreshocks are only known **post-hoc** and depend on window choices.
        """)

        st.markdown("#### 3) Ground Motion & Liquefaction (Simplified)")
        st.markdown("""
**Peak Ground Acceleration (PGA)** (toy model):
""")
        st.latex(r"\mathrm{PGA} = \exp\!\big( -2.0 + 0.6M - \ln(r+10) - 0.003\,r \big)\times S")
        st.markdown("""
where **M** = magnitude, **r** = hypocentral distance (km), and **S** = site factor  
(A=0.8, B=1.0, C=1.2, D=1.4). Output is **g** (approx.).  
**Liquefaction** uses a simple index from **PGA**, **groundwater depth**, and **fines content** to classify  
**Very Low → Very High**.  
**Caveats**: These are coarse, educational estimates — **not** code-compliant hazard calculations.
        """)

    with st.expander("🔭 Forecasting — Models & Assumptions", expanded=False):
        st.markdown("#### A) Activity Forecast (Counts)")
        st.markdown("""
Daily event counts are smoothed with **7-day** and **30-day** moving averages.  
A naive short-term forecast extrapolates the recent mean over the chosen horizon.  
**Goal**: visualize short-term trends; **not** a predictive seismic hazard model.
        """)

        st.markdown("#### B) Predictive Risk Map (Heuristic Score)")
        st.markdown("""
We first group **significant events** (default **M ≥ `pred_min_mag`**) into **1° × 1°** bins.  
For each bin, we compute a **risk score** combining:
- **Time component**: `time_risk = min(1, time_since_last / pred_time_window)`  
- **Magnitude component**: `mag_risk = min(1, (last_mag - pred_min_mag) / (8 - pred_min_mag))`  
- **Frequency component**: `freq_risk = min(1, count / 20)`
- **Weighted sum**:  
""")
        st.code("risk_score = 0.5 * time_risk + 0.3 * mag_risk + 0.2 * freq_risk")
        st.markdown("""
Only bins with `risk_score ≥ pred_risk_threshold` are shown.  
**Color legend**: low → medium → high risk bubbles; historical events shown in light grey.  
**Caveats**: Heuristic, assumes stationarity and catalog completeness; **do not** interpret as deterministic risk.
        """)

        st.markdown("#### C) Gutenberg–Richter (Magnitude–Frequency)")
        st.latex(r"\log_{10} N(M \ge m) = a - b\,m")
        st.markdown("""
- We bin magnitudes (0.5 units), compute counts, and fit a **line** to `log10(count)` vs **magnitude**.  
- **b-value** ≈ **−slope** (typical values near **1.0** for many regions).  
- We display **N₀ = 10^a** (derived from the fitted intercept) for interpretability.  
- Simple probabilities (e.g., **M6+**, **M7+**) are estimated from the fitted curve relative to dataset size.  
**Caveats**: Requires **magnitude of completeness**; mixing magnitude types (Mw/Ml/mb) and truncation bias the fit.
        """)

    with st.expander("📈 Reading Charts & Stats", expanded=False):
        st.markdown("""
- **Magnitude/Depth histograms**: highlight distribution and tails.  
- **Depth vs Magnitude scatter**: look for clusters (e.g., Wadati–Benioff zones).  
- **Time series (counts)**: check for swarms or quiet periods.  
- **Stats page**: totals, averages, extrema, year counts, and **corr(magnitude, depth)** (weak by design in many regions).
        """)

    with st.expander("⚙️ Performance Tips & Limits", expanded=False):
        st.markdown("""
- Prefer **Past Week/Month** for responsive maps; use **Min magnitude to fetch** for long windows.  
- **Hexagon** mode scales better for dense datasets; **Heatmap radius** smaller for local, larger for global views.  
- Caching is enabled; repeated queries with identical params load faster.  
- USGS API may cap results; very long queries can be truncated.
        """)

    with st.expander("❓ FAQ & Troubleshooting", expanded=False):
        st.markdown("""
**No points on map?**  
• You may have filtered them out (country/depth/min magnitude). Reset filters or widen timeframe.

**Times look off?**  
• USGS timestamps are UTC; your browser renders local time in tooltips where applicable.

**Why is a country missing?**  
• Country parsing is heuristic from the `place` text. Use **Worldwide** or filter by keyword in the table.

**Are forecasts “predictions”?**  
• No. They are **statistical/heuristic visualizations** for education and exploration only.
        """)

    with st.expander("🔒 Notes, Attribution & Version", expanded=False):
        st.markdown("""
- **Data**: © USGS Earthquake Hazards Program.  
- **Basemap**: © OpenStreetMap contributors; HOT & OpenTopoMap where selected.  
- **Author**: Zakaria Bouidane.  
- **License/Use**: Educational/analytical; no operational decision-making.  
- **Disclaimer**: This app **must not** be used for emergency planning or life-safety decisions.
        """)

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
current_year = datetime.now().year

st.caption(
    f"Developed by Zakaria Bouidane. © {current_year} All rights reserved."
)
