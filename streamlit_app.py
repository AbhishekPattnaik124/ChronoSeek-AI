import streamlit as st
import requests
import json
import os
import pandas as pd
from datetime import datetime
from PIL import Image

# --- Configuration ---
API_BASE = "http://localhost:8000"
HISTORY_FILE = "query_history.json"
st.set_page_config(page_title="ChronoSeek AI", layout="wide", page_icon="🔍")

# --- Custom CSS ---
st.markdown("""
<style>
    .result-card {
        border-radius: 10px;
        padding: 15px;
        background-color: #1e1e1e;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    .top-result {
        border: 2px solid #ffd700 !important;
        box-shadow: 0px 0px 15px rgba(255, 215, 0, 0.3);
    }
    .timestamp {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00d4ff;
    }
    .confidence-badge {
        padding: 4px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .high { background-color: #28a745; color: white; }
    .medium { background-color: #ffc107; color: black; }
    .low { background-color: #6c757d; color: white; }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            st.session_state.history = json.load(f)
    else:
        st.session_state.history = []

def save_history(query):
    if query not in st.session_state.history:
        st.session_state.history.insert(0, query)
        st.session_state.history = st.session_state.history[:10]
        with open(HISTORY_FILE, "w") as f:
            json.dump(st.session_state.history, f)

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio("Search Mode", ["Simple", "Smart (AI decomposed)"])
index_mode = st.sidebar.selectbox("Index Mode", ["auto", "frame", "temporal"], 
                                 help="Temporal mode uses context fusion for better semantic accuracy.")
top_k = st.sidebar.slider("Top-K Results", 5, 50, 10)

st.sidebar.subheader("⏳ Temporal Filter")
t_start = st.sidebar.text_input("Start Time", "00:00:00")
t_end = st.sidebar.text_input("End Time", "23:59:59")

rerank = st.sidebar.toggle("Re-rank Results", value=False)

health = {}
# Stats Section
try:
    health = requests.get(f"{API_BASE}/health").json()
    st.sidebar.success(f"Backend: Online ({health.get('device', 'Unknown').upper()})")
    
    # Mock/Read from metadata
    if os.path.exists("index_metadata.json"):
        with open("index_metadata.json", "r") as f:
            meta_count = len(json.load(f))
        st.sidebar.metric("Frames Indexed", meta_count)
except:
    st.sidebar.error("Backend: Offline")

# History Section
st.sidebar.subheader("📜 Query History")
for h_query in st.session_state.history:
    if st.sidebar.button(h_query, key=f"hist_{h_query}"):
        st.session_state.search_query = h_query

# --- Main Area ---
st.title("🔍 ChronoSeek AI")

query = st.text_input("What are you looking for?", 
                      value=st.session_state.get("search_query", ""),
                      placeholder="e.g. red car near entrance",
                      key="main_search")

col1, col2, _ = st.columns([1,1,4])
search_triggered = col1.button("Search", type="primary", width="stretch")

if search_triggered and query:
    save_history(query)
    
    endpoint = "/search" if mode == "Simple" else "/smart-search"
    payload = {
        "query": query,
        "top_k": top_k,
        "start_time": t_start,
        "end_time": t_end,
        "rerank": rerank,
        "mode": index_mode
    }

    with st.spinner("Analyzing video frames..."):
        try:
            response = requests.post(f"{API_BASE}{endpoint}", json=payload)
            if response.status_code == 200:
                data = response.json()
                results = data["results"]
                latency = data["query_latency_ms"]
                
                # Smart mode breakdown
                if mode == "Smart (AI decomposed)":
                    with st.expander("📝 Query Breakdown", expanded=True):
                        st.write(f"Decomposing: *{query}*")
                        # Real breakdown would come from API, here we show the logic
                        words = query.split()
                        st.info("Searching for visual tokens: " + ", ".join([f"`{w}`" for w in words if len(w)>3]))

                if not results:
                    st.warning("No results found. Try adjusting the query or time range.")
                    st.info("💡 Try: 'red color', 'scene changes', or broader terms.")
                else:
                    st.write(f"Found {len(results)} results in {latency}ms (Index Mode: {data.get('mode_used', 'unknown')})")
                    
                    # Results Grid
                    cols = st.columns(3)
                    for i, res in enumerate(results):
                        with cols[i % 3]:
                            is_top = i == 0
                            card_class = "result-card top-result" if is_top else "result-card"
                            
                            score = res['score']
                            if score > 0.35: # CLIP typical scores for small datasets
                                badge_class = "high"
                                badge_text = "High Confidence"
                            elif score > 0.25:
                                badge_class = "medium"
                                badge_text = "Medium"
                            else:
                                badge_class = "low"
                                badge_text = "Low"

                            # HTML Card
                            st.markdown(f"""
                            <div class="{card_class}">
                                <div class="timestamp">{res['timestamp']}</div>
                                <div style="margin-bottom:10px;">
                                    <span class="confidence-badge {badge_class}">{badge_text}</span>
                                    <span style="font-size:0.8rem; color:#888;">{res['video_name']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Image
                            if os.path.exists(res['frame_path']):
                                st.image(res['frame_path'], width="stretch")
                            else:
                                st.error("Image missing")

                            # Score bar
                            st.progress(min(max(float(score) * 2, 0.0), 1.0)) # Scaled for visibility
                            st.caption(f"Relevance Score: {score:.4f}")
                            
                            # Action
                            ffmpeg_cmd = f"ffmpeg -ss {res['timestamp']} -i {res['video_name']}.mp4 -t 5 out.mp4"
                            st.button("Jump to Timestamp", key=f"jump_{i}", 
                                      help=f"Copies: {ffmpeg_cmd}")

            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            if st.button("Retry Connection"):
                st.rerun()

# --- Benchmark Panel ---
st.markdown("---")
with st.expander("📊 System Benchmark & Stats"):
    col_bench_1, col_bench_2 = st.columns(2)
    
    if col_bench_1.button("Run Full System Benchmark"):
        # This would normally take long, we show existing report if avail
        try:
            bench_data = requests.get(f"{API_BASE}/stats/benchmark").json()
            if "error" not in bench_data:
                st.json(bench_data)
            else:
                st.info("No benchmark report found. Please run `python benchmark.py` from terminal.")
        except:
            st.error("Could not fetch benchmark data.")

    if st.session_state.get("last_latency"):
        col_bench_2.metric("Last Query Latency", f"{st.session_state.last_latency} ms")
    
    st.write("**Hardware Info:** " + health.get("device", "Unknown").upper())
