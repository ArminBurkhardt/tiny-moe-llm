import streamlit as st
import json
import os
import pandas as pd
from typing import List, Dict

st.set_page_config(page_title="Training Metrics Dashboard", layout="wide")

st.title("Tiny-MOE-LLM Training Dashboard")

# --- Configuration ---
LOG_FILE = "ckpts/trained/pretrain/pretrain_metrics.jsonl"
CONTROL_FILE = "ckpts/trained/pretrain/control.json"

st.sidebar.header("Configuration")
log_path_input = st.sidebar.text_input("Path to metrics log file", LOG_FILE)
control_path_input = st.sidebar.text_input("Path to control file", CONTROL_FILE)

# --- Control Logic ---
st.sidebar.header("Controls")
is_paused = False
if os.path.exists(control_path_input):
    try:
        with open(control_path_input, "r") as f:
            data = json.load(f)
            is_paused = data.get("pause", False)
    except Exception:
        pass

if st.sidebar.button("Pause Training" if not is_paused else "Resume Training"):
    new_state = not is_paused
    with open(control_path_input, "w") as f:
        json.dump({"pause": new_state}, f)
    st.sidebar.success(f"Sent {'PAUSE' if new_state else 'RESUME'} signal!")
    is_paused = new_state

st.sidebar.markdown(f"**Current Status:** {'⏸ PAUSED' if is_paused else '▶ RUNNING'}")

# --- Auto Refresh ---
import streamlit_autorefresh
streamlit_autorefresh.st_autorefresh(interval=5000, limit=None, key="data_refresh")

# --- Data Loading ---
@st.cache_data(ttl=2)
def load_data(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

records = load_data(log_path_input)

if not records:
    st.warning(f"No data found in {log_path_input}. Waiting for training to start...")
    st.stop()

# Parse metrics
df = pd.DataFrame(records)
latest = df.iloc[-1]

st.header("Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Progress (Step)", f"{latest.get('step', -1)}")
col2.metric("Total Loss", f"{latest.get('total_loss', 0.0):.4f}")
col3.metric("LM Loss", f"{latest.get('lm_loss', 0.0):.4f}")
col4.metric("Dataset", f"{latest.get('dataset_name', 'Unknown')}")
col5.metric("Batch Similarity", f"{latest.get('similarity', 0.0):.4f}")

# Layout for charts
st.header("Loss Curves")
st.line_chart(df.set_index("step")[["total_loss", "lm_loss", "router_loss"]])

col_ex1, col_ex2 = st.columns(2)

with col_ex1:
    st.header("Expert Usage")
    st.line_chart(df.set_index("step")[["num_experts"]])

with col_ex2:
    st.header("Learning Rate")
    if "learning_rate" in df.columns:
         st.line_chart(df.set_index("step")[["learning_rate"]])

st.header("Expert Usage Analytics")
# Flatten expert_stats
expert_stats_df = []
for r in records:
    stats = r.get("expert_stats", {})
    row = {"step": r.get("step")}
    for k, v in stats.items():
        row[k] = v
    expert_stats_df.append(row)

ex_df = pd.DataFrame(expert_stats_df).set_index("step")
if not ex_df.empty and len(ex_df.columns) > 0:
    st.line_chart(ex_df)
else:
    st.info("No expert usage stats available yet.")


# streamlit run webui.py