import streamlit as st
from src.agent.agent_core import ManufacturingAgent
import pandas as pd
import json
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="GenAI Manufacturing Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom Tab Styling (optional)
# -----------------------------
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        padding: 10px;
        font-weight: bold;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #d0ebff;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 GenAI Manufacturing Assistant")
st.markdown("---")

# -----------------------------
# Initialize Agent
# -----------------------------
if "agent" not in st.session_state:
    st.session_state.agent = ManufacturingAgent()
agent = st.session_state.agent

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "💬 Q&A + Maintenance",
    "📊 Simulation",
    "📈 Trend Forecast"
])

# =============================
# TAB 1: Q&A and Maintenance
# =============================
with tab1:
    col1, col2 = st.columns(2)

    # -- Left: Chat/Q&A
    with col1:
        st.subheader("💬 Ask about defects or maintenance logs")
        question = st.text_area("Enter your question", height=120, placeholder="Type your question here...")

        if st.button("Submit Question"):
            if question.strip():
                with st.spinner("Processing your question..."):
                    try:
                        response = agent.ask(question)
                        st.markdown("### Answer:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error: {e}")

    # -- Right: Predictive Maintenance with File Upload
    with col2:
        st.subheader("🛠 Predictive Maintenance Dashboard")
        uploaded_file = st.file_uploader("Upload current machine state CSV", type=["csv"])

        if uploaded_file and st.button("Run Predictive Maintenance"):
            try:
                df = pd.read_csv(uploaded_file)
                df_defects = df[df["defect_flag"] == 1]
                df_top5 = df_defects.sort_values("timestamp", ascending=False).head(5)

                current_states = [
                    {
                        "machine_id": row["machine_id"],
                        "sensor_1_temp": row["sensor_1_temp"],
                        "sensor_2_vibration": row["sensor_2_vibration"],
                        "product_id": row["product_id"],
                        "timestamp": str(row["timestamp"])
                    }
                    for _, row in df_top5.iterrows()
                ]

                results_json = agent.predictive_maintainace(current_states)
                data = json.loads(results_json)

                machines = data.get("top_machines", [])
                if machines:
                    df_display = pd.DataFrame([{
                        "Machine ID": m.get("machine_id", "N/A"),
                        "Urgency Score": m.get("urgency_score", 0),
                        "Reported Issues": ", ".join(m.get("reported_issues", []))
                    } for m in machines])


                    def highlight_red(val):
                        return "color: red; font-weight: bold"


                    st.dataframe(df_display.style.applymap(highlight_red).set_properties(**{
                        "text-align": "center",
                        "font-size": "14px"
                    }))
                else:
                    st.info("No machines currently require urgent maintenance.")

            except Exception as e:
                st.error(f"Error processing data: {e}")

# =============================
# TAB 2: Simulation
# =============================
with tab2:
    st.subheader("📊 Run Simulation Scenario")
    simulation_text = st.text_area("Enter your simulation scenario", height=120,
                                   placeholder="Provide scenario data or instructions...")

    if st.button("Run Simulation"):
        if simulation_text.strip():
            with st.spinner("Generating simulation..."):
                try:
                    response = agent.ask(simulation_text)
                    df_sim = pd.read_csv(StringIO(response))
                    st.dataframe(df_sim, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing simulation: {e}")

# =============================
# TAB 3: Trend Forecasting
# =============================

with tab3:
    st.subheader("📈 Trend Forecasting")

    trend_target = st.selectbox(
        "Select the parameter to forecast: I will forecast a given sensor/production metric using historical data.",
        options=["sensor_1_temp", "sensor_2_vibration", "output_qty"],
        index=0
    )

    if st.button("Show Trend Forecast"):
        with st.spinner("Generating forecast..."):
            try:
                # Call agent to get trend forecast
                forecast_result_1 = agent.trend({
                    "task": "forecast_trend",
                    "target_column": trend_target,
                    "forecast_steps": 10,
                    "lag": 3
                })

                forecast_json_str = forecast_result_1.get("output", "")
                if isinstance(forecast_json_str, str):
                    forecast_data = json.loads(forecast_json_str)

                    print(">>>>>>>>>>>>>>>>", forecast_data)
                    df = pd.DataFrame(forecast_data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    # Define number of forecast points (last N points)
                    forecast_points = min(5, len(df))

                    # Split history and forecast
                    history = df.iloc[:-forecast_points]
                    forecast = df.iloc[-forecast_points:]

                    # -------------------------
                    # Plot 1: History
                    # -------------------------
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                    ax_hist.plot(history["timestamp"], history[trend_target], label="Actual History",
                                 linewidth=2, color="blue")
                    ax_hist.set_title(f"📊 Historical {trend_target}", fontsize=14)
                    ax_hist.set_xlabel("Timestamp")
                    ax_hist.set_ylabel(trend_target)
                    ax_hist.legend()
                    ax_hist.grid(True)
                    st.pyplot(fig_hist)

                    # -------------------------
                    # Plot 2: Forecast
                    # -------------------------
                    fig_forecast, ax_forecast = plt.subplots(figsize=(10, 4))
                    ax_forecast.plot(forecast["timestamp"], forecast["forecast"], label="Forecast",
                                     linestyle="--", linewidth=2, color="orange")

                    # Forecast start marker
                    forecast_start_time = forecast["timestamp"].iloc[0]
                    ax_forecast.axvline(forecast_start_time, color="red", linestyle="--", label="Forecast Start")
                    ax_forecast.text(forecast_start_time, ax_forecast.get_ylim()[1] * 0.95,
                                     "Forecast Start", rotation=90, color="red", verticalalignment="top", fontsize=10)

                    ax_forecast.set_title(f"📈 Forecasted {trend_target}", fontsize=14)
                    ax_forecast.set_xlabel("Timestamp")
                    ax_forecast.set_ylabel(trend_target)
                    ax_forecast.legend()
                    ax_forecast.grid(True)
                    st.pyplot(fig_forecast)
                else:
                    st.error(f"Unexpected forecast result type: {type(forecast_json_str)}")

            except Exception as e:
                st.error(f"Error generating forecast: {e}")
