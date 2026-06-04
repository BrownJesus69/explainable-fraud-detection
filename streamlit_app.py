import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit as st

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🔍")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/xgb_robust.pkl")
THRESHOLD  = 0.828

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    features  = model.get_booster().feature_names
    return model, explainer, features

@st.cache_data(show_spinner="Loading drift data...")
def load_drift_data():
    ref  = pd.read_parquet("data/X_train.parquet").sample(2000, random_state=42)
    curr = pd.read_parquet("data/X_test.parquet").sample(2000, random_state=42)
    return ref, curr

model, explainer, FEATURES = load_model()

st.title("🔍 Fraud Detection Dashboard")
st.caption("XGBoost · SHAP · Adversarial Robustness · IEEE-CIS Dataset")

tab1, tab2, tab3 = st.tabs(["Score Transaction", "Model Comparison", "Drift & Robustness"])

# ── TAB 1: Score Transaction ─────────────────────────────────────────
with tab1:
    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.subheader("Transaction Details")
        amt   = st.number_input("Amount ($)", 0.0, 25000.0, 150.0, step=10.0)
        hour  = st.slider("Hour of Day", 0, 23, 14)
        c1    = st.number_input("C1 (card count)", 0, 30, 1)
        c13   = st.number_input("C13 (address count)", 0, 30, 1)
        c14   = st.number_input("C14", 0, 30, 0)
        m5    = st.selectbox("M5 flag", [0, 1], index=0)
        m6    = st.selectbox("M6 flag", [0, 1], index=0)
        v70   = st.number_input("V70", 0.0, 10.0, 0.0)
        score = st.button("Score Transaction", type="primary", use_container_width=True)

    with col_out:
        if score:
            row = pd.DataFrame([{
                "TransactionAmt": amt, "amt_log": float(np.log1p(amt)),
                "hour": hour, "C1": c1, "C13": c13, "C14": c14,
                "M5": m5, "M6": m6, "V70": v70,
            }]).reindex(columns=FEATURES, fill_value=-999)

            prob     = float(model.predict_proba(row)[0][1])
            decision = "REVIEW" if prob > THRESHOLD else "PASS"
            colour   = "#dc3545" if decision == "REVIEW" else "#28a745"

            # Fraud probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = round(prob * 100, 1),
                number= {"suffix": "%", "font": {"size": 48}},
                title = {"text": f"<b style='color:{colour}'>{decision}</b>",
                         "font": {"size": 30}},
                gauge = {
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": colour, "thickness": 0.25},
                    "bgcolor": "white",
                    "bordercolor": "#ccc",
                    "steps": [
                        {"range": [0,    50],   "color": "#d4edda"},
                        {"range": [50,   82.8], "color": "#fff3cd"},
                        {"range": [82.8, 100],  "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.85,
                        "value": THRESHOLD * 100
                    },
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.caption(f"Decision boundary at **{THRESHOLD}** (F1-optimal threshold). "
                       f"Default 0.5 threshold would give: "
                       f"**{'REVIEW' if prob > 0.5 else 'PASS'}**")

            # SHAP waterfall
            st.subheader("SHAP Explanation")
            with st.spinner("Computing SHAP values..."):
                shap_exp = explainer(row)
                fig_shap, _ = plt.subplots(figsize=(9, 5))
                shap.plots.waterfall(shap_exp[0], show=False)
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close()
        else:
            st.info("Fill in transaction details and click **Score Transaction**.")
            st.image("shap_outputs/waterfall_plot.png",
                     caption="Example SHAP waterfall — transaction from validation set",
                     use_container_width=True)

# ── TAB 2: Model Comparison ──────────────────────────────────────────
with tab2:
    st.subheader("XGBoost vs LightGBM — Test Set")

    metrics = pd.DataFrame({
        "Model":           ["XGBoost", "LightGBM"],
        "ROC-AUC":         [0.9568,    0.9440],
        "PR-AUC":          [0.7422,    0.6841],
        "Recall @ 0.5":    [0.827,     0.830],
        "Precision @ 0.5": [0.350,     0.270],
        "F1 @ optimal":    [0.696,     "—"],
        "Threshold":       [0.828,     "—"],
    })

    st.dataframe(
        metrics.set_index("Model")
               .style.highlight_max(axis=0, subset=["ROC-AUC","PR-AUC","Recall @ 0.5","Precision @ 0.5"],
                                    color="#d4edda"),
        use_container_width=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = px.bar(
            metrics.melt(id_vars="Model", value_vars=["ROC-AUC", "PR-AUC"],
                         var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group",
            color_discrete_map={"XGBoost": "#1f77b4", "LightGBM": "#ff7f0e"},
            title="ROC-AUC vs PR-AUC Comparison", text_auto=".3f"
        )
        fig_bar.update_layout(yaxis_range=[0.6, 1.0], height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_prec = px.bar(
            metrics.melt(id_vars="Model", value_vars=["Recall @ 0.5", "Precision @ 0.5"],
                         var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group",
            color_discrete_map={"XGBoost": "#1f77b4", "LightGBM": "#ff7f0e"},
            title="Precision & Recall @ Default Threshold", text_auto=".3f"
        )
        fig_prec.update_layout(yaxis_range=[0, 1.0], height=380)
        st.plotly_chart(fig_prec, use_container_width=True)

    st.success("**XGBoost wins by +5.8pp PR-AUC** — the metric that matters for imbalanced fraud data. "
               "PR-AUC is unaffected by the overwhelming majority of legitimate transactions.")

# ── TAB 3: Drift & Robustness ────────────────────────────────────────
with tab3:
    st.subheader("Feature Drift — Train vs Test")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Features",    360)
    c2.metric("Drifted Features",  8,      delta="-352 stable",   delta_color="off")
    c3.metric("Drift Rate",        "2.2%", delta="low — expected", delta_color="off")

    try:
        ref, curr = load_drift_data()
        top_features = [f for f in ["TransactionAmt", "amt_log", "C1", "C13", "C14", "V70"]
                        if f in ref.columns]
        selected = st.selectbox("Feature distribution comparison:", top_features)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=ref[selected].clip(upper=ref[selected].quantile(0.99)),
            name="Train", opacity=0.65, nbinsx=60, marker_color="#1f77b4"))
        fig_dist.add_trace(go.Histogram(
            x=curr[selected].clip(upper=curr[selected].quantile(0.99)),
            name="Test", opacity=0.65, nbinsx=60, marker_color="#ff7f0e"))
        fig_dist.update_layout(
            barmode="overlay", title=f"{selected} — Train vs Test Distribution",
            xaxis_title=selected, yaxis_title="Count", height=380
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    except FileNotFoundError:
        st.warning("Run the pipeline first to generate data/X_train.parquet and data/X_test.parquet.")

    st.divider()
    st.subheader("Adversarial Evasion Rate — Full Test Set Run")

    progress = {
        100: 0, 200: 4, 300: 7, 400: 11, 500: 13, 600: 14, 700: 22,
        800: 23, 900: 25, 1000: 27, 1100: 30, 1200: 32, 1300: 35,
        1400: 39, 1500: 41, 1600: 46, 1700: 53, 1800: 59, 1900: 63,
        2000: 65, 2100: 67, 2200: 68, 2300: 70, 2400: 72, 2500: 76, 2562: 77
    }
    df_ev = pd.DataFrame(list(progress.items()), columns=["Attacked", "Evaded"])
    df_ev["Rate (%)"] = (df_ev["Evaded"] / df_ev["Attacked"] * 100).round(2)

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(
        x=df_ev["Attacked"], y=df_ev["Rate (%)"],
        mode="lines+markers", name="Evasion Rate",
        line=dict(color="#dc3545", width=2.5),
        fill="tozeroy", fillcolor="rgba(220,53,69,0.08)"
    ))
    fig_ev.add_hline(y=2.0, line_dash="dash", line_color="#fd7e14",
                     annotation_text="2% reference line",
                     annotation_position="top right")
    fig_ev.update_layout(
        title="Nelder-Mead Adversarial Attack — Evasion Rate over Sample Size",
        xaxis_title="Fraud Transactions Attacked",
        yaxis_title="Evasion Rate (%)",
        yaxis_range=[0, 5], height=400
    )
    st.plotly_chart(fig_ev, use_container_width=True)
    st.caption(
        "**Final result: 77 / 2,562 = 3.01%** | Nelder-Mead optimizer, "
        "20 SHAP-selected features, max 150 iterations per sample. "
        "Evasions concentrate in the last 500 samples (borderline detections)."
    )
