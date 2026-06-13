from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Sales Prediction | Task 5",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    :root {
        --bg: #090b0f;
        --panel: #11151b;
        --panel-2: #151a21;
        --panel-3: #1b212b;
        --text: #f4f5f7;
        --muted: #a9b0bb;
        --accent: #ff9f43;
        --accent-2: #6ec6ff;
        --border: rgba(255, 255, 255, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(255, 159, 67, 0.11), transparent 28%),
            radial-gradient(circle at bottom left, rgba(110, 198, 255, 0.10), transparent 24%),
            linear-gradient(180deg, #06070a 0%, #0b0e12 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: var(--text);
    }

    .hero {
        background: linear-gradient(135deg, rgba(255,159,67,0.16), rgba(17,21,27,0.9));
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 2rem;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
    }

    .eyebrow {
        letter-spacing: 0.24em;
        text-transform: uppercase;
        font-size: 0.78rem;
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: clamp(2.3rem, 5vw, 4.7rem);
        font-weight: 900;
        line-height: 0.95;
        margin: 0;
        color: var(--text);
    }

    .hero-title span {
        color: var(--accent);
    }

    .hero-copy {
        margin-top: 1rem;
        max-width: 62rem;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.8;
    }

    .glass-card {
        background: linear-gradient(180deg, rgba(17,21,27,0.96), rgba(13,16,22,0.96));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
        height: 100%;
    }

    .section-title {
        font-size: 1.1rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent-2);
        margin-bottom: 0.9rem;
        font-weight: 800;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-subtext {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.25rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(27,33,43,0.95), rgba(17,21,27,0.95));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.85rem 1rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
        border-radius: 12px 12px 0 0;
        color: var(--muted);
        padding: 0.65rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(255,159,67,0.14) !important;
        color: var(--text) !important;
        border-bottom-color: transparent !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ff9f43, #ff7a18);
        color: #111111;
        border: 0;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        font-weight: 800;
        box-shadow: 0 12px 30px rgba(255, 122, 24, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ffad5c, #ff8a30);
        color: #111111;
        border: 0;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }

    .stSlider [role="slider"] {
        background: var(--accent) !important;
        box-shadow: 0 0 0 6px rgba(255, 159, 67, 0.16) !important;
    }

    .stDataFrame, .stTable {
        border-radius: 16px;
        overflow: hidden;
    }

    a {
        color: var(--accent-2);
    }

    .note-box {
        border-left: 4px solid var(--accent);
        background: rgba(255, 159, 67, 0.08);
        padding: 0.9rem 1rem;
        border-radius: 0 14px 14px 0;
        color: var(--muted);
        line-height: 1.7;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

DATA_PATH = Path(__file__).with_name("Advertising.csv")


@st.cache_data

def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    return data


@st.cache_resource

def train_model(data: pd.DataFrame):
    features = ["TV", "Radio", "Newspaper"]
    target = "Sales"
    x = data[features]
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", ColumnTransformer([("scale", StandardScaler(), features)], remainder="drop")),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": math.sqrt(mean_squared_error(y_test, predictions)),
    }
    return model, metrics


def format_currency_like(value: float) -> str:
    return f"{value:,.2f}"


def main() -> None:
    data = load_data()
    model, metrics = train_model(data)

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Oasis Infobyte • Task 5</div>
            <h1 class="hero-title">Sales Prediction <span>Using Python</span></h1>
            <p class="hero-copy">
                A cleaner black themed Streamlit dashboard for predicting sales from advertising spend.
                Use the sliders on the left to test different campaign budgets, then review the predicted
                sales and model quality on the right.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    left, center, right = st.columns([1.1, 1.8, 1.1], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Campaign Inputs</div>', unsafe_allow_html=True)
        tv = st.slider("TV Advertising", 0.0, float(data["TV"].max()) * 1.15, float(data["TV"].median()), 0.1)
        radio = st.slider("Radio Advertising", 0.0, float(data["Radio"].max()) * 1.15, float(data["Radio"].median()), 0.1)
        newspaper = st.slider(
            "Newspaper Advertising",
            0.0,
            float(data["Newspaper"].max()) * 1.15,
            float(data["Newspaper"].median()),
            0.1,
        )
        st.markdown(
            """
            <div class="note-box">
                The model is trained on the provided <b>Advertising.csv</b> dataset and returns sales in the same unit as the original data.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    prediction = float(model.predict(pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"]))[0])

    with center:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Overview</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted Sales", format_currency_like(prediction))
        with col_b:
            st.metric("Model R²", f"{metrics['r2']:.3f}")
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("MAE", f"{metrics['mae']:.2f}")
        with col_d:
            st.metric("RMSE", f"{metrics['rmse']:.2f}")

        chart_df = pd.DataFrame(
            {
                "Channel": ["TV", "Radio", "Newspaper"],
                "Spend": [tv, radio, newspaper],
            }
        )
        fig = px.bar(
            chart_df,
            x="Channel",
            y="Spend",
            color="Channel",
            color_discrete_sequence=["#ff9f43", "#6ec6ff", "#9b7bff"],
            title="Campaign Spend Snapshot",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
            title_font=dict(size=18, color="#f4f5f7"),
            font=dict(color="#f4f5f7"),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Snapshot</div>', unsafe_allow_html=True)
        st.metric("Training Rows", f"{len(data):,}")
        st.metric("Features", "TV, Radio, Newspaper")
        st.metric("Target", "Sales")
        st.caption("A linear regression baseline is used to keep the app fast, stable, and easy to explain during internship review.")
        st.markdown('</div>', unsafe_allow_html=True)

    tab_data, tab_plots, tab_insights = st.tabs(["Data Preview", "Relationship Plots", "Insight"])

    with tab_data:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Advertising.csv Preview</div>', unsafe_allow_html=True)
        st.dataframe(data.head(12), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_plots:
        left_plot, right_plot = st.columns(2, gap="large")
        with left_plot:
            fig_tv = px.scatter(data, x="TV", y="Sales", template="plotly_dark", title="TV vs Sales")
            tv_line = np.polyfit(data["TV"], data["Sales"], 1)
            tv_x = np.linspace(data["TV"].min(), data["TV"].max(), 100)
            fig_tv.add_scatter(x=tv_x, y=np.polyval(tv_line, tv_x), mode="lines", name="Trend", line=dict(color="#ff9f43"))
            fig_tv.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tv, use_container_width=True)
        with right_plot:
            fig_radio = px.scatter(data, x="Radio", y="Sales", template="plotly_dark", title="Radio vs Sales")
            radio_line = np.polyfit(data["Radio"], data["Sales"], 1)
            radio_x = np.linspace(data["Radio"].min(), data["Radio"].max(), 100)
            fig_radio.add_scatter(x=radio_x, y=np.polyval(radio_line, radio_x), mode="lines", name="Trend", line=dict(color="#6ec6ff"))
            fig_radio.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_radio, use_container_width=True)

    with tab_insights:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        corr = data.corr(numeric_only=True)["Sales"].sort_values(ascending=False).to_frame(name="Correlation with Sales")
        st.markdown('<div class="section-title">Feature Correlation</div>', unsafe_allow_html=True)
        st.dataframe(corr, use_container_width=True)
        st.markdown(
            """
            <div class="note-box" style="margin-top:1rem;">
                TV spend is usually the strongest single driver in this dataset, while radio also contributes meaningfully.
                Newspaper is weaker, which is why a simple model is enough for the internship demo.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
