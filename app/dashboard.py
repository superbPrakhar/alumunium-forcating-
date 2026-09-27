import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Aluminium Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background-color:#0f172a;
}

[data-testid="stHeader"]{
    background:transparent;
}

[data-testid="stSidebar"]{
    background-color:#111827;
}

h1,h2,h3,h4{
    color:white !important;
}

p{
    color:#cbd5e1;
}

[data-testid="metric-container"]{
    background:#1e293b;
    border:1px solid #334155;
    padding:20px;
    border-radius:18px;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);
}

.hero{
    padding:35px;
    border-radius:25px;
    background: linear-gradient(
        135deg,
        #0ea5e9,
        #2563eb,
        #7c3aed
    );
    margin-bottom:25px;
}

.hero-title{
    font-size:52px;
    font-weight:700;
    color:white;
}

.hero-subtitle{
    font-size:20px;
    color:#e2e8f0;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("outputs/predictions.csv")

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("📊 Navigation")

    selected = option_menu(
        menu_title=None,
        options=[
            "Overview",
            "Predictions",
            "Models",
            "Feature Importance"
        ],
        icons=[
            "house",
            "graph-up",
            "cpu",
            "bar-chart"
        ],
        default_index=0
    )

# =====================================
# HERO SECTION
# =====================================

st.markdown("""
<div class="hero">

<div class="hero-title">
📈 Aluminium Intelligence Dashboard
</div>

<div class="hero-subtitle">
AI-Powered Commodity Forecasting Platform
</div>

</div>
""", unsafe_allow_html=True)

# =====================================
# OVERVIEW
# =====================================

if selected == "Overview":

    st.subheader("📌 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R² Score",
            "98.79%"
        )

    with col2:
        st.metric(
            "MAE",
            "34.26"
        )

    with col3:
        st.metric(
            "RMSE",
            "47.54"
        )

    with col4:
        st.metric(
            "Best Model",
            "Linear Regression"
        )

    st.markdown("---")

    st.subheader("📈 Actual vs Predicted Prices")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=df["Actual_Price"],
            mode="lines",
            name="Actual Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=df["Predicted_Price"],
            mode="lines",
            name="Predicted Price"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title="Actual vs Predicted Aluminium Prices",
        xaxis_title="Test Data Points",
        yaxis_title="Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# PREDICTIONS
# =====================================

elif selected == "Predictions":

    st.subheader("📋 Prediction Results")

    st.dataframe(
        df,
        use_container_width=True
    )

# =====================================
# MODELS
# =====================================

elif selected == "Models":

    st.subheader("🤖 Model Comparison")

    comparison = pd.DataFrame({
        "Model":[
            "Linear Regression",
            "Random Forest",
            "SVR"
        ],
        "MAE":[
            34.26,
            330.67,
            816.79
        ],
        "RMSE":[
            47.54,
            472.44,
            922.44
        ],
        "R²":[
            0.9879,
            -0.1963,
            -3.5606
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    fig = px.bar(
        comparison,
        x="Model",
        y="R²",
        text="R²",
        template="plotly_dark",
        title="Model Performance Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# FEATURE IMPORTANCE
# =====================================

elif selected == "Feature Importance":

    feature_df = pd.DataFrame({

        "Feature":[
            "Unemployment",
            "Industrial Production",
            "Price Lag 1",
            "USD Index",
            "Interest Rate",
            "MA 7",
            "Price Lag 7",
            "Oil Price",
            "Inflation",
            "MA 30",
            "SP500",
            "Price Lag 30",
            "GDP"
        ],

        "Importance":[
            -2.687791,
            -1.250449,
            0.783054,
            -0.579491,
            0.475533,
            0.311784,
            -0.126908,
            0.090008,
            -0.063365,
            0.027304,
            0.013095,
            -0.009625,
            -0.005560
        ]
    })

    fig = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        template="plotly_dark",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        feature_df,
        use_container_width=True
    )