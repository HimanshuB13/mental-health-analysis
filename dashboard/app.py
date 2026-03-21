import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Mental Health & Social Media Paradox",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "data", "processed", "master.csv")
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    path = os.path.join(BASE_DIR, "data", "processed", "xgb_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

master = load_data()
model  = load_model()

# Sidebar filters
st.sidebar.title("Filters")
selected_region = st.sidebar.multiselect(
    "Select region",
    options=master["region"].dropna().unique().tolist(),
    default=master["region"].dropna().unique().tolist()
)
selected_income = st.sidebar.multiselect(
    "Select income group",
    options=["Q1 Low", "Q2 Mid-Low", "Q3 Mid-High", "Q4 High"],
    default=["Q1 Low", "Q2 Mid-Low", "Q3 Mid-High", "Q4 High"]
)
selected_cluster = st.sidebar.multiselect(
    "Select cluster",
    options=master["cluster_name"].dropna().unique().tolist(),
    default=master["cluster_name"].dropna().unique().tolist()
)

filtered = master[
    (master["region"].isin(selected_region)) &
    (master["income_quartile"].isin(selected_income)) &
    (master["cluster_name"].isin(selected_cluster))
]

# Header
st.title("🧠 Mental Health & Social Media Paradox")
st.markdown("Analyzing how online social connectivity relates to mental health outcomes across **{:,} US counties**".format(len(filtered)))
st.divider()

# Metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties analyzed", f"{len(filtered):,}")
col2.metric("Avg depression rate", f"{filtered['depression_rate'].mean():.1f}%")
col3.metric("Avg median income", f"${filtered['income'].mean():,.0f}")
col4.metric("Avg SCI score", f"{filtered['sci'].mean():.3f}")
st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ County Map",
    "📊 The Paradox",
    "🤖 Model Insights",
    "🔍 County Explorer"
])

# TAB 1 - Map
with tab1:
    st.subheader("County-level depression rates across the US")
    map_metric = st.selectbox(
        "Select metric to display",
        ["depression_rate", "mental_distress_rate", "income", "poverty_rate", "sci"]
    )
    # Aggregate county data to state level for map
    state_data = filtered.groupby("state").agg(
        avg_depression  = ("depression_rate", "mean"),
        avg_distress    = ("mental_distress_rate", "mean"),
        avg_income      = ("income", "mean"),
        avg_poverty     = ("poverty_rate", "mean"),
        avg_sci         = ("sci", "mean"),
        counties        = ("county", "count")
    ).reset_index()

    # Add 2-letter state codes
    state_codes = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
        "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
        "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
        "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
        "Wisconsin": "WI", "Wyoming": "WY"
    }

    state_data["state_code"] = state_data["state"].map(state_codes)

    metric_map = {
        "depression_rate":      "avg_depression",
        "mental_distress_rate": "avg_distress",
        "income":               "avg_income",
        "poverty_rate":         "avg_poverty",
        "sci":                  "avg_sci"
    }

    plot_col = metric_map.get(map_metric, "avg_depression")

    fig_map = px.choropleth(
        state_data,
        locations="state_code",
        locationmode="USA-states",
        color=plot_col,
        scope="usa",
        color_continuous_scale="RdYlGn_r",
        hover_name="state",
        hover_data={
            "avg_depression": ":.1f",
            "avg_income":     ":,.0f",
            "avg_sci":        ":.3f",
            "counties":       True
        },
        labels={
            "avg_depression":  "Avg depression %",
            "avg_distress":    "Avg distress %",
            "avg_income":      "Avg income",
            "avg_poverty":     "Avg poverty %",
            "avg_sci":         "Avg SCI",
            "counties":        "Counties"
        }
    )
    fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True, key="main_map")
# fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
# st.plotly_chart(fig_map, use_container_width=True)

# TAB 2 - Paradox
with tab2:
    st.subheader("The SCI vs Depression paradox by income group")
    col_left, col_right = st.columns(2)
    with col_left:
        colors = {
            "Q1 Low": "#E24B4A",
            "Q2 Mid-Low": "#D85A30",
            "Q3 Mid-High": "#BA7517",
            "Q4 High": "#1D9E75"
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        for quartile, color in colors.items():
            subset = filtered[filtered["income_quartile"] == quartile]
            if len(subset) > 10:
                ax.scatter(subset["sci"], subset["depression_rate"],
                          alpha=0.4, s=15, color=color, label=quartile)
                z = np.polyfit(subset["sci"], subset["depression_rate"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subset["sci"].min(), subset["sci"].max(), 100)
                ax.plot(x_line, p(x_line), color=color, linewidth=2)
        ax.set_title("SCI vs Depression by income group", fontsize=12)
        ax.set_xlabel("Social Connectedness Index (SCI)")
        ax.set_ylabel("Depression rate (%)")
        ax.legend(title="Income group")
        st.pyplot(fig)
        plt.close()
    with col_right:
        st.markdown("### What this chart shows")
        st.markdown("""
        - **Red line (Q1 Poor)** slopes UP — more connectivity, more depression in poor counties
        - **Green line (Q4 Rich)** shows opposite trend in wealthy counties
        - **The paradox** — connectivity does not equally benefit all income groups
        """)
        quartile_stats = filtered.groupby("income_quartile").agg(
            counties=("county", "count"),
            avg_depression=("depression_rate", "mean"),
            avg_income=("income", "mean")
        ).round(2)
        st.dataframe(quartile_stats, use_container_width=True)

# TAB 3 - Model
with tab3:
    st.subheader("What drives depression rates — XGBoost feature importance")
    col_left, col_right = st.columns(2)
    with col_left:
        importance = model.get_booster().get_score(importance_type="gain")
        imp_df = pd.DataFrame({
            "feature": list(importance.keys()),
            "importance": list(importance.values())
        }).sort_values("importance", ascending=True)
        colors_imp = [
            "#E24B4A" if f in ["economic_stress", "income", "poverty_rate"]
            else "#7F77DD" if f == "sci"
            else "#888780"
            for f in imp_df["feature"]
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp_df["feature"], imp_df["importance"],
                color=colors_imp, edgecolor="white", linewidth=0.5)
        ax.set_title("Feature importance (gain)", fontsize=12)
        ax.set_xlabel("Gain score")
        for i, (feat, imp) in enumerate(zip(imp_df["feature"], imp_df["importance"])):
            ax.text(imp * 1.01, i, f"{imp:.1f}", va="center", fontsize=9)
        st.pyplot(fig)
        plt.close()
    with col_right:
        st.markdown("### Key findings")
        st.markdown("""
        - 🔴 **Economic stress ranks #1** — strongest depression predictor
        - 🔵 **SCI ranks last (#9)** — weakest predictor out of 9 features
        - 📍 **Region ranks #2** — Southern counties systematically worse
        - 💡 **SCI only matters when combined with income context**
        """)
        cluster_counts = filtered["cluster_name"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.pie(
            cluster_counts.values,
            labels=cluster_counts.index,
            colors=["#E24B4A", "#7F77DD", "#BA7517", "#1D9E75"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax2.set_title("County vulnerability profiles", fontsize=11)
        st.pyplot(fig2)
        plt.close()

# TAB 4 - Explorer
with tab4:
    st.subheader("Explore individual counties")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_state = st.selectbox(
            "Select state",
            options=sorted(master["state"].dropna().unique().tolist())
        )
    with col_s2:
        state_counties = master[master["state"] == selected_state]["county"].tolist()
        selected_county = st.selectbox(
            "Select county",
            options=sorted(state_counties)
        )

    county_data = master[
        (master["state"] == selected_state) &
        (master["county"] == selected_county)
    ].iloc[0]

    st.markdown(f"### {selected_county} County, {selected_state}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Depression rate", f"{county_data['depression_rate']:.1f}%")
    c2.metric("Mental distress", f"{county_data['mental_distress_rate']:.1f}%")
    c3.metric("Median income", f"${county_data['income']:,.0f}")
    c4.metric("SCI score", f"{county_data['sci']:.3f}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Poverty rate", f"{county_data['poverty_rate']:.1f}%")
    c6.metric("Broadband rate", f"{county_data['broadband_rate']:.1f}%")
    c7.metric("Income group", str(county_data["income_quartile"]))
    c8.metric("Cluster", str(county_data["cluster_name"]))

    st.markdown("### How this county compares to national average")
    compare_cols = ["depression_rate", "mental_distress_rate", "income", "poverty_rate", "sci"]
    compare_labels = ["Depression %", "Mental distress %", "Income ($)", "Poverty %", "SCI"]
    compare_df = pd.DataFrame({
    "Metric": compare_labels,
    "This county": [round(float(county_data[c]), 2) for c in compare_cols],
    "National avg": [round(float(master[c].mean()), 2) for c in compare_cols]
})
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Data: CDC PLACES 2023 · Census ACS 2021 · Social Capital Atlas · XGBoost · Streamlit")  