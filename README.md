# Mental Health & Social Media Paradox

## Project overview
End-to-end data analytics project analyzing how online social connectivity relates to mental health outcomes across 2,947 US counties (administrative districts across 48 US states).

## Key finding
Higher social media connectivity (SCI) correlates with **worse** mental health in poor counties but shows no harm in wealthy counties — disproving the universal "more connection = better wellbeing" assumption.

## Tech stack
Python · pandas · XGBoost · scikit-learn · Streamlit · Plotly · GeoPandas

## Data sources
- CDC PLACES 2023 — county-level depression and mental distress rates
- US Census ACS 2021 — income, poverty, broadband access
- Social Capital Atlas proxy — digital connectivity index

## Results
- XGBoost model (R² 0.65) with 9 engineered features
- Economic stress ranks 2x above SCI in feature importance (32.6 vs 16.5)
- 4 county vulnerability profiles via K-means clustering
- Interactive Streamlit dashboard with choropleth map and county explorer

## How to run
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Project structure
```
notebooks/     — data ingestion, cleaning, EDA, modeling
dashboard/     — Streamlit app
data/processed — cleaned master dataset and trained model
```
