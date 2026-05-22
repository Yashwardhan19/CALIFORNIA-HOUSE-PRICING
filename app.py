import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}
.main { background-color: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1e2130, #252840);
    border: 1px solid #2e3250;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    color: #8b92b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8ff;
}
.predict-result {
    background: linear-gradient(135deg, #1a3a2a, #1e4535);
    border: 1px solid #2d6b4a;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.predict-label {
    font-size: 0.8rem;
    color: #6fcf97;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.predict-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #27ae60;
}
</style>
""", unsafe_allow_html=True)


# ── Train model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    housing = pd.read_csv("housing.csv")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_idx].drop("income_cat", axis=1)
        test_set  = housing.loc[test_idx].drop("income_cat", axis=1)

    # Train
    housing_labels   = train_set["median_house_value"].copy()
    housing_features = train_set.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    housing_prepared = pipeline.fit_transform(housing_features)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Test metrics
    test_features = test_set.drop("median_house_value", axis=1)
    test_labels   = test_set["median_house_value"]
    test_prepared = pipeline.transform(test_features)
    test_preds    = model.predict(test_prepared)

    metrics = {
        "rmse": root_mean_squared_error(test_labels, test_preds),
        "mae":  mean_absolute_error(test_labels, test_preds),
        "r2":   r2_score(test_labels, test_preds),
    }

    return model, pipeline, metrics


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏠 California House Price Predictor")
st.markdown("<p style='color:#8b92b8; margin-top:-0.5rem;'>XGBoost model trained on California Housing dataset</p>", unsafe_allow_html=True)

with st.spinner("Training model on startup..."):
    model, pipeline, metrics = load_model()

# Model metrics
st.markdown("#### Model Performance")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{metrics['r2']:.4f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Test RMSE</div>
        <div class="metric-value">${metrics['rmse']:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">MAE</div>
        <div class="metric-value">${metrics['mae']:,.0f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("#### Enter House Details")

col1, col2 = st.columns(2)

with col1:
    longitude          = st.number_input("Longitude",           value=-122.0, format="%.4f")
    housing_median_age = st.number_input("Housing Median Age",  value=20,     min_value=1,  max_value=52)
    total_bedrooms     = st.number_input("Total Bedrooms",       value=400,    min_value=1)
    households         = st.number_input("Households",           value=300,    min_value=1)
    ocean_proximity    = st.selectbox("Ocean Proximity",
                            ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

with col2:
    latitude           = st.number_input("Latitude",             value=37.0,   format="%.4f")
    total_rooms        = st.number_input("Total Rooms",          value=2000,   min_value=1)
    population         = st.number_input("Population",           value=1000,   min_value=1)
    median_income      = st.number_input("Median Income (x$10k)", value=3.0,   min_value=0.5, max_value=15.0, step=0.1)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "longitude":          longitude,
        "latitude":           latitude,
        "housing_median_age": housing_median_age,
        "total_rooms":        total_rooms,
        "total_bedrooms":     total_bedrooms,
        "population":         population,
        "households":         households,
        "median_income":      median_income,
        "ocean_proximity":    ocean_proximity
    }])

    prediction = model.predict(pipeline.transform(input_df))[0]

    st.markdown(f"""
    <div class="predict-result">
        <div class="predict-label">Predicted Median House Value</div>
        <div class="predict-value">${prediction:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)