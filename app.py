import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

MODEL_FILE    = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

@st.cache_resource  # ✅ sirf ek baar train karega, baar baar nahi
def load_or_train():
    housing = pd.read_csv("housing.csv")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, _ in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_index].drop("income_cat", axis=1)

    housing_labels   = train_set["median_house_value"].copy()
    housing_features = train_set.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(housing_prepared, housing_labels)

    return model, pipeline

# App UI
st.title("🏠 California House Price Predictor")

model, pipeline = load_or_train()

longitude          = st.number_input("Longitude",          value=-122.0)
latitude           = st.number_input("Latitude",           value=37.0)
housing_median_age = st.number_input("Housing Median Age", value=20)
total_rooms        = st.number_input("Total Rooms",        value=2000)
total_bedrooms     = st.number_input("Total Bedrooms",     value=400)
population         = st.number_input("Population",         value=1000)
households         = st.number_input("Households",         value=300)
median_income      = st.number_input("Median Income",      value=3.0)
ocean_proximity    = st.selectbox("Ocean Proximity",
                    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

if st.button("Predict Price"):
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
    st.success(f"Predicted House Value: **${prediction:,.0f}**")