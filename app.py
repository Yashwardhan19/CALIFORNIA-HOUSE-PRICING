import streamlit as st
import pandas as pd
import joblib

model    = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

st.title("🏠 California House Price Predictor")

# User se input lo
longitude         = st.number_input("Longitude",          value=-122.0)
latitude          = st.number_input("Latitude",           value=37.0)
housing_median_age= st.number_input("Housing Median Age", value=20)
total_rooms       = st.number_input("Total Rooms",        value=2000)
total_bedrooms    = st.number_input("Total Bedrooms",     value=400)
population        = st.number_input("Population",         value=1000)
households        = st.number_input("Households",         value=300)
median_income     = st.number_input("Median Income",      value=3.0)
ocean_proximity   = st.selectbox("Ocean Proximity", 
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