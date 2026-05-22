# 🏠 California House Price Predictor

A machine learning web app that predicts California housing prices using XGBoost.

## 🔗 Live Demo
👉 [californiahousepricing.streamlit.app](https://californiahousepricing.streamlit.app)

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| R² Score | 0.8323 |
| Test RMSE | $46,751 |
| MAE | $31,776 |

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer)
- XGBoost
- Streamlit

## 📁 Project Structure
CALIFORNIA-HOUSE-PRICING/
├── housing.csv       # Dataset
├── app.py            # Streamlit web app
├── main.py           # Model training & evaluation
└── requirements.txt  # Dependencies

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py
