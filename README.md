# 🏠 California House Price Predictor
 
A end-to-end machine learning web app that predicts median house prices in California based on location, demographics, and housing features.
 
🔗 **Live Demo:** [californiahousepricing.streamlit.app](https://californiahousepricing.streamlit.app)
 
---
 
## 📊 Model Performance
 
| Metric | Score |
|--------|-------|
| R² Score | 0.8323 |
| Test RMSE | $46,751 |
| MAE | $31,776 |
 
---
 
## 🛠️ Tech Stack
 
- **Language:** Python
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, XGBoost
- **App:** Streamlit
- **Deploy:** Streamlit Cloud
---
 
## 🔍 Project Highlights
 
- **Stratified sampling** — Income-based train/test split for unbiased evaluation
- **Full ML pipeline** — Imputation → Scaling → One-Hot Encoding using `ColumnTransformer`
- **XGBoost Regressor** — Best performer out of Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- **Cross-validation** — 10-fold CV to prevent overfitting
- **Live web app** — Deployed on Streamlit Cloud, no setup required
---
 
## 📁 Project Structure
 
```
CALIFORNIA-HOUSE-PRICING/
├── housing.csv          # California Housing dataset
├── app.py               # Streamlit web app
├── main.py              # Model training & evaluation
└── requirements.txt     # Python dependencies
```
 
---
 
## 🚀 Run Locally
 
```bash
# 1. Clone the repo
git clone https://github.com/your-username/california-house-pricing.git
cd california-house-pricing
 
# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Run the app
streamlit run app.py
```
 
---
 
## 📈 Model Comparison
 
| Model | CV RMSE |
|-------|---------|
| Linear Regression | ~$69,204 |
| Decision Tree | ~$69,081 |
| Gradient Boosting | ~$55,428 |
| Random Forest | ~$49,432 |
| **XGBoost ✅** | **~$48,297** |
 
---
 
## 📦 Dataset
 
California Housing dataset from the 1990 US Census.
Features: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`
 
Target: `median_house_value`
