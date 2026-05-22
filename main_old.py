import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
 
# 1. Load the data
housing = pd.read_csv("housing.csv")
 
# 2. Create a stratified test set based on income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)
 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)
 
# Work on a copy of training data
housing = strat_train_set.copy()
 
# 3. Separate features and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
 
# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]
 
# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
 
# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
 
# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
 
# housing_prepared is now a NumPy array ready for training
# print(housing_prepared.shape)

# Linear Regression
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
 
# Decision Tree
# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(housing_prepared, housing_labels)
 
# Random Forest
# forest_reg = RandomForestRegressor(random_state=42)
# forest_reg.fit(housing_prepared, housing_labels)

#Gradient Boosting Regressor
# gradient_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
# gradient_reg.fit(housing_prepared, housing_labels)

#XGBoost Regressor
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_reg.fit(housing_prepared, housing_labels)
 
# Predict using training data
# lin_preds = lin_reg.predict(housing_prepared)
# tree_preds = tree_reg.predict(housing_prepared)
# forest_preds = forest_reg.predict(housing_prepared)
# gradient_preds = gradient_reg.predict(housing_prepared)
xgb_preds = xgb_reg.predict(housing_prepared)
 
# Calculate RMSE
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# tree_rmse = root_mean_squared_error(housing_labels, tree_preds)
# forest_rmse = root_mean_squared_error(housing_labels, forest_preds)
# gradient_rmse = root_mean_squared_error(housing_labels, gradient_preds)
# xgb_rmse = root_mean_squared_error(housing_labels, xgb_preds)
 
# print("Linear Regression RMSE:", lin_rmse)
# print("Decision Tree RMSE:", tree_rmse)
# print("Random Forest RMSE:", forest_rmse)
# print("Gradient Boosting RMSE:", gradient_rmse)
# print("Xgb Regressor RMSE:", xgb_rmse)

#Output Rmse before cross validation
'''
Linear Regression RMSE: 69050.56219504567
Decision Tree RMSE: 0.0
Random Forest RMSE: 18342.366362322846
Gradient Boosting RMSE: 53494.18776891836
Xgb Regressor RMSE: 38916.150808727194
'''

#Predicting rmses using Cross Validation
# lin_rmses = -cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)

# tree_rmses = -cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)

# forest_rmses = -cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)

# gradient_rmses = -cross_val_score(gradient_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)

xgb_rmses = -cross_val_score(xgb_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)


# print("Linear Regression RMSE:", lin_rmses)
# print(pd.Series(lin_rmses).describe())
# print("Decision Tree RMSE:", tree_rmses)
# print(pd.Series(tree_rmses).describe())
# print("Random Forest RMSE:", forest_rmses)
# print(pd.Series(forest_rmses).describe())
# print("Gradient Boosting RMSE:", gradient_rmses)
# print(pd.Series(gradient_rmses).describe())
print("Xgb Regressor RMSE:", xgb_rmses)
print(pd.Series(xgb_rmses).describe())



#Output mean rmse using cross validation
'''
Linear Regression RMSE:  69204.322755
Decision Tree RMSE:  69081.361563
Random Forest RMSE: 49432.126788
Gradient Boosting RMSE: 55427.942029
Xgb Regressor RMSE :  48296.905277#best
'''