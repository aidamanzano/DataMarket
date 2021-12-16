
import shap
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
import xgboost

data = pd.read_csv("datapoints.csv")

X = data[['latitude', 'longitude']]
y = data[['probability']]

model = xgboost.XGBRegressor().fit(X, y)
print(model)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])