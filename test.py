
import shap
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
import sklearn

data = pd.read_csv("datapoints.csv")

#X = data[['date', 'time', 'latitude', 'longitude']] I'm not sure what the best model would be to pick for all 4 attributes
X = data[['latitude', 'longitude']]
y = data[['probability']]


# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

#SHAP values
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)


print('SHAPLEY VALUES', shap_values.values) #this returns the shapley value of each value of each datapoint.

"""we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes 
over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. 
The color represents the feature value (red high, blue low). 
This reveals for example that a high latitude increases the predicted probability that the parking slot is empty."""

shap.plots.beeswarm(shap_values)