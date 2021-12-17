from math import exp
import shap
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
import sklearn

data = pd.read_csv("datapoints.csv")
def SM(datapoint1, list): #datapoint1 is the item we are checking
    """The similarity measure function, a higher value for the output
    will penalise datapoints more in the shapley robust algorithm"""

    #currently location is split into longitude and latitude to train a model.
    #If we can input location as a vector, we could calculate similarity using cosine similarity.
    similarityScore = 0
    for datapoint in list:
        if datapoint != datapoint1:
            similarityScore += abs(datapoint- datapoint1)
    return similarityScore

def ShapleyValue():
    #https://www.analyticsvidhya.com/blog/2019/11/shapley-value-machine-learning-interpretability-game-theory/
    return psi

TrainingColumnLabels = ['latitude', 'longitude']
PredictionColumnLabels = ['probability']
dataPath = "datapoints.csv"

def generateModel(dataset_path, XcolumnLabels, YcolumnLabels, model = sklearn.linear_model.LinearRegression()):
    """Generates a predictive task to input into the shapley value function. Takes dataset path, the column labels for the
    training and test data. The default model choice is a linear regression from sklearn.learn
    Also supports XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models."""
    data = pd.read_csv(dataset_path)

    #X = data[['date', 'time', 'latitude', 'longitude']] I'm not sure what the best model would be to pick for all 4 attributes
    X = data[XcolumnLabels]
    y = data[YcolumnLabels]


    # fit the model
    model.fit(X, y)
    return model

#SHAP values
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)


print('SHAPLEY VALUES', shap_values.values) #this returns the shapley value of each value of each datapoint.

"""we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes 
over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. 
The color represents the feature value (red high, blue low). 
This reveals for example that a high latitude increases the predicted probability that the parking slot is empty."""

shap.plots.beeswarm(shap_values)


""" def ShapleyApprox(Y_n, X_M, K):

    #randomly subsample shapley value
    return psi_n  """

def ShapleyRobust(Y_n, X_M, K, SM, lambda):
    psi_n_bar = ShapleyApprox(Y_n, X_M, K)
    psi_n = psi_n_bar*exp(-lambda)
    """need M sums"""
    #for each point1
        sum + 0
        for each point that is not point1
            find_sim_metric
            sum +equal 
    #for each datapoint excpet datapoint i,
    #check the similarity between i and every datapoint
    #sum over the result
    #then do psi*e^(-Lambda*Sum)
