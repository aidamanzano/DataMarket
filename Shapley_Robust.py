from math import exp
import shap
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
import sklearn
from scipy.spatial.distance import cosine
import numpy as np

def cosineSimilarity(a, b):
    similarity = cosine(a, b) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
    return similarity

data = pd.read_csv("datapoints.csv")
TrainingColumnLabels = ['latitude', 'longitude']
PredictionColumnLabels = ['probability']
dataPath = "datapoints.csv"

def generateModel(dataset_path, XcolumnLabels, YcolumnLabels, model = sklearn.linear_model.LinearRegression()):
    """Generates a predictive task to input into the shapley value function. Takes dataset path, the column labels for the
    training (XcolumnLabels) and the prediction task (YcolumnLabels) data. The default model choice is a linear regression from sklearn.learn
    Also supports XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models."""
    data = pd.read_csv(dataset_path)

    #X = data[['date', 'time', 'latitude', 'longitude']] I'm not sure what the best model would be to pick for all 4 attributes
    X = data[XcolumnLabels]
    y = data[YcolumnLabels]

    # fit the model
    model.fit(X, y)
    return model


def ShapleyValue(model, X_features):
    #https://www.analyticsvidhya.com/blog/2019/11/shapley-value-machine-learning-interpretability-game-theory/
 
    explainer = shap.Explainer(model.predict, X_features)
    shap_values = explainer(X_features)
    psi = shap_values.values #vector of the shapley values of each feature of each datapoint (the X_features).

    #print("shapley values", psi)
    #shap.plots.beeswarm(shap_values)
    return psi


"""we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes 
over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. 
The color represents the feature value (red high, blue low). 
This reveals for example that a high latitude increases the predicted probability that the parking slot is empty."""



""" def ShapleyApprox(Y_n, X_M, K):

    #randomly subsample shapley value
    return psi_n  """

#dahleh paper would take the following inputs: ShapleyRobust(Y_n, X_M, K, SimilarityMeasure_Function, lambda):
def ShapleyRobust(model, X_features, K, SimilarityMeasure_Function, lamda):
    psi_n_bar = ShapleyValue(model, X_features)
    finalSimilarities = np.zeros(X_features.shape[0])

    for i,rowOfInterest in enumerate(np.array(X_features)):
        print('I',i,'row of interest',rowOfInterest)
        for row in np.array(X_features):
            print('row',row)
            if (rowOfInterest != row).all(): #checks all elements are the same
                print('COS Similarity', SimilarityMeasure_Function(rowOfInterest,row))
                finalSimilarities[i] += SimilarityMeasure_Function(rowOfInterest,row)

    print(finalSimilarities.reshape(-1,1))
    ShapleyRobust = psi_n_bar* np.exp(-lamda * finalSimilarities.reshape(-1,1))
    return ShapleyRobust



data = pd.read_csv("datapoints.csv")
TrainingColumnLabels = ['latitude', 'longitude']
PredictionColumnLabels = ['probability']
dataPath = "datapoints.csv"