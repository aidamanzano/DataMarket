
import numpy as np
import shap
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("datapoints.csv")

#X = data[['date', 'time', 'latitude', 'longitude']] I'm not sure what the best model would be to pick for all 4 attributes
X = data[['latitude', 'longitude']]
y = data[['probability']]

def cosineSimilarity(a, b):
    similarity = cosine_similarity(a, b)
    return similarity


finalSimilarities = np.zeros(X.shape[0])

for i,rowOfInterest in enumerate(np.array(X)):
    print('I',i,'row of interest',rowOfInterest)
    for row in np.array(X):
        print('row',row)
        if (rowOfInterest != row).all(): #checks all elements are the same
            print('COS Similarity', cosineSimilarity(rowOfInterest,row))
            finalSimilarities[i] += cosineSimilarity(rowOfInterest,row)

print(finalSimilarities.reshape(-1,1))

#ShapleyRobust = ShapleyValues* np.exp(-l * finalSimilarities.reshape(-1,1))

    

""" create a zeros array of size X.

similarityscore = 0
for row in X:
    if row != datapoint_ofinterest:
        similarityscore += cosineSimilarity(item,datapoint_ofinterest)
    put similarityscore of item in corresponding zeros place """
""" 

for datapoint_of_interest in X:
    sim_score = 0

    for datapoint in data

        if datapoint is not datapoint_of_interest

# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

#SHAP values
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)


print('SHAPLEY VALUES', shap_values.values) #this returns the shapley value of each value of each datapoint.

we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes 
over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. 
The color represents the feature value (red high, blue low). 
This reveals for example that a high latitude increases the predicted probability that the parking slot is empty.

shap.plots.beeswarm(shap_values) """