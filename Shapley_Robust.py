from math import exp
def SM(datapoint1, datapoint2):
    """The similarity measure function, a higher value for the output
    will penalise datapoints more in the shapley robust algorithm"""
    similarityScore = abs(datapoint2- datapoint1)
    return similarityScore

def ShapleyApprox(Y_n, X_M, K):


    return psi_n 

def ShapleyRobust(Y_n, X_M, K, SM, lambda):
    psi_n_bar = ShapleyRobust(Y_n, X_M, K)
    psi_n = psi_n_bar*exp(-lambda)