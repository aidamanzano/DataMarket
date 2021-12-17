# DataMarket

Currently this contains an implementation of the Shapley Robust algorithm from the following paper: https://dl.acm.org/doi/pdf/10.1145/3328526.3329589
This is done in the Shapley_Robust.py file. Ignore all other python files, they're not being used.

Pick a similarity measure of choice, here we are using cosine similarity from the scipy module.
Generate a model and a matrix of training features, the function generateModel() contains more information on the supported models.The default type is a LinearRegression model using the scikit-learn module.

The Shapley Values are calculated using the SHAP library, documentation available here: https://shap.readthedocs.io/en/latest/index.html and github repo available here: https://github.com/slundberg/shap

The Shapley Robust values are calculated in the ShapleyRobust() function. 


