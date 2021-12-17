from datetime import datetime
from random import random, randrange


#https://geopy.readthedocs.io/en/stable/#googlev3

#Probably make this into a class

def make_data(number_of_datapoints):

    # Getting a list of timestamps

    for i in range(number_of_datapoints):
        datetime = datetime.now()
        probability = random()
        #longitude = random()
        #latitude = random()

        #make up a decimal degree data point
        #https://gis.stackexchange.com/questions/25877/generating-random-locations-nearby
        #https://www.thepythoncode.com/article/get-geolocation-in-python


