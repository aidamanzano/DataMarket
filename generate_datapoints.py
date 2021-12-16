from datetime import datetime
from random import random, randrange


#https://geopy.readthedocs.io/en/stable/#googlev3

def make_data(number_of_datapoints):

    # Getting a list of random timestamps

    for i in range(number_of_datapoints):
        datetime = datetime.now()
        print(datetime)

    for i in range(number_of_datapoints):
        probability = random()
        print(probability)

    for i in range(number_of_datapoints):
        #make up a decimal degree data point
        #https://gis.stackexchange.com/questions/25877/generating-random-locations-nearby


