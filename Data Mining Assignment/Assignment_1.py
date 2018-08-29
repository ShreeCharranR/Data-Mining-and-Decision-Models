# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:47:32 2018

@author: Lenovo
"""

# Importing necessary libraries

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

"#############################################################################"

"""
Variables:
    L - flight Legs
    I -Itineraries
    C - capacity of the flight leg
    D - demand for the itinerary
    X - portion of the demand to be considered for the itinery without violating population
    
Objective Function:
    
"""

"""##########################################################################"""


def read_data(filename):
    """ Reads the text file and returns the data"""
    with open(filename, "r") as f:
        data = f.readlines()
        array_data = []
        for line in data:
            words = line.split()
            array_data.append(words)
    array_data = pd.DataFrame(array_data)
    array_data.columns = ["ID","Qty"]
    return(array_data)

capacity_data = read_data("capacity.txt")
demand_data = read_data("demand.txt")
fare_data = read_data("fare.txt")
it_leg_data = read_data("it_leg.txt")

capacity_data.Qty = capacity_data.Qty.astype('int64')
fare_data.Qty = fare_data.Qty.astype("float64")
demand_data.Qty = demand_data.Qty.astype("float64")

"""
it_leg_data.describe()

                             ID                     Qty
count                      9513                    9513
unique                  $$ 7949                   $$ 53
top     862008**I******20020920  LX0024720020920DXBZRHY
freq                          3                     904

fare_data.shape
Out[41]: (7949, 2)

fare_data.head()
Out[42]: 
                    ID     Qty
0  20**I******20020920  378.48
1  20BEIBOMBRU20020920  217.54
2  20INIBOMBRU20020920  539.42
3  23**I******20020920  682.10
4  23CHIZRHNRT20020920  682.10

demand_data.shape
Out[44]: (7949, 2)

demand_data.head()
Out[43]: 
                    ID     Qty
0  20**I******20020920  0.0922
1  20BEIBOMBRU20020920  0.2844
2  20INIBOMBRU20020920  1.1751
3  23**I******20020920  0.0000
4  23CHIZRHNRT20020920  0.0000

Total of 53 itinerary
Total of 7494 itinerary


"""

Z = np.array()

"""Constraints - Demand and Capacity """

C = np.array([demand_data.iloc[:,1].values,demand_data.iloc[:,1].values])
D = 







