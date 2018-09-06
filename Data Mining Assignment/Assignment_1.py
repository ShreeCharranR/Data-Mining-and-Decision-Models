# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:47:32 2018

@author: Lenovo
"""

# Importing necessary libraries

import numpy as np
import pandas as pd
from scipy.optimize import linprog

"""#############################################################################
Variables:
    L - flight Legs
    I - Itineraries
    C - capacity of the flight leg
    D - demand for the itinerary
    X - portion of the demand to be considered for the itinery without violating population
    
    I - varies from 1 to 7949
    L - varies from 1 to 53
    
    
Objective Function:
    
    Z -  Fare data transpose * X 
    
##########################################################################"""


def read_data(filename):
    """ Reads the text file and returns the data"""
    with open(filename, "r") as f:
        data = f.readlines()
        array_data = []
        for line in data:
            words = line.split()
            array_data.append(words)
    array_data = pd.DataFrame(array_data)
    return(array_data)

capacity_data = read_data("capacity.txt")
demand_data = read_data("demand.txt")
fare_data = read_data("fare.txt")
it_leg_data = read_data("it_leg.txt")

#Renaming column name for clarity
capacity_data.columns = np.array(['Flight','Capacity'])
demand_data.columns =np.array(['Itinerary','Demand'])
fare_data.columns = np.array(['Itinerary','Fare'])
it_leg_data.columns = np.array(['Itinerary','Flight'])

#changing the data type
capacity_data.Capacity = capacity_data.Capacity.astype('int64')
fare_data.Fare = fare_data.Fare.astype("float64")
demand_data.Demand = demand_data.Demand.astype("float64")


"""
Information about the data

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

Total of 53 flights
Total of 7494 itinerary combinations

len(it_leg_data.Itinerary.unique())
Out[17]: 7949

len(it_leg_data.Flight.unique())
Out[18]: 53

we are optimizing for each itinerary hence the formulation take following structure

z =  7949 x 1

b = (7949 + 53) x 7949

The it_flight dictionary will have flight as key and the itinerary as the value
The matrix should have 1 in the place for every itinerary else 0 adding sparsness
in the exisiting B matrix

"""

Z = fare_data.iloc[:,1].values#.reshape(len(fare_data.iloc[:,1].values),1)
Z.shape


"""Constraints - Demand and Capacity 

demand_data.iloc[:,0].values == fare_data.iloc[:,0].values
Out[30]: array([ True,  True,  True, ...,  True,  True,  True])

demand and fare data are in correct array

"""
#Dictionary for flight used in itenerary - 53 rows
it_flight_dict = {}

for i in it_leg_data.Flight.unique():
    it_flight_dict[i] = it_leg_data[it_leg_data['Flight']==i]['Itinerary'].tolist()

#Dictionary for itinerary to flight mappings reverse
flight_it_dict = {}

for i in it_leg_data.Itinerary.unique():
    flight_it_dict[i] = it_leg_data[it_leg_data['Itinerary']==i]['Flight'].tolist()

# Constraints
A = np.diag(np.array([1 for i in range(fare_data.shape[0])]))
B = demand_data.iloc[:,1].values.reshape(len(demand_data.iloc[:,1].values),1)


for k,v in it_flight_dict.items():
    temp_array= np.zeros(demand_data.shape[0])
    for i in v:
        temp_array = np.logical_or(temp_array,np.array(demand_data.iloc[:,0].values == i))
    temp_array = temp_array.astype(int)
    A = np.vstack([A,temp_array])
    #Appending the capacity data for each flight to B matrix [num] -  , 53 times iteration
    #directly adding the rows to B to avoid creation of new variable
    B = np.vstack([B, int(capacity_data[capacity_data.Flight == k]['Capacity'])])

#Applying optimization using linear programming
res = linprog(c = Z, A_ub=A, b_ub=B,options={"disp": True})


