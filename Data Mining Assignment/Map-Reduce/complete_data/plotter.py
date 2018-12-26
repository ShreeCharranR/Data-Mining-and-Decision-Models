# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 05:14:28 2018

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\mapreduce\\complete_data\\reducer_output.csv",header=None)

df.index = df[0]
del df[0]
df.columns = ['Frequency']

df.plot.bar()
plt.title("Complete Consumption")
plt.show()

