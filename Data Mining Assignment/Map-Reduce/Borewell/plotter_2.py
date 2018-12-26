# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 05:14:28 2018

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\mapreduce\\Borewell\\reducer_output.csv",header=None)

def plotter(index,df,i):
    df.columns = ['Frequency']
    df.index = index
    df.plot.bar(label="%s"%(i),ls='dashed',alpha=0.5)
    plt.title("The Histogram for Borewell %s"%(i))
    plt.legend()
    
for i in df[0].unique():
    plotter(df[df[0]==i][:][1].values,df[df[0]==i][:][2] ,i)
plt.show()

plotter(df[df[0]==False][:][1].values,df[df[0]==False][:][2] ,"False")

plt.clf()