# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 07:17:20 2018

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\mapreduce\\Connection_and_borewell\\reducer_output.csv",header=None)

def plotter(index,df,i):
    df.columns = ['Frequency']
    df.index = index
    df.plot.bar(label="%s"%(i),ls='dashed',alpha=0.5)
    plt.title("The Histogram for Connection_Type and Borewell %s"%(i))
    plt.legend()

df['indexing'] = df[0].apply(str)+" "+df[1].apply(str)
del df[0]
del df[1]
    
for i in df['indexing'].unique():
    plotter(df[df['indexing']==i][:][3].values, df[df['indexing']==i][:][3],i)
plt.show()

plotter(df[df['indexing']=="All Kinds of Hotels False"][:][3].values, df[df['indexing']=="All Kinds of Hotels False"][:][3],"All Kinds of Hotels False")
plt.show()

plt.clf()