# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:22:23 2018

@author: Lenovo
"""

import sys
consumption_freq ={}
f = open("reducer_output.csv","w+")   
for line in sys.stdin:
    line=line.strip()
    try:
        consumption_freq[line] +=1
    except:
        consumption_freq[line] =1

for consumption in consumption_freq.keys():
    f.write("%s,%s\n"%(consumption, consumption_freq[consumption]))
f.close()