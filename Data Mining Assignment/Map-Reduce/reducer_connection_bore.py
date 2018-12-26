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
    line= line.split(',')
    try:
        consumption_freq[(line[0],line[1],line[2])] +=1
    except:
        consumption_freq[(line[0],line[1],line[2])] =1

column = sorted(consumption_freq.keys())
for consumption in column:
    f.write("%s,%s,%s,%s\n"%(consumption[0],consumption[1],consumption[2], consumption_freq[consumption]))
f.close()