# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:45:25 2018

@author: Lenovo
"""

#!/usr/bin/env python
import sys

"""
filepath="consumption_trail.csv"
with open(filepath) as fp:
    for line in fp:
        #Get only one line of the input data at a time
        #Remove leading and trailing whitespace---
        line = line.strip()
        #Split the input line by finding presence of square brackets
        consumption =line.split(',')[4]
        print('%s' % (consumption))
        
"""  
def str_check(char):
    if char not in ["TRUE","FALSE"]:
        return False
    else:
        return True
    
f = open("mapper_output.txt","w+")      
for line in sys.stdin:
    #Get only one line of the input data at a time
    #Remove leading and trailing whitespace---
    line = line.strip()
    line = line.split(',')
    try:
        #Split the input line by finding presence of square brackets
        borewell = line[6]
        consumption = int(line[4])
        if str_check(borewell):
            f.write('%s\t%d\n' %(borewell,consumption))
        else:
            continue
    except ValueError:
    #in case of anything other than number ignore
        continue
f.close()

