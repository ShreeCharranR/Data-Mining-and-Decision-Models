#!/usr/bin/env python
import sys, math, random
import clustering
import removingoutliers
#from collections import defaultdict
from defdict import *
# this program performs the final data point assignment
#and calls removingoutliers.py to remove outliers
def main(args):
    point2centroid = defaultdict(list)
    # variable to hold map output
    infiledat = open("datapoints.txt", "r")
    datap = infiledat.readline()
    dpoints = datap.split()
    cfile = "centroidinput.txt"
    infilecen = open(cfile,"r")
    centroid = infilecen.readline()
    #print centroid
    for point in dpoints:
        #remove leading and trailing whitespace
        point = point.strip()
        #split the line into words
        points = point.split()
        #remove leading and trailing whitespace
        centroid = centroid.strip()
        #split the centroid into centroids
        centroids = centroid.split()
        #increase counters
        for value in points:
            dist = 0
            minDist = 999999
            bestCent = 0
            for c in centroids:
                #print c
                # split each co-ordinate of the centroid and the point
                cSplit = c.split(',')
                vSplit = value.split(',')
                # To handle non-numeric value in centroid or input points
            try:
                dist = (((int(cSplit[0]) - int(vSplit[0]))**2) + ((int(cSplit[1]) - int(vSplit[1]))**2))**.5
                #print dist
                if dist < minDist:
                    minDist = dist
                    bestCent = c
            except ValueError:
                pass
         #print '%s\t%s' % (bestCent, value)
        point2centroid[bestCent].append(value)
        #clustering.main(point2centroid)
        removingoutliers.main(point2centroid)
        #print point2centroid

if __name__ == "__main__": main(sys.argv)