#!/usr/bin/env python
import sys, math, random
# mapper for 2d points
class Point:
    # Instance variables
    # self.coords is a list of coordinates for this Point
    # self.n is the number of dimensions this Point lives in (ie, its space)
    # self.reference is an object bound to this Point
    # Initialise new Points
    def __init__(self, coords, reference=None):
        self.coords = coords
        self.n = len(coords)
        self.reference = reference
    # Return a string representation of this Point
    def __repr__(self):
        return str(self.coords)

def main(args):
    # variable to hold map output
    outputmap = ''
    #centroid details are stored in an input file
    #cfile = " ~/hadoop/centroidinput.txt"
    cfile = "centroidinput.txt"
    infilecen = open(cfile,"r")
    centroid = infilecen.readline()
    #print centroid
    for point in sys.stdin:
        # remove leading and trailing whitespace
        point = point.strip()
        # split the line into words
        points = point.split()
        #yield point.split()
        # remove leading and trailing whitespace
        centroid = centroid.strip()
        # split the centroid into centroids
        centroids = centroid.split()
        #outfile = open("mapoutput1.txt","w")
        # increase counters
    for value in points:
        dist = 0
        minDist = 999999
        bestCent = 0
        for c in centroids:
            # split each co-ordinate of the centroid and the point
            cSplit = c.split(',')
            vSplit = value.split(',')
            # To handle non-numeric value in centroid or input points
            try:
                #dist = abs(int(cSplit[0]) - int(vSplit[0]))
                dist = (((int(cSplit[0]) - int(vSplit[0]))**2) +
                ((int(cSplit[1]) - int(vSplit[1]))**2))**.5
                #print dist
                if dist < minDist:
                    minDist = dist
                    bestCent = c
            except ValueError:
                pass
            #outfile.writelines(outputmap)
            print('%s\t%s' % (bestCent, value))
            # outfile = open("mapoutput1.txt","w")
            # outfile.writelines(outputmap)
            # outfile.close()

def makeCentroids():
    cfile = open("centroidinput.txt", "r")
    seed = cfile.readline()
    cfile.close()
    #seed = "43,211 130,62 387,253 454,734 545,920"
    FILE1 = open("centroidpoints.txt","w")
    # Write all the lines for seed = "15,40 110,10 25,9 69,678 240,78"
    FILE1.writelines(seed)
    FILE1.close()
    return seed

if __name__ == "__main__": main(sys.argv)