#!/usr/bin/env python
import sys, math, random
import os
import shutil
import time
# starting point for the system
#This program generates the required number of data points and calls convergence
#check and mapreduce.
def main(args):
    DP = open("datapoints.txt","w").close()
    num_points=0
    pointx = ''
    while num_points <10000000:
        num_iter =0
        num_points +=1000000
        num_gen = 1000000
        # Create num_points random Points in n-dimensional space
        num_gen, coords, lowerx, upperx, lowery, uppery = int(num_points*.5), 2,1, 900, 1,900
        pointx = makeRandomPoint(num_gen, lowerx, upperx, lowery, uppery)
        num_gen, coords, lowerx, upperx, lowery, uppery = int(num_points*.1), 2,50, 350, 50,500
        pointx = makeRandomPoint(num_gen, lowerx, upperx, lowery, uppery)
        num_gen, coords, lowerx, upperx, lowery, uppery = int(num_points*.1), 2,410, 600,10,550
        pointx = makeRandomPoint(num_gen, lowerx, upperx, lowery, uppery)
        num_gen, coords, lowerx, upperx, lowery, uppery = int(num_points*.1), 2,600, 890, 600,900
        pointx = makeRandomPoint(num_gen, lowerx, upperx, lowery, uppery)
        num_gen, coords, lowerx, upperx, lowery, uppery = int(num_points*.1), 2,100, 500,650,900
        pointx = makeRandomPoint(num_gen, lowerx, upperx, lowery, uppery)
        num_gen, coords, lowerx, upperx, lowery, uppery = int(num_points*.1), 2,650, 880,50,470
        pointx = makeRandomPoint(num_gen, lowerx, upperx, lowery, uppery)
    while num_iter <5:
        num_iter +=1
        filename = "centroid2d_{0}.txt".format(num_iter)
        shutil.copy(filename,"centroidinput.txt")
        datafilename = "datapoints{0}.txt".format(num_points)
        shutil.copy(datafilename,"datapoints.txt")
        kmeans.main(num_points)
        # final clustering of the datapoints
        os.system("python mapperfinal.py")
        #Renaming and finalising of files
        newfilename = "statistics_{0}.txt".format(num_points)
        os.rename("statistics.txt", newfilename)
        newname = "datapoints{0}.txt".format(num_points)
        shutil.copy("datapoints.txt", newname)
        newname = "cluster1_{0}.txt".format(num_points)
        os.rename("cluster1.txt", newname)
        newname = "cluster2_{0}.txt".format(num_points)
        os.rename("cluster2.txt", newname)
        newname = "cluster3_{0}.txt".format(num_points)
        os.rename("cluster3.txt", newname)
        newname = "cluster4_{0}.txt".format(num_points)
        os.rename("cluster4.txt", newname)
        newname = "cluster5_{0}.txt".format(num_points)
        os.rename("cluster5.txt", newname)
        newname = "cluster6_{0}.txt".format(num_points)
        os.rename("cluster6.txt", newname)
# DP = open("datapoints.txt","w").close()
    num_points=0
    pointx = ''
    while num_points <10000000:
        num_points +=1000000
        num_gen = 1000000
        num_iter = 0
# # Create num_points random Points in n-dimensional space
        num_gen, coords, lowerx, upperx, lowery, uppery,lower,upper = int(num_points*.4), 2,1, 900, 1,900,1,900
        pointx = makeRandom3dPoint(num_gen, lowerx, upperx, lowery, ppery,lower,upper)
        num_gen, coords, lowerx, upperx, lowery, uppery,lower,upper = int(num_points*.1), 2,50, 300, 50,450,50,600
        pointx = makeRandom3dPoint(num_gen, lowerx, upperx, lowery,uppery,lower,upper)
        num_gen, coords, lowerx, upperx, lowery, uppery,lower,upper = int(num_points*.1), 2,410, 600,10,550,100,200
        pointx = makeRandom3dPoint(num_gen, lowerx, upperx, lowery,uppery,lower,upper)
        num_gen, coords, lowerx, upperx, lowery, uppery,lower,upper = int(num_points*.1), 2,600, 890, 0,200,10,300
        pointx = makeRandom3dPoint(num_gen, lowerx, upperx, lowery,uppery,lower,upper)
        num_gen, coords, lowerx, upperx, lowery, uppery,lower,upper = int(num_points*.1), 2,100, 500,650,900,600,700
        pointx = makeRandom3dPoint(num_gen, lowerx, upperx, lowery,uppery,lower,upper)
        num_gen, coords, lowerx, upperx, lowery, uppery,lower,upper = int(num_points*.1), 2,650, 880,50,470,800,900
        pointx = makeRandom3dPoint(num_gen, lowerx, upperx, lowery,uppery,lower,upper)
        num_gen, coords, lowerx, upperx, lowery, uppery,lower,upper = int(num_points*.1), 2,800, 900,750,900,800,900
        pointx = makeRandom3dPoint(num_gen, lowerx, upperx, lowery,uppery,lower,upper)
    while num_iter <5:
        num_iter +=1
        print(num_iter)
        filename = "centroid3d_{0}.txt".format(num_iter)
        shutil.copy(filename,"centroidinput.txt")
        datafilename = "datapoints3d{0}.txt".format(num_points)
        shutil.copy("datapoints.txt", datafilename)
        kmeans3d.main(num_points)
        # final clustering of the datapoints
        #Renaming and finalising of files
        newfilename = "statistics3d_{0}_{1}.txt".format(num_points, num_iter)
        os.rename("statistics3d.txt", newfilename)
        newname = "cluster3d1_{0}.txt".format(num_points)
        os.rename("cluster3d1.txt", newname)
        newname = "cluster3d2_" + num_points +".txt"
        os.rename("cluster3d2.txt", newname)
        newname = "cluster3d3_" + num_points +".txt"
        os.rename("cluster3d3.txt", newname)
        newname = "cluster3d4_" + num_points +".txt"
        os.rename("cluster3d4.txt", newname)
        newname = "cluster3d5_" + num_points +".txt"
        os.rename("cluster3d5.txt", newname)
        newname = "cluster3d6_" + num_points +".txt"
        os.rename("cluster3d6.txt", newname)

def makeRandomPoint(num_points, lowerx, upperx, lowery, uppery):
    datapoints = ''
    coordX = ''
    coordY = ''
    for i in range(num_points):
        coordX=random.randint(lowerx, upperx)
        coordY=random.randint(lowery, uppery)
        datapoints +=str(coordX)+','+str(coordY)
        datapoints += ' '
        DP = open("datapoints.txt","a")
        # Write all the lines for datapoints at once:
        DP.writelines(datapoints)
        DP.close()
    return datapoints

def makeRandom3dPoint(num_points, lowerx, upperx, lowery, uppery,lowerz,upperz):
    datapoints = ''
    coordX = ''
    coordY = ''
    coordZ = ''
    for i in range(num_points):
        coordX = random.randint(lowerx, upperx)
        coordY = random.randint(lowery, uppery)
        coordZ = random.randint(lowerz, upperz)
        datapoints += coordX + ',' + coordY + ',' + coordZ
        datapoints += ' '
        DP = open("datapoints.txt", "w")
        # Write all the lines for datapoints at once:
        DP.writelines(datapoints)
        DP.close()
    return datapoints

if __name__ == "__main__": main(sys.argv)