# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:21:25 2018

@author: Karthick
"""

import simpy
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt


class Monitor(object):
    passengers = []
    def addPass(self, p):
        self.passengers.append(p)
    
class Queue:
    """ Lift Queue """
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()
    
    def dequeue_n(self, n):
        send_list = []
        for i in range(n):
            send_list.append(self.dequeue())
        return send_list

    def size(self):
        return len(self.items)
    


class Elevator(object):
    """Elevator functions."""

    def __init__(self):
        #self.travel_time = travel_time
        self.passenger_list = list()
        self.capacity = 15
        self.floor = 0
        self.up_floors_list = list()
    
    def floor_7(self):
        self.floor = 7
        
    def get_floor(self, d, default=None):
        rev = (len(d) - idx for idx, item in enumerate(reversed(d), 1) if item)
        return 1+next(rev, default)
        
    def find_travel_time(self, t):
        self.trip_up = {}
        for j in range(2,8):
            self.trip_up[j] = 0
        for i in self.passenger_list:
            i.boarding_time = t
            i.deboarding_time = t + ((i.floor_choice)-1)*15
            i.request_time = i.deboarding_time + i.work_time
            if self.trip_up[i.floor_choice]==0:
                self.trip_up[i.floor_choice] = 1
            else:
                self.trip_up[i.floor_choice]+=1
        self.drop_floor = self.get_floor(list(self.trip_up.values()))
        deliver_floor = sum(np.array(list(self.trip_up.values()))!= 0)
        return 5*(len(set(self.up_floors_list))) + 10*(max(self.up_floors_list) - 1)
    
    def get_upFloors(self):
        self.up_floors_list = []
        for i in self.passenger_list:
            self.up_floors_list.append(i.floor_choice)
        
    def get_down(self):
        top_floor = max(self.up_floors_list)
        return 10*top_floor
             
            
    def ferry(self,env,main_queue):
        """ Elevator car ferry """
        i = 0
        while (i == 0):
            self.add_people(main_queue)
            self.get_upFloors()
            print("The trip starts at ground floor by lift", G.elCount, env.now)
            print("Number of people in main queue", main_queue.size())
            self.update_floor_lists()
            yield env.timeout(self.find_travel_time(env.now))
            self.remove_people()
            tf = 7
            for fl in range(tf, 1, -1):
                #print(fl)
                if fl in G.workingOnFloor.keys():
                    if (len(G.workingOnFloor[fl]) > 0):
                        for j in G.workingOnFloor[fl]:
                            if env.now > j.request_time:
                                if j.floor_choice in G.waitingOnFloor.keys():
                                    G.waitingOnFloor[j.floor_choice].append(j)
                                    G.workingOnFloor[j.floor_choice].remove(j)
                                else:
                                    G.waitingOnFloor[j.floor_choice] = [j]
                                    G.workingOnFloor[j.floor_choice].remove(j)
                if fl in G.waitingOnFloor.keys():
                    if ((len(G.waitingOnFloor[fl]) > 0) and (len(self.passenger_list)< self.capacity)):
                        G.waitingOnFloor[fl] = self.add_down(G.waitingOnFloor[fl], env.now)
                        yield env.timeout(15)
                else:
                    yield env.timeout(10)
                        
            
            #tDown = self.get_down()
            #yield env.timeout(tDown)
            print("The trip up is finished by lift", G.elCount, env.now)
            G.elCount = G.elCount - 1
            i = i + 1
            
            #self.floor_7()
      
    def add_down(self, waitList, t):
        original_count = len(self.passenger_list)
        nInLift = self.capacity - (original_count)
        self.passenger_list = self.passenger_list + waitList[:nInLift]
        for j in waitList[:nInLift]:
            j.retBoarding_time = t
            j.departure_time = t + ((j.floor_choice)-1)*15
            G.passList.addPass(j)
        waitList = waitList[nInLift:]
        return waitList
    
    def add_people(self,queue_temp):
        """Adding people from the queue"""
        self.passenger_list = []
        if (queue_temp.size() < self.capacity):
            nInLift = queue_temp.size()
        else:
            nInLift = self.capacity
        self.passenger_list = self.passenger_list + queue_temp.dequeue_n(nInLift)
        
    def remove_people(self):
        self.passenger_list = []
    
    def update_floor_lists(self):
        for i in self.passenger_list:
            if i.floor_choice in G.workingOnFloor.keys():
                G.workingOnFloor[i.floor_choice].append(i)
            else:
                G.workingOnFloor[i.floor_choice] = [i]
                
    def update_waiting_lists(t, floorNo):
        for i in G.workingOnFloor[floorNo]:
            if t > (i.deboarding_time + i.work_time):
                if i.floor_choice in G.waitingOnFloor.keys():
                    G.waitingOnFloor[i.floor_choice].append(i)
                    G.workingOnFloor[i.floor_choice].remove(i)
                else:
                    G.waitingOnFloor[i.floor_choice] = [i]
                    G.workingOnFloor[i.floor_choice].remove(i)
                    

class G:
    elCount = 0
    TotalPerson = 0
    workingOnFloor = {}
    waitingOnFloor = {}
    passList = Monitor()
    maxPerson = 5000
    down = 0


class Passenger:
    """ properties of the passenger"""
    def __init__(self):
        self.name = "P" + str(G.TotalPerson)
        self.floor_choice = np.random.choice([2,3,4,5,6,7])
        self.work_time = random.expovariate(1/(59*60))
        self.arrival_time = 0
        self.boarding_time = 0
        self.deboarding_time = 0
        self.request_time = 0
        self.retBoarding_time = 0
        self.departure_time = 0

class Elevatorsim(object):
    def run(self, seed):
        random.seed(seed)
        env = simpy.Environment()
        main_queue = Queue()
        s = Scheduling()
        env.process(s.generate(env,main_queue))
        env.run(until=8*60*60) #8 hours of working time

    
class Scheduling:
    """ Source generates passengers 
    adding people to queue"""
    
    def generate(self, env , main_queue):
        #i = 0
        while True:
            arrival_rate = 1/(1.9557*60)
            t = random.expovariate(arrival_rate)
            yield env.timeout(t)
            if (G.TotalPerson < G.maxPerson):
                G.TotalPerson += 1
                pass_person = Passenger()
                pass_person.arrival_time = env.now
                main_queue.enqueue(pass_person)
                #print("Total number of people in the queue" , main_queue.size())
            
            if (((main_queue.size() > 0) and (G.elCount < 3)) or (G.down == 1)):
                e = Elevator()
                G.elCount = G.elCount + 1
                env.process(e.ferry(env,main_queue))

            #i += 1
            
            
elevator1 = Elevatorsim()
elevator1.run(1)

###############################################################################


DT = []
RT = []
BT = []
AT = []
PASS = []
FLOOR_C =[]
WORKTIME = []
for i in G.passList.passengers:
    DT.append(i.departure_time)
    BT.append(i.deboarding_time)
    RT.append(i.request_time)
    AT.append(i.arrival_time)
    PASS.append(i.name)
    FLOOR_C.append(i.floor_choice)
    WORKTIME.append(i.work_time)

UPResTime = np.array(BT)  - np.array(AT)
DNResTime = np.array(DT)  - np.array(RT) 
################################################################################



Final_results = pd.DataFrame() 

Final_results["Passenger"] = PASS
Final_results["FloorChoice"] = FLOOR_C
Final_results["WorkTime"] = WORKTIME
Final_results["ArrivalTime"] = AT
Final_results["DeboardingAtFloor"] = BT
Final_results["RequestTime"] = RT
Final_results["DepartureTime"] = DT

###############################################################################

responsetime_down = simulation_test(pd.DataFrame(DNResTime)[0])

responsetime_down.plots()
responsetime_up.distribution_check('norm')
#responsetime_up.distribution_check('expon')
responsetime_down.descriptive_stat()
responsetime_down.moving_average_curve()


responsetime_up = simulation_test(pd.DataFrame(UPResTime)[0])

responsetime_up.plots()
responsetime_up.distribution_check('norm')
responsetime_up.distribution_check('expon')
responsetime_up.descriptive_stat()
responsetime_up.moving_average_curve()

