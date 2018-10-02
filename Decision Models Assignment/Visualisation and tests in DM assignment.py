# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:52:53 2018

@author: Karthick
"""

import random
import math
import scipy.special as spc
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from scipy.stats import norm,expon, chisquare, probplot

it = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\it.csv",header =None)
st = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\st.csv",header = None)





class simulation_test():
    
    def __init__(self, series):
        self.series = series
        self.mu_estimate = series.mean()
    
    def plots(self):
        plt.clf()
        self.series.plot(title="Time series plot")
        plot_acf(self.series)
        #plot_pacf(self.series)
        plt.show()
        
    def independence_test(self):
        plt.clf()
        plt.scatter(self.series[:999],self.series[1:1000])
        plt.title=("X(t) vs X(t-1)")
        plt.show()
        
    def distribution_check(self,dist):
        #histogram and normal probability plot
         if dist=='norm':
             sns.distplot(self.series, fit=norm);
             (mu, sigma) = norm.fit(self.series)
             print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
             #Now plot the distribution
             plt.legend(['normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                        loc='best')
             plt.ylabel('Frequency')
             #plt.title('Series distribution')
             plt.show()
         if dist=='expon':
             plt.clf()
             sns.distplot(self.series, fit=expon);
             (mu, sigma) = expon.fit(self.series)
             print( '\n mu = {:.2f} '.format(sigma))
             #Now plot the distribution
             plt.legend(['expon dist. ($\mu=$ {:.2f}  )'.format(sigma)],
                         loc='best')
             plt.ylabel('Frequency')
             #plt.title('Series distribution')
             plt.show()
             
    def prob_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        probplot(self.series, plot=ax,dist=expon)
        ax.set_title("Probability plot for exponential distribution")
        plt.show()
            
    def descriptive_stat(self):
        return (self.series.describe())
    
    def binning(self,variate):
        """ """   
        chisq_dict = {}
        chisq_dict["0_2.5"] = 0
        chisq_dict["2.5_5"] = 0
        chisq_dict["5_7.5"] = 0
        chisq_dict["7.5_10"] = 0
        chisq_dict["10_12.5"] = 0
        chisq_dict["12.5_15"] = 0
        chisq_dict["15"] = 0
        for i in variate.values:
            if 0 < i <= 2.5:
                if chisq_dict["0_2.5"] ==0:
                    chisq_dict["0_2.5"] = 1
                else:
                    chisq_dict["0_2.5"] += 1
            if 2.5 < i <= 5:
                if chisq_dict["2.5_5"] == 0:
                    chisq_dict["2.5_5"] = 1
                else:
                    chisq_dict["2.5_5"] += 1
            if 5 < i <= 7.5:
                if chisq_dict["5_7.5"] == 0:
                    chisq_dict["5_7.5"] = 1
                else:
                    chisq_dict["5_7.5"] += 1
            if 7.5 < i <= 10:
                if chisq_dict["7.5_10"] == 0:
                    chisq_dict["7.5_10"] = 1
                else:
                    chisq_dict["7.5_10"] += 1
            if 10 < i <= 12.5:
                if chisq_dict["10_12.5"] == 0:
                    chisq_dict["10_12.5"] = 1
                else:
                    chisq_dict["10_12.5"] += 1
            if 12.5 < i <= 15:
                if chisq_dict["12.5_15"] == 0:
                    chisq_dict["12.5_15"] = 1
                else:
                    chisq_dict["12.5_15"] += 1
            if i > 15:
                if chisq_dict["15"] == 0:
                    chisq_dict["15"] = 1
                else:
                    chisq_dict["15"] += 1
        return list(chisq_dict.values())

    
    def generate_variate(self):
        self.exp_variate = pd.DataFrame(np.random.exponential(self.mu_estimate,10000))
        return self.exp_variate

    def chisq_test_results(self):
        chi_2, p_value = chisquare(self.binning(self.series),self.binning(self.exp_variate))
        return chi_2, p_value
    
    def cal_moving_average(self):
        av=[]
        for i in range(len(self.series)):
            #print(i)
            d = self.series[:i+1]
            d = np.mean(d)
            av.append(d)
        return av
    
    def moving_average_curve(self):
        plt.plot(self.cal_moving_average())
        plt.show() 
        
simtest = simulation_test(it[0])
simtest.plots()
simtest.independence_test()
simtest.distribution_check('norm')
simtest.distribution_check('expon')
simtest.descriptive_stat()
simtest.moving_average_curve()


"""
simtest.generate_variate()
chisq , p_value = simtest.chisq_test_results()  
print (" The pvalue of the chisq test is ..", p_value)
 The pvalue of the chisq test is .. 0.00010482029605263125
"""

        
""" Hypothesis:
The chi square test tests the null hypothesis that the categorical data has the given frequencies.
"""   
simtest.generate_variate()
chisq , p_value = simtest.chisq_test_results()  
print (" The pvalue of the chisq test is ..", p_value) 

"""
chisq
Out[409]: 18.765334537122662

p_value
Out[410]: 0.004578860365336984
"""

simtest_st = simulation_test(st[0])
simtest_st.plots()
simtest_st.prob_plot()
simtest_st.independence_test()
simtest_st.distribution_check('norm')
simtest_st.distribution_check('expon')
simtest_st.descriptive_stat()

#checkign for independence
import os
os.system("start cmd")
"""randomnesstest.R"""


simtest_st.generate_variate()
chisq , p_value = simtest_st.chisq_test_results()  
print (" The pvalue of the chisq test is ..", p_value) 

"""
simtest_st.generate_variate()
chisq , p_value = simtest_st.chisq_test_results()  
print (" The pvalue of the chisq test is ..", p_value)
 The pvalue of the chisq test is .. 0.2291646688715093
"""
