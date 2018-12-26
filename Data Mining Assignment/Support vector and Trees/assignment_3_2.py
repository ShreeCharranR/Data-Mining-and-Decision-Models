# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:19:40 2018

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_helper_2.data_helper import completeanalysis
import statsmodels.api as sm

import os
os.chdir("C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\vis\\check")

df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\pageblock.csv")

df.columns= ['height','length','area','eccen','p_black','p_and','mean_tr','blackpix','blackand','wb_trans','classification']
df.classification = df.classification-1

from data_helper_2.data_helper import completeanalysis
df_ca = completeanalysis(df)

meta_data = df_ca.col_meta_data()
df_ca.response_distribution()
df_ca.distribution_plots()

plt.savefig('C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\vis\\distribution.pdf', bbox_inches='tight')

df_ca.correlation_plot()
plt.savefig('C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\vis\\corr_plot.jpg', bbox_inches='tight')
df_ca.correlation_plot(low=-0.5,high=0.5)
plt.savefig('C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\vis\\corr_plot_5_5.jpg', bbox_inches='tight')
df_ca.pairplot()
plt.savefig('C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\vis\\pairplot.pdf', bbox_inches='tight')
df_ca.pairplot(['area', 'mean_tr'])

df_ca.numerical_plots()
plt.savefig('C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\vis\\numeric.jpg', bbox_inches='tight')
df_ca.boxplots()
plt.savefig('C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\vis\\boxplot.pdf', bbox_inches='tight')

#df_ca.save_img()
VIF = df_ca.variance_explained()

df_ca.delete_column(['blackand'])


from data_helper_2.data_helper import completeanalysis

df_ca = completeanalysis(df.drop('blackand',axis=1))
VIF = df_ca.variance_explained()
#df_ca.compare_algo()



df_ca.random_forest_classifier()

df_ca.svc_param_selection()

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
from sklearn.ensemble import RandomForestClassifier

results =[]

result_dict={}
 
for key in ['log2','auto','none']:
    results=[]
    for i in range(15,200):
        classifier = RandomForestClassifier(n_estimators = i,criterion = 'entropy',oob_score=True , random_state = 0,max_features = key)
        classifier.fit(X,y)
        results.append(1-classifier.oob_score_)
    result_dict[key].append(results)

    
  
plt.plot(range(15,200),result_dict['log2'],label ='log2')
plt.plot(range(15,200),result_dict['auto'],label ='auto')
plt.plot(range(15,200),result_dict['none'],label ='none')
plt.legend(loc="upper right")
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.show()

    
ensemble_clfs = [
        ("RandomForestClassifier, max_features='auto'",
            RandomForestClassifier(n_estimators=100,
                                   warm_start=True, oob_score=True,
                                   max_features="auto",
                                   random_state=0)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(n_estimators=100,
                                   warm_start=True, max_features='log2',
                                   oob_score=True,
                                   random_state=0)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(n_estimators=100,
                                   warm_start=True, max_features=None,
                                   oob_score=True,
                                   random_state=0))
    ]
    
# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 200

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [80, 90, 100, 110],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [50, 100, 150, 200, 250]
}

classifier = RandomForestClassifier(n_estimators=100, warm_start=True, oob_score=True, random_state=0)

grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search.fit(X, y)  

grid_search.best_params_
best_grid = grid_search.best_estimator_


from data_helper_2.data_helper import completeanalysis

df_ca = completeanalysis(df.drop('blackand',axis=1))

df_ca.get_roc_auc()