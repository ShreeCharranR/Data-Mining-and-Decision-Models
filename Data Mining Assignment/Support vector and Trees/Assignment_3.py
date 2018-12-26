# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\Data Mining and Decision Models\\Data Mining Assignment\\pageblock.csv")

"""
   height:   integer.         | Height of the block.
   lenght:   integer.     | Length of the block. 
   area:     integer.    | Area of the block (height * lenght);
   eccen:    continuous.  | Eccentricity of the block (lenght / height);
   p_black:  continuous.  | Percentage of black pixels within the block (blackpix / area);
   p_and:    continuous.        | Percentage of black pixels after the application of the Run Length Smoothing Algorithm (RLSA) (blackand / area);
   mean_tr:  continuous.      | Mean number of white-black transitions (blackpix / wb_trans);
   blackpix: integer.    | Total number of black pixels in the original bitmap of the block.
   blackand: integer.        | Total number of black pixels in the bitmap of the block after the RLSA.
   wb_trans: integer.          | Number of white-black transitions in the original bitmap of the block.
"""

df.columns= ['height','length','area','eccen','p_black','p_and','mean_tr','blackpix','blackand','wb_trans','classification']

#df.classification = df.classification.astype('object')
df.classification = df.classification-1

cat_column = list(df.select_dtypes(include = ['object']).columns)
num_column = list(df.select_dtypes(include = ['float64','int64']).columns)

def col_meta_data(data):
    objCol = list(data.select_dtypes(include = ['object']).columns)
    numCol = list(data.select_dtypes(include = ['float64','int64']).columns)
    columndetails = []
    for i in objCol:
        columndetails.append({'Column Name':i,'Type' : 'Object' ,'Number of NULL values': float(data[i].isna().sum()),'Number of Unique Values':len(df[i].unique())})
    for i in numCol:
        columndetails.append({'Column Name':i,'Type' : 'Numeric' ,'Number of NULL values': float(data[i].isna().sum()),'Number of Unique Values':len(df[i].unique())})
    return(pd.DataFrame(columndetails))
    
col_meta_data = col_meta_data(df) 

fig,ax = plt.subplots(1,1)
ax.axis([0, 5, 0, 5000])
for i in df.classification.unique():
    y=i+1
    ax.text(i-1,len(df[df.classification==i]), str(len(df[df.classification==i])), transform=ax.transData)
sns.countplot(x=df['classification'], alpha=0.7, data=df)

       
df[num_column].hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8)
#sns.distplot(df_sales.nItems)
#sns.lmplot(x="nItems", y="nCats",data = df_sales)

fig,axes = plt.subplots(nrows=(round(len(num_column)/3)),ncols=3,figsize =(18,12))
for i, ax in enumerate(fig.axes):
    if i < len(num_column):
        #ax.axis([0, max(df[num_column[i]]), 0, 5])
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        sns.distplot(df[num_column[i]], ax=ax)
fig.tight_layout()
plt.show()

fig,axes = plt.subplots(nrows=(round(len(num_column)/3)),ncols=3,figsize =(18,12))
for i, ax in enumerate(fig.axes):
    if i < len(num_column):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.title(i)
        sns.regplot(x=df[num_column[i]], y=df["classification"],ax=ax)
fig.tight_layout()
plt.show()

sns.pairplot(df,diag_kind='kde',vars=['height','length','area','eccen','p_black'],hue="classification")
sns.pairplot(df,diag_kind='kde',vars=['p_and','mean_tr','blackpix','blackand','wb_trans'],hue="classification")


df_corr = df.drop('classification',axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr,#[(df_corr >= 0.5) | (df_corr <= -0.4)],
 cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
 annot=True, annot_kws={"size": 8}, square=True);

fig,axes = plt.subplots(nrows=(round(len(num_column)/3)),ncols=3,figsize =(18,12))
for i, ax in enumerate(fig.axes):
    if i < len(num_column):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.title(i)
        sns.regplot(x=df[num_column[i]], y=df["classification"],ax=ax)
fig.tight_layout()
plt.show()

fig,axes = plt.subplots(nrows=(round(len(num_column)/3)),ncols=3,figsize =(18,12))
for i, ax in enumerate(fig.axes):
    if i < len(num_column):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.title(i)
        sns.boxplot(y=df[num_column[i]], x=df["classification"],ax=ax)
fig.tight_layout()
plt.show()



###############################################################################

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

#gather features
features = "+".join(df.drop(['classification','height','length','blackand'],axis=1).columns)

# get y and X dataframes based on this regression:
y, X = dmatrices('classification ~' + features, df, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.round(1)

###############################################################################

# Logistics regression
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(multi_class ='ovr')


X = df.iloc[:, :10].values
y = df.iloc[:, -1].values
y = y.reshape(len(y),1)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0,stratify = y)

logit_model = logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0] + cm[1,1] +cm[2,2]+cm[3,3]+cm[4,4])/len(y_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LogisticRegression(multi_class ='ovr'),X =X_train, y=y_train,cv=10 )
print(accuracies.mean())
print(accuracies.std())

###############################################################################
# SVC analysis

X = df.iloc[:, :10].values
y = df.iloc[:, -1].values
y = y.reshape(len(y),1)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0] + cm[1,1] +cm[2,2]+cm[3,3]+cm[4,4])/len(y_test)

#Applying cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X =X_train, y=y_train,cv=10 )
print(accuracies.mean())
print(accuracies.std())

###############################################################################

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0,oob_score=True)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0,0] + cm[1,1] +cm[2,2]+cm[3,3]+cm[4,4])/len(y_test)

#Applying cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X =X_train, y=y_train,cv=10 )
print(accuracies.mean())
print(accuracies.std())

plt.bar(df.drop("classification",axis=1).columns, classifier.feature_importances_)
plt.xticks(rotation=90)
plt.show()

## Feature Importance
# Get feature importances from our random forest model
importances = classifier.feature_importances_

# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels 
labels = np.array(df.drop('classification',axis=1).columns)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)

# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()

#################################################################################
# Feature selection
#################################################################################

plt.bar(df.drop("classification",axis=1).columns, classifier.feature_importances_)
plt.xticks(rotation=90)
plt.show()

#XGBoost Classifier - Feature importance

# plot feature importance using built-in function

from xgboost import XGBClassifier
from xgboost import plot_importance

# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
res = plot_importance(model)
plt.show()


##Correlation selection

feature_name = df.columns.tolist()

def cor_selector(X, y):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-100:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

cor_support , cor_feature = cor_selector( df.iloc[:,:-1], df.iloc[:,-1])

""" Chi -2 feature selection """

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

X = MinMaxScaler().fit_transform(df.iloc[:,:-1].values)
y = df.iloc[:,-1].values.astype("int64")

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
# summarize scores

print(fit.scores_)
features_importance_list_chi_sq = list(np.array(df.columns[np.argsort(fit.scores_)]))
features_importance_list_chi_sq.reverse()

""" Recurrent Feature Elemination """

# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# feature extraction
model = LogisticRegression()
rfe = RFE(estimator = model,n_features_to_select = 3,step=5)
fit = rfe.fit(X, y)

df_train_x = df.iloc[:,:-1]
df_train_y = df.iloc[:,-1]

rfe_support = fit.get_support()
rfe_feature = df_train_x.loc[:,rfe_support].columns.tolist()
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print("The column name:%s" % df.drop("classification",axis=1).columns)

""" RFE using Random Forest Classifier """

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), '1.25*median')
embeded_lr_selector.fit(df_train_x.values, df_train_y.values)

###############################################################################
""" Random Forest tunining
1. tree splitting strategies
2. accuracy measures

* Minimum samples for a node split
* Minimum samples for a terminal node (leaf)
* Maximum depth of tree (vertical depth)
* Maximum number of terminal nodes
* Maximum features to consider for split

"""
from sklearn.model_selection import validation_curve, learning_curve
train_sizes = list(range(1,4002,100))

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = RandomForestClassifier(), X = df.iloc[:, :10].values,
                                                   y = df.iloc[:, -1].values, train_sizes = train_sizes, cv = 5,
                                                   scoring = 'neg_mean_squared_error',shuffle=True)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)

import matplotlib.pyplot as plt
plt.style.use('seaborn')

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()



#################
"""max_features"""
########

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[1, 10,25,50],'n_jobs':[1, 5,10],'min_samples_split':[2,5,10,50]}
classifier = RandomForestClassifier()
gs_clf = GridSearchCV(classifier,parameters,cv =5)
gs_clf.fit(df.iloc[:,:10].values, df.iloc[:,-1].values)
print("The best parameters are ",gs_clf.best_estimator_)

###############################################################################
"""OOB score"""
###############################################################################
from collections import OrderedDict
from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0,oob_score=True)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

RANDOM_STATE = 123

# Generate a binary classification dataset.
X, y = make_classification(n_samples=500, n_features=10,
                           n_clusters_per_class=1, #n_informative=15,
                           random_state=RANDOM_STATE)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 15
max_estimators = 175

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


################################################################################
"""PCA"""
##################################################################################


from sklearn.preprocessing import StandardScaler

# Separating out the features
x = df.iloc[:,:-1].values
# Separating out the target
y = df.iloc[:,-1].values

x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['classification']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1,2,3,4]
colors = ['r', 'g', 'b','y','o']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['classification'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'],c = color, s = 50)
ax.legend(targets)
ax.grid()

plt.scatter(principalComponents[:, 0], principalComponents[:, 1],
            c=df.classification, edgecolor='none', alpha=1.0,
            cmap=plt.cm.get_cmap('Spectral',5))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

###############################################################################
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Logistics regression
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(multi_class ='multinomial',solver='newton-cg')


X = df.iloc[:, :10].values
y = df.iloc[:, -1].values
y = y.reshape(len(y),1)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0,stratify = y)

logit_model = logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0] + cm[1,1] +cm[2,2]+cm[3,3]+cm[4,4])/len(y_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LogisticRegression(multi_class ='multinomial'),X =X_train, y=y_train,cv=10 )
print(accuracies.mean())
print(accuracies.std())

from sklearn import metrics
y_score = logistic_regression.decision_function(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_score.shape[1]):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


