# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:56:11 2022
@author: mumu
"""
from sklearn import svm 
import numpy as np  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import warnings
warnings.filterwarnings("ignore")
model = svm.SVC()
params = [
{'kernel':['linear'],'C':[1,10,100,1000]},
{'kernel':['poly'],'C':[1,10],'degree':[1,2,3]},
{'kernel':['rbf'],'C':[1,10,100,1000], 
 'gamma':[1,0.1, 0.01, 0.001]}]
cv1 = ShuffleSplit(n_splits=10, test_size=0.3, random_state = 1)
model = GridSearchCV(estimator=model, param_grid=params, cv=cv1)	 
path = 'E:/changweike/Github_desk/PCnet-and-STC/classification/SVM/data/'
files = os.listdir(path)
for file in files[2:5]:
    print(file)
    
    A = np.load(path+file)
    data = A['data']
    label = A['label']
    model.fit(data, label)
    clf = model.best_estimator_
    
    num = 1000
    acc_mean = []
    f1_mean = []
    recall_mean = []
    precision_mean = []
    for i in range(num):
        data,label = shuffle(data,label)
        cv = ShuffleSplit(n_splits=10, test_size=0.3)
        
        scores = cross_val_score(clf, data, label, cv=cv)
        precision = cross_val_score(clf, data, label, cv=cv, scoring = 'precision')
        recall = cross_val_score(clf, data, label, cv=cv, scoring = 'recall')
        f1 = cross_val_score(clf, data, label, cv=cv, scoring = 'f1')
        acc_mean.append(scores.mean())
        f1_mean.append(f1.mean())
        recall_mean.append(recall.mean())
        precision_mean.append(precision.mean())
    acc_mean = np.array(acc_mean)
    f1_mean = np.array(f1_mean)
    recall_mean = np.array(recall_mean)
    precision_mean = np.array(precision_mean)
    print(acc_mean.mean(),f1_mean.mean(),recall_mean.mean(),precision_mean.mean())
    print('#####################################')