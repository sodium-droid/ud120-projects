#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from mlxtend.plotting import plot_decision_regions as pdr
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
clf = tree.DecisionTreeClassifier()

t0 = time()
clf.fit(features_train, labels_train)
print("\nTraining Time:", round(time() - t0,3), "s")

t0 = time()
label = clf.predict(features_test)
print("Predicting Time:", round(time() - t0,3), "s\n")

#########################################################


