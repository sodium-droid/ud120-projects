#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

kernels = {"1":"linear", "2": "rbf", "3": "sigmoid"}

try:
    
    while True:
        print("""Please enter your choice kernel:
        1. Linear
        2. RBF
        3. Sigmoid""")

        choice = input("Enter a number or q (to quit): ")
        match choice:
            case "1":
                print("Linear selected\n")
                break
            case "2":
                print("RBF selected\n")
                break
            case "3":
                print("Sigmoid selected\n")
                break
            case "q":
                print("Bye for now!\n")
                sys.exit()
            case _:
                print("choice kernel can only be 1 - 3\n")

except SystemExit:
    pass

kernel = kernels[choice]

#clf = svm.SVC()

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

print(f"Train data reduced\nfeatures_train is: {len(features_train)}\nlabels_train is: {len(labels_train)}\n")

for c in [1, 10, 100, 1000, 10000]:
    print("Classifier with C as:", c)
    clf = svm.SVC(kernel = kernel, C=c)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")

    t0 = time()
    label = clf.predict(features_test)
    print("Predicting Time:", round(time()-t0, 3), "s")
    print("Prediction Accuracy:", clf.score(features_test, labels_test), "\n")
#########################################################
