# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:42:43 2019

@author: Kem
"""


from joblib import load, dump

##############################################################################
#

from sklearn.naive_bayes import MultinomialNB


x_train = load("x_train.joblib")
y_train = load("y_train.joblib")
x_test = load("x_test.joblib")
y_test = load("y_test.joblib")
clf = MultinomialNB().fit(x_train, y_train)


dump(clf, "testcode.joblib")
