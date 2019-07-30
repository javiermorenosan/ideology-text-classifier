# coding=utf-8

# This script contains methods for training algorithms

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def NBtrain(X_train, Y_train):
    # This method trains a Naive Bayes algorithm.

    nb = MultinomialNB()
    nb = nb.fit(X_train, Y_train)

    return nb


def LRtrain(X_train, Y_train):
    # This method trains a Logistic Regression algorithm.

    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr = lr.fit(X_train, Y_train)

    return lr


def DTtrain(X_train, Y_train):
    # This method trains a Decision Tree algorithm.

    dt = DecisionTreeClassifier()
    dt = dt.fit(X_train, Y_train)

    return dt


def SVMtrain(X_train, Y_train, X_test, Y_test, test_dataset, feature_names):
    # This method trains a Suport Vector Machine algorithm.

    svm = SVC()
    svm = svm.fit(X_train, Y_train)

    return svm
