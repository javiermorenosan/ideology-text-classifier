# coding=utf-8

# This script implements several experiments.

import features
import evaluation
import models
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def RemoveCenterExperimentBoW():

    # This experiment evaluates the performance of the model in
    # relation with the amount of moderate tests we remove. To do
    # so, we calculate the accuracy of the model when we eliminate
    # some percentage of the DW Nominate space around zero.
    # Considering the highest positive and negative observations,
    # we extract a 10% more of the Nomi-nate scale from zero to
    # these values on each iteration, and calculate the accuracy
    # over the rest of the observations.

    centers = [None, [-0.068, 0.091], [-0.136, 0.182], [-0.204, 0.273], [-0.272, 0.364],
               [-0.34, 0.455], [-0.408, 0.546], [-0.476, 0.637], [-0.544, 0.728], [-0.612, 0.819]]
    #centers = [[-0.2, 0.2]]
    nb_accuracies = []
    lr_accuracies = []
    t0 = time.time()
    t1 = time.time()
    for i in range(0, len(centers)):
        print(i)
        X_train, Y_train, X_test, Y_test, vectorizer, feature_names = features.ExtractWordFeatures(
            "speeches_110_dwnominate_nonames.txt", "speeches_112_dwnominate_nonames.txt", vectorizer_type="CountVectorizer", ngrams=None, balance_dataset=False, remove_center_interval=centers[i])
        print(len(Y_train))
        print(len(Y_test))
        print("NB")
        nb = models.NBtrain(X_train, Y_train)
        (accuracy, precision, recall) = evaluation.Test(X_test, Y_test, nb)
        nb_accuracies.append(accuracy)
        print("LR")
        lr = models.LRtrain(X_train, Y_train)
        accuracy, precision, recall = evaluation.Test(X_test, Y_test, lr)
        lr_accuracies.append(accuracy)
        t2 = time.time()
        print(t2-t1, " seconds")
        t1 = time.time()
    t3 = time.time()
    print("Total time: ", t3-t0)

    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.plot(x, nb_accuracies, color="#2D00B7", label="Naive Bayes")
    plt.title("Accuracy and center removal relation")
    plt.xlabel("Percentage of DW Nominate removed")
    plt.ylabel("Accuracy")
    plt.plot(x, lr_accuracies, color="#64EEBA", label="Logistic Regression")
    plt.legend()

    return nb_accuracies, lr_accuracies


def crossValidationExperiment():

    # This method implements a experiment in which we
    # apply a 10-fold cross validation technique to the
    # train dataset for both, Naive Bayes and Logistic Regression
    # models.

    (X_train, Y_train, X_test, Y_test, vectorizer, feature_names) = features.ExtractWordFeatures("speeches_110_dwnominate_nonames.txt",
                                                                                                 "speeches_112_dwnominate_nonames.txt", vectorizer_type="CountVectorizer", ngrams=None, balance_dataset=False, remove_center_interval=None)
    nb = MultinomialNB()
    scores_nb = cross_val_score(nb, X_train, Y_train, cv=10)
    score_nb = np.average(scores_nb)
    print("Accuracy NB: ", score_nb)

    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    scores_lr = cross_val_score(lr, X_train, Y_train, cv=10)
    score_lr = np.average(scores_lr)
    print("Accuracy LR: ", score_lr)


def differentSetsCominationsExperiment():

    # In this experiment we introduce more congresses among
    # our datasets. We try different combinations as
    # train-test datasets including data from 109, 110, 111 and 112 Congresses.
    # We carry out the experiment always with the the BoW model and
    # with Naïve Bayes and Logistic Regression algorithms

    congresses = ["109", "110", "111", "112"]

    print("Only one congress for training.")
    for i in range(0, len(congresses)):
        train_congress = congresses[i]
        for j in range(i+1, len(congresses)):
            test_congress = congresses[j]
            print(train_congress, test_congress)
            (X_train, Y_train, X_test, Y_test, vectorizer, feature_names) = features.ExtractWordFeatures("speeches_"+train_congress+"_dwnominate_nonames.txt",
                                                                                                         "speeches_"+test_congress+"_dwnominate_nonames.txt", vectorizer_type="CountVectorizer", ngrams=None, balance_dataset=False, remove_center_interval=None)
            evaluation.Evaluate(X_train, Y_train, X_test, Y_test, "speeches_" +
                                test_congress+"_dwnominate_nonames.txt", feature_names)

    print("Two congresses for training")
    for i in range(0, len(congresses)-1):
        train_congress_1 = congresses[i]
        train_congress_2 = congresses[i+1]
        train_dataset_df_1 = features.datasetTodf(
            "../datasets/train/speeches_"+train_congress_1+"_dwnominate_nonames.txt")
        train_dataset_df_2 = features.datasetTodf(
            "../datasets/train/speeches_"+train_congress_2+"_dwnominate_nonames.txt")
        train_dataset_df = pd.concat(
            [train_dataset_df_1, train_dataset_df_2], ignore_index=True)

        for j in range(i+2, len(congresses)):
            test_congress = congresses[j]
            test_dataset_df = features.datasetTodf(
                "../datasets/test/speeches_"+test_congress+"_dwnominate_nonames.txt")
            print("Train:", train_congress_1,
                  train_congress_2, "Test:", test_congress)
            (X_train, Y_train, X_test, Y_test, vectorizer, feature_names) = features.ExtractWordFeaturesWithDataframes(
                train_dataset_df, test_dataset_df, vectorizer_type="CountVectorizer", ngrams=None, balance_dataset=False, remove_center_interval=None)
            evaluation.Evaluate(X_train, Y_train, X_test, Y_test, "speeches_" +
                                test_congress+"_dwnominate_nonames.txt", feature_names)

    print("Three congresses for training")
    train_congresses = congresses[:3]
    train_datasets = []
    for congress in train_congresses:
        train_dataset_df_i = features.datasetTodf(
            "../datasets/train/speeches_"+congress+"_dwnominate_nonames.txt")
        train_datasets.append(train_dataset_df_i)
    train_dataset_df = pd.concat(train_datasets, ignore_index=True)

    test_congress = congresses[len(congresses)-1]
    test_dataset_df = features.datasetTodf(
        "../datasets/test/speeches_"+test_congress+"_dwnominate_nonames.txt")

    print("Train:", train_congresses, "Test:", test_congress)

    (X_train, Y_train, X_test, Y_test, vectorizer, feature_names) = features.ExtractWordFeaturesWithDataframes(train_dataset_df,
                                                                                                               test_dataset_df, vectorizer_type="CountVectorizer", ngrams=None, balance_dataset=False, remove_center_interval=None)
    evaluation.Evaluate(X_train, Y_train, X_test, Y_test, "speeches_" +
                        test_congress+"_dwnominate_nonames.txt", feature_names)
