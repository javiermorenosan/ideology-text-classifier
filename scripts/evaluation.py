# coding=utf-8

# This script contains methods for the evaluation of the
# models.

import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def Test(X_test, Y_test, trained_model):

    # Calculates the accuracy of the model, together
    # with the recall and precision of the positive
    # class

    Y_pred = trained_model.predict(X_test)

    accuracy = 0
    precision = 0
    recall = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0, len(Y_test)):
        if Y_pred[i] == 1.0:
            if Y_test[i] == Y_pred[i]:
                TP += 1
            else:
                FP += 1
        else:
            if Y_test[i] == Y_pred[i]:
                TN += 1
            else:
                FN += 1
    if len(Y_test) != 0:
        accuracy = (TP+TN)/(len(Y_test))
    else:
        accuracy = 0.0
    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    else:
        precision = 0.0
    if (TP+FN) != 0:
        recall = TP/(TP+FN)
    else:
        recall = 0.0
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    return accuracy, precision, recall


def Test2(test_dataset, vectorizer, trained_model):

    # Another different implementation for the Test
    # mehod. This one uses the test dataset, the vectorizer
    # instead of the X and Y matrices of the test.

    path_test = "../datasets/test/"
    test_dataset_df = pd.read_csv(
        path_test+test_dataset, sep="|", encoding="latin_1", header=None)
    test_dataset_df.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    test_dataset_df = test_dataset_df.dropna()

    test_dataset_df['ideology'] = test_dataset_df['nominate_dim1'].apply(
        lambda x: 1.0 if (float(x) >= 0) else -1.0)
    test_speeches = test_dataset_df['speech'].values.tolist()
    Y_test = test_dataset_df['ideology'].values.tolist()

    X_test = vectorizer.transform(test_speeches)
    Y_pred = trained_model.predict(X_test)

    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    precision = metrics.precision_score(Y_test, Y_pred)
    recall = metrics.recall_score(Y_test, Y_pred)
    print("Accuracy: ")
    print(accuracy)
    print("Precision: ")
    print(precision)
    print("Recall: ")
    print(recall)
    return accuracy


def MostImportantFeatures(feature_names, trained_model, n, X_train, NB=True):

    # Prints out the n most important features of each class.
    # Algorithms supported are Naive Bayes (NB = True) and
    # Logistic Regression (NB = false).

    feature_avgs = X_train.mean(axis=-2).tolist()
    #feature_names = vectorizer.get_feature_names()
    if NB == True:
        c = (trained_model.feature_log_prob_[
             1]-trained_model.feature_log_prob_[0]).tolist()
        coefs_with_fns = sorted(
            zip(np.multiply(feature_avgs[0], c), feature_names))
    else:
        coefs_by_avg = np.multiply(
            feature_avgs[0][:], trained_model.coef_[0][:])
        coefs_with_fns = sorted(zip(coefs_by_avg, feature_names))

    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])

    positive_features = []
    negative_features = []

    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
        negative_features.append(fn_1)
        positive_features.append(fn_2)

    return negative_features, positive_features


def PlotAccuracies(test_dataset, X_test, Y_test, trained_model, intervals, legend):

    # This method plots the accuracy of the model in relation with the
    # DW Nominate. To do that it calculates the average accuracy per
    # interval of DW Nominate.

    path_test = "../datasets/test/"
    test_dataset_df = pd.read_csv(
        path_test+test_dataset, sep="|", encoding="latin_1", header=None)
    test_dataset_df.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    #test_dataset_df = test_dataset_df[test_dataset_df.bioname != 'UNKNOWN']
    test_dataset_df = test_dataset_df.dropna()
    test_dataset_df['ideology'] = test_dataset_df['nominate_dim1'].apply(
        lambda x: 1.0 if (float(x) >= 0) else -1.0)

    dw_nominates = test_dataset_df['nominate_dim1'].values.tolist()

    Y_pred = trained_model.predict(X_test)

    accuracies = [0] * int(intervals)
    samples_per_interval_pos = [0] * int(intervals/2)
    samples_per_interval_neg = [0] * int(intervals/2)
    hits_per_interval_pos = [0] * int(intervals/2)
    hits_per_interval_neg = [0] * int(intervals/2)

    for i in range(0, len(Y_test)):
        if i < len(dw_nominates):
            dw_nominate = float(dw_nominates[i])
            if dw_nominate > 0:
                interval = int(dw_nominate*intervals/2)
                samples_per_interval_pos[interval] += 1
                if Y_pred[i] == Y_test[i]:
                    hits_per_interval_pos[interval] += 1
            else:
                interval = int(intervals/2)-1-int(-dw_nominate*intervals/2)
                samples_per_interval_neg[interval] += 1
                if Y_pred[i] == Y_test[i]:
                    hits_per_interval_neg[interval] += 1

    total_samples_per_interval = samples_per_interval_neg + samples_per_interval_pos
    total_hits_per_interval = hits_per_interval_neg + hits_per_interval_pos
    for i in range(0, intervals):
        if total_samples_per_interval[i] != 0:
            accuracies[i] = total_hits_per_interval[i] / \
                total_samples_per_interval[i]
        else:
            accuracies[i] = None

    x = np.arange(-1, 1, (1--1)/len(accuracies))
    plt.plot(x, accuracies, label=legend)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("DW NOMINATE")
    plt.ylabel("Accuracy")


def Evaluate(X_train, Y_train, X_test, Y_test, test_dataset, feature_names):

    # This method train and test a Naive Bayes and a Logistic Regression
    # model and evaluates them with the above functions, for the given feature
    # matrices and labels matries.

    print("NAIVE BAYES: ")
    nb = models.NBtrain(X_train, Y_train)
    Test(X_test, Y_test, nb)
    MostImportantFeatures(feature_names, nb, 20, X_train, NB=True)
    PlotAccuracies(test_dataset, X_test, Y_test, nb, 20, "Naive Bayes")

    print("LOGISTIC REGRESSION: ")
    lr = models.LRtrain(X_train, Y_train)
    Test(X_test, Y_test, lr)
    MostImportantFeatures(feature_names, lr, 20, X_train, NB=False)
    PlotAccuracies(test_dataset, X_test, Y_test, lr, 20, "Logistic Regression")
