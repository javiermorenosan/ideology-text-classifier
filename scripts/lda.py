# coding=utf-8

# This script compiles all the methods used
# in the LDA analysis of the project.

from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import pyLDAvis
import time
import models
import evaluation


def LDAmodel(X_train, number_topics):
    # Creates a LDA model.

    t0 = time.time()
    lda = LDA(n_components=number_topics)
    lda.fit(X_train)
    t1 = time.time()
    t = t1-t0
    print("Time fitting the model : ", t)
    return lda


def print_topics(model, feature_names, n_top_words=10):
    # Prints out the most important words for each topic.

    print("Topics found via LDA:")
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def LDAvisualization(lda, X_train, vectorizer):
    # Creates a HTML document that graphicaly shows
    # the performance of a LDA model.

    pyLDAvis.enable_notebook()
    panel = pyLDAvis.sklearn.prepare(lda, X_train, vectorizer, mds='tsne')
    pyLDAvis.save_html(panel, "./ldavis_prepared.html")


def extractTopicsProbabilitiesFeatures(lda, X_train, X_test):
    # Extracts the probability of each topic for each
    # observatiton and create the feature matrices.

    lda_output_train = lda.transform(X_train)
    df_topics_probabilities_train = pd.DataFrame(np.round(lda_output_train, 2))
    X_train = csr_matrix(df_topics_probabilities_train)

    lda_output_test = lda.transform(X_test)
    df_topics_probabilities_test = pd.DataFrame(np.round(lda_output_test, 2))
    X_test = csr_matrix(df_topics_probabilities_test)

    return df_topics_probabilities_train, df_topics_probabilities_test


def extractDominantTopics(lda, X):
    # Extracts the dominant topic of each observation.

    lda_output = lda.transform(X)
    df_dominant_topics = pd.DataFrame(np.round(lda_output, 2))
    dominant_topic = np.argmax(df_dominant_topics.values, axis=1)
    df_dominant_topics['dominant_topic'] = dominant_topic
    df_dominant_topics = df_dominant_topics.filter(["dominant_topic"])
    return df_dominant_topics


def topicDistribution(df_dominant_topics, Y):
    # Returns a dataframe with the number of topics per topic
    # and the positive/negative distribution.

    df_topic_distribution = df_dominant_topics['dominant_topic'].value_counts(
    ).reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    df_topic_distribution = df_topic_distribution.sort_values(by=['Topic Num'])

    positive_per_topic = {}
    negative_per_topic = {}
    dominant_topic_list = df_dominant_topics['dominant_topic'].values
    for i in range(0, len(dominant_topic_list)):
        if Y[i] == 1.0:
            if int(dominant_topic_list[i]) not in positive_per_topic:
                positive_per_topic[dominant_topic_list[i]] = 1
            else:
                positive_per_topic[dominant_topic_list[i]] += 1
        else:
            if dominant_topic_list[i] not in negative_per_topic:
                negative_per_topic[dominant_topic_list[i]] = 1
            else:
                negative_per_topic[dominant_topic_list[i]] += 1
    ordered_positive_per_topic = collections.OrderedDict(
        sorted(positive_per_topic.items()))
    ordered_negative_per_topic = collections.OrderedDict(
        sorted(negative_per_topic.items()))
    negatives = ordered_negative_per_topic.values()
    positives = ordered_positive_per_topic.values()
    df_topic_distribution['Number of positives'] = positives
    df_topic_distribution['Number of negatives'] = negatives
    df_topic_distribution["Percentage of positives"] = df_topic_distribution['Number of positives']/(
        df_topic_distribution['Number of positives']+df_topic_distribution['Number of negatives'])

    return df_topic_distribution


def extractTextsPerTopic(df_dominant_topics, n_topics):
    # Returns a dictionary with the topics as keys and the
    # indices of observations within that topic.

    topics = list(range(n_topics))
    topic_speeches = {}
    for i in topics:
        speeches = df_dominant_topics[df_dominant_topics["dominant_topic"]
                                      == i].index.values
        topic_speeches[i] = speeches
    return topic_speeches


def createMatricesForTopic(indices, X_total, Y_total):
    # Creates X and Y matrices of a topic given the total X
    # and Y matrices and the indices of the speeches of that
    # topic.

    X_df = pd.DataFrame(X_total.toarray()).loc[indices]
    X_csr = csr_matrix(X_df)
    Y = [Y_total[index] for index in indices]
    return X_csr, Y


def evaluateTopic(X_train_topic, X_test_topic, Y_train_topic, Y_test_topic):
    # Calculates the accuracy for both Naive Bayes
    # and Logistic Regression for one single topic.

    print("NB model:")
    nb = models.NBtrain(X_train_topic, Y_train_topic)
    (accuracy_nb, precision_nb, recall_nb) = evaluation.Test(
        X_test_topic, Y_test_topic, nb)

    print("LR model:")
    lr = models.LRtrain(X_train_topic, Y_train_topic)
    (accuracy_lr, precision_lr, recall_lr) = evaluation.Test(
        X_test_topic, Y_test_topic, lr)

    return accuracy_nb, accuracy_lr


def plotPerplexities(perplexities):
    # Plot perplexities of the LDA models.

    x = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    plt.plot(x, perplexities, color='#2D00B7', label="")
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity")
    plt.show()


def plotScores(scores):
    # Plot scores of the LDA models.

    x = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    plt.plot(x, scores, color='#64EEBA', label="")
    plt.xlabel("Number of topics")
    plt.ylabel("Score")
    plt.show()


def LDAexperiment(X_train, feature_names):
    # This experiment calculates a LDA module for different
    # amount of topics (from 10 to 20) and plots the
    # perplexty and the score of each of them. The lower the
    # perplexity and the higer the score the better the model is.

    n_topics_list = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ldas = []
    perplexities = []
    scores = []
    for i in range(0, len(n_topics_list)):
        n_topics = n_topics_list[i]
        print("Number of topics: ", n_topics)
        lda = LDAmodel(X_train, n_topics)
        ldas.append(lda)
        print_topics(lda, feature_names, n_top_words=10)
        perplexity = lda.perplexity(X_train)
        score = lda.score(X_train)
        print("Perplexity: ", perplexity)
        print("Score: ", score)
        perplexities.append(perplexity)
        scores.append(score)
    plotPerplexities(perplexities)
    plotScores(scores)

    return ldas, perplexities, scores


def plotAccuraciesPerTopic(accuracies_nb, accuracies_lr):
    # Plot the accuracies of each topic for both the
    # Naive Bayes and the Logistic Regression models.
    fig = plt.figure(figsize=(10, 5))
    N = len(accuracies_nb)

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, accuracies_nb, width, label='Naive Bayes', color="#2D00B7")
    plt.bar(ind + width, accuracies_lr, width,
            label='Logistic Regression', color="#64EEBA")

    plt.ylabel('Accuracy')
    plt.xlabel('Topic')
    plt.title('Accuracy per topic')

    plt.xticks(ind + width / 2, list(range(0, N)))
    plt.legend(loc='best')
    fig.tight_layout()

    plt.show()


def evaluateAlltopics(train_topic_speeches, test_topic_speeches, X_train_total, X_test_total, Y_train_total, Y_test_total):
    # Calculates the accuracies of each topic given dictionaries
    # that contain the indices of the observation for each topic.
    # These dictionaries are calulates with the extractDominantTopics
    # function.

    accuracies_nb = []
    accuracies_lr = []
    for i in range(0, len(train_topic_speeches)):
        print("Creating topic matrices from total X_train and X_train for the topic", i)
        print("Train topic", i)
        X_train_topic, Y_train_topic = createMatricesForTopic(
            train_topic_speeches[i], X_train_total, Y_train_total)
        print("Test topic", i)
        X_test_topic, Y_test_topic = createMatricesForTopic(
            test_topic_speeches[i], X_test_total, Y_test_total)

        (accuracy_nb, accuracy_lr) = evaluateTopic(
            X_train_topic, X_test_topic, Y_train_topic, Y_test_topic)
        accuracies_nb.append(accuracy_nb)
        accuracies_lr.append(accuracy_lr)

    return accuracies_nb, accuracies_lr


def accuracyPerTopicExperiment(lda, n_topics, X_train_total, X_test_total, Y_train_total, Y_test_total):
    # Given a LDA model and the number of topics in this LDA, this
    # experiment calculates and plots the accuracies for both the
    # Naive Bayes and Logistic Regression models for each of the
    # topics.

    X_train_df_dominant_topics = extractDominantTopics(lda, X_train_total)
    X_test_df_dominant_topics = extractDominantTopics(lda, X_test_total)

    train_topic_speeches = extractTextsPerTopic(
        X_train_df_dominant_topics, n_topics)
    test_topic_speeches = extractTextsPerTopic(
        X_test_df_dominant_topics, n_topics)

    (accuracies_nb, accuracies_lr) = evaluateAlltopics(train_topic_speeches,
                                                       test_topic_speeches, X_train_total, X_test_total, Y_train_total, Y_test_total)

    plotAccuraciesPerTopic(accuracies_nb, accuracies_lr)
