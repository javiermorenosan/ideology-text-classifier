# coding=utf-8

# This script contains functions to analyze in deep
# the data from the given datasets in terms of Information Gain
# and lenght of the speeches. Understanding for Information 
# Gain of a speech the average Information Gain of the words
# of that particular speech.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import string
import time
import numpy as np

def getSpeeches(dataset_path):

    # This method extracts a list of the speeches
    # from a given dataset.

    dataset = pd.read_csv(dataset_path, sep = "|", encoding = "latin_1", header = None)
    dataset.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    speeches = dataset['speech'].values.tolist()
    
    return speeches

def getNominates(dataset_path):

    # This method extract a list of the DW Nominates
    # from a given dataset.

    dataset = pd.read_csv(dataset_path, sep = "|", encoding = "latin_1", header = None)
    dataset.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    nominates = dataset['nominate_dim1'].values.tolist()
    
    return nominates

def getSpeechesAndNominates(dataset_path):

    # This method combines the two above mentioned methods.

    speeches = getSpeeches(dataset_path)
    nominates = getNominates(dataset_path)
    
    return speeches, nominates

def extractLenghtData(X_test, Y_test, trained_model):

    # This method returns the following six lists of length 
    # of the speeches:
    # - lengths_train: train speeches.
    # - lengths_test: test speeches.
    # - lengths_pos: positive class speeches.
    # - lengths_neg: negative class speeches.
    # - lengths_correct: correctly predicted speeches.
    # - lengths_wrong: wrongly predicted speeches.

    (train_speeches, train_nominates) = getSpeechesAndNominates("../datasets/train/speeches_110_dwnominate_nonames.txt")
    (test_speeches, test_nominates) = getSpeechesAndNominates("../datasets/test/speeches_112_dwnominate_nonames.txt")

    Y_pred = trained_model.predict(X_test)
    
    lengths_train = []
    lengths_pos = []
    lengths_neg = []
    for i in range(0, len(train_speeches)):
        speech = train_speeches[i].split()
        lengths_train.append(len(speech))
        if float(train_nominates[i]) < 0:
            lengths_neg.append(len(speech))
        else:
            lengths_pos.append(len(speech))
    
    lengths_test = []
    lengths_correct = []
    lengths_wrong = []
    for i in range(0, len(test_speeches)):
        speech = test_speeches[i].split()
        lengths_test.append(len(speech))
        if float(test_nominates[i]) < 0:
            lengths_neg.append(len(speech))
        else:
            lengths_pos.append(len(speech))
        if Y_pred[i] == Y_test[i]:
            lengths_correct.append(len(speech))
        else:
            lengths_wrong.append(len(speech))
    
    return lengths_train, lengths_test, lengths_pos, lengths_neg, lengths_correct, lengths_wrong

def plotLenghtSpeeches(lengths_train, lengths_test, lengths_pos, lengths_neg, lengths_correct, lengths_wrong):

    # This method plots several graphs with the length lists
    # extracted with the extractLenghtData method.

    fig = plt.figure(figsize=(15, 15))
    
    plt.subplot(4, 2, 1)
    plt.hist(lengths_train, bins = int((max(lengths_train)+min(lengths_train))/50))
    plt.title("Training dataset")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    
    plt.subplot(4, 2, 2)
    plt.hist(lengths_test, bins = int((max(lengths_test)+min(lengths_test))/50), color = '#2D00B7')
    plt.title("Test dataset")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    
    plt.subplot(4, 2, 3)
    plt.hist(lengths_pos, bins = int((max(lengths_pos)+min(lengths_pos))/50), color = '#59B2BD')
    plt.title("Total positive")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    
    plt.subplot(4, 2, 4)
    plt.hist(lengths_neg, bins = int((max(lengths_neg)+min(lengths_neg))/50), color = '#0000FF')
    plt.title("Total negative")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    
    plt.subplot(4, 2, 5)
    plt.hist(lengths_correct, bins = int((max(lengths_correct)+min(lengths_correct))/50), color = "green")
    plt.title("Correctly predicted")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    
    plt.subplot(4, 2, 6)
    plt.hist(lengths_wrong, bins = int((max(lengths_wrong)+min(lengths_wrong))/50), color = "red")
    plt.title("Wrongly predicted")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    
    plt.subplot(4, 2, 7)
    plt.hist(lengths_neg, bins = int((max(lengths_neg)+min(lengths_neg))/50), label = "Total negative", color = '#0000FF')
    plt.title("Positive vs Negative")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    
    plt.hist(lengths_pos, bins = int((max(lengths_pos)+min(lengths_pos))/50), label = "Total positive", color = '#59B2BD')
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    plt.legend()
    
    plt.subplot(4, 2, 8)
    plt.hist(lengths_correct, bins = int((max(lengths_correct)+min(lengths_correct))/50), label = "Correctly predicted", color = "green")
    plt.title("Correct vs Incorrect")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    
    plt.hist(lengths_wrong, bins = int((max(lengths_wrong)+min(lengths_wrong))/50), label = "TWrongly predicted", color = "red")
    plt.xlim(0, 3000)
    plt.xlabel("Speech length")
    plt.ylabel("Number of speeches")
    plt.legend
    
    fig.tight_layout()
    
    plt.show()

def extractIGs(X_train, Y_train):

    # Returns a list with the Information Gain of each 
    # word feature in the training dataset.

    print("Extracting mutual information...")
    t1 = time.time()
    IGs = mutual_info_classif(X_train, Y_train, discrete_features=True)
    t2 = time.time()
    print(t2-t1, " seconds")
    
    return IGs

def getIGAvg(speech, word_IG_dict):

    # It calculates the average IG of the words in 
    # given speech.

    speech = speech.lower()
    speech = speech.translate(str.maketrans('', '', string.punctuation))
    words = speech.split()
    n_words = 0
    sum_igs = 0
    for word in words:
        if word in word_IG_dict:
            ig_i = word_IG_dict[word]
            sum_igs += ig_i
            n_words += 1
    ig_avg = sum_igs/n_words
        
    return ig_avg

def getIGAvgs(speeches, IGs, feature_names):

    # Using the getIGAvg method it calculates the average IG 
    # of the words in all the speeches of a given list of speeches.

    word_IG_dict = dict(zip(feature_names, IGs))
    ig_avgs = []
    print(len(speeches))
    for i in range(0, len(speeches)):
        if i%1000 == 0:
            print(i)
        speech = speeches[i]
        ig_avg = getIGAvg(speech, word_IG_dict)
        ig_avgs.append(ig_avg)  
    return ig_avgs

def plotInformationGain(X_test, Y_test, trained_model, feature_names, IGs):

    # This method plots several graphs with the average 
    # IG of the speeches list.

    fig = plt.figure(figsize=(10, 10))
    
    (train_speeches, train_nominates) = getSpeechesAndNominates("../datasets/train/speeches_110_dwnominate_nonames.txt")
    (test_speeches, test_nominates) = getSpeechesAndNominates("../datasets/test/speeches_112_dwnominate_nonames.txt")
    
    Y_pred = trained_model.predict(X_test)
    
    ig_avgs_train = getIGAvgs(train_speeches, IGs, feature_names)
    
    samples_per_interval_pos = [0] * 10
    samples_per_interval_neg = [0] * 10
    sum_igs_per_interval = [0] * 20
    
    for i in range(0, len(train_speeches)):
        nominate_i = train_nominates[i]
        if nominate_i > 0:
            interval = int(nominate_i*10)
            samples_per_interval_pos[interval] += 1
            sum_igs_per_interval[interval+10] += ig_avgs_train[i]
        else:
            interval = 10-1-int(-nominate_i*10)
            samples_per_interval_neg[interval] += 1
            sum_igs_per_interval[interval] += ig_avgs_train[i]
    total_samples_per_interval = samples_per_interval_neg+samples_per_interval_pos
    avg_per_interval = []
    for i in range(len(sum_igs_per_interval)):
        if total_samples_per_interval[i] != 0:
           avg_per_interval.append(sum_igs_per_interval[i]/total_samples_per_interval[i])
        else:
            avg_per_interval.append(0)
            
    x = np.arange(-1, 1, (1--1)/20)
    
    plt.subplot(2, 1, 1)
    plt.plot(x, avg_per_interval, label = "Train dataset", color = '#0000FF')
    plt.title("Average Information Gain per speech")
    plt.xlabel("DW NOMINATE")
    plt.ylabel("Information Gain")
    plt.xlim(-0.7, 1)
    
    ig_avgs_test = getIGAvgs(test_speeches, IGs, feature_names)
    
    samples_per_interval_pos = [0] * 10
    samples_per_interval_neg = [0] * 10
    sum_igs_per_interval = [0] * 20
    
    samples_per_interval_pos_correct = [0] * 10
    samples_per_interval_neg_correct = [0] * 10
    sum_igs_per_interval_correct = [0] * 20
    
    samples_per_interval_pos_wrong = [0] * 10
    samples_per_interval_neg_wrong = [0] * 10
    sum_igs_per_interval_wrong = [0] * 20
    
    for i in range(0, len(test_speeches)):
        nominate_i = test_nominates[i]
        if nominate_i > 0:
            interval = int(nominate_i*10)
            samples_per_interval_pos[interval] += 1
            sum_igs_per_interval[interval+10] += ig_avgs_test[i]
            if Y_pred[i] == Y_test[i]:
                samples_per_interval_pos_correct[interval] += 1
                sum_igs_per_interval_correct[interval+10] += ig_avgs_test[i]
            else:
                samples_per_interval_pos_wrong[interval] += 1
                sum_igs_per_interval_wrong[interval+10] += ig_avgs_test[i]
        else:
            interval = 10-1-int(-nominate_i*10)
            samples_per_interval_neg[interval] += 1
            sum_igs_per_interval[interval] += ig_avgs_test[i]
            if Y_pred[i] == Y_test[i]:
                samples_per_interval_neg_correct[interval] += 1
                sum_igs_per_interval_correct[interval] += ig_avgs_test[i]
            else:
                samples_per_interval_neg_wrong[interval] += 1
                sum_igs_per_interval_wrong[interval] += ig_avgs_test[i]
                
    total_samples_per_interval = samples_per_interval_neg+samples_per_interval_pos
    avg_per_interval = []
    
    for i in range(len(sum_igs_per_interval)):
        if total_samples_per_interval[i] != 0:
           avg_per_interval.append(sum_igs_per_interval[i]/total_samples_per_interval[i])
        else:
            avg_per_interval.append(0)
            
    plt.subplot(2, 1, 1)
    plt.plot(x, avg_per_interval, color = '#2D00B7', label = "Test dataset")
    plt.legend()
    plt.xlim(-0.7, 1)
    
    total_samples_per_interval_correct = samples_per_interval_neg_correct+samples_per_interval_pos_correct
    avg_per_interval_correct = []
    
    for i in range(len(sum_igs_per_interval_correct)):
        if total_samples_per_interval_correct[i] != 0:
           avg_per_interval_correct.append(sum_igs_per_interval_correct[i]/total_samples_per_interval_correct[i])
        else:
            avg_per_interval_correct.append(0)
            
    plt.subplot(2, 1, 2)
    plt.plot(x, avg_per_interval_correct, color = 'green', label = "Correctly predicted")
    plt.legend()
    plt.title("Average Information Gain per speech: Correct/Wrong")
    plt.xlabel("DW NOMINATE")
    plt.ylabel("Information Gain")
    plt.xlim(-0.7, 1)
    
    total_samples_per_interval_wrong = samples_per_interval_neg_wrong+samples_per_interval_pos_wrong
    avg_per_interval_wrong = []
    
    for i in range(len(sum_igs_per_interval_wrong)):
        if total_samples_per_interval_wrong[i] != 0:
           avg_per_interval_wrong.append(sum_igs_per_interval_wrong[i]/total_samples_per_interval_wrong[i])
        else:
            avg_per_interval_wrong.append(0)
            
    plt.subplot(2, 1, 2)
    plt.plot(x, avg_per_interval_wrong, color = 'red', label = "Wrongly predicted")
    plt.legend()
    plt.xlim(-0.7, 1)
    
    fig.tight_layout()
    plt.show()

def plotIGHistogram(IGs):

    # It plots an histogram of the IGs of the speeches
    # to see the concentration of speeches per interval of
    # IG.

    fig = plt.figure(figsize=(10, 5))
    plt.hist(IGs, bins = 4000)
    plt.title("Information Gain Histogram")
    plt.xlim(0, 0.000125)
    plt.xlabel("Information Gain ")
    plt.ylabel("Number of words")
    
    fig.tight_layout()
    plt.show()

def plotIGLength(total_ig_avgs, lengths_train, lengths_test, lengths_pos, lengths_neg, lengths_correct, lengths_wrong, feature_names):
    
    # This method plots several graphs realting the length
    # of the speeches with the information gain of the speeches.
    
    fig = plt.figure(figsize=(10, 10))
    
    total_lengths = lengths_train+lengths_test
    
    train_nominates = getNominates("../datasets/train/speeches_110_dwnominate_nonames.txt")
    test_nominates = getNominates("../datasets/test/speeches_112_dwnominate_nonames.txt")
    
    total_nominates = train_nominates+test_nominates
    
    samples_per_interval_total = [0] * 150
    sum_igs_per_interval_total = [0] * 150
    
    samples_per_interval_train = [0] * 150
    sum_igs_per_interval_train = [0] * 150
    
    samples_per_interval_test = [0] * 150
    sum_igs_per_interval_test = [0] * 150
    
    samples_per_interval_pos = [0] * 150
    sum_igs_per_interval_pos = [0] * 150
    
    samples_per_interval_neg = [0] * 150
    sum_igs_per_interval_neg = [0] * 150
    
    for i in range(0, len(total_lengths)):
        length_i = total_lengths[i]
        interval = int(length_i*150/15000)
        samples_per_interval_total[interval] += 1
        sum_igs_per_interval_total[interval] += total_ig_avgs[i]
        if i < len(lengths_train):
            samples_per_interval_train[interval] += 1
            sum_igs_per_interval_train[interval] += total_ig_avgs[i]
        else:
            samples_per_interval_test[interval] += 1
            sum_igs_per_interval_test[interval] += total_ig_avgs[i]
        if total_nominates[i] < 0:
            samples_per_interval_neg[interval] += 1
            sum_igs_per_interval_neg[interval] += total_ig_avgs[i]
        else:   
            samples_per_interval_pos[interval] += 1
            sum_igs_per_interval_pos[interval] += total_ig_avgs[i]
    
    avg_ig_per_interval_total = []
    for i in range(len(sum_igs_per_interval_total)):
        if samples_per_interval_total[i] != 0:
            avg_ig_per_interval_total.append(sum_igs_per_interval_total[i]/samples_per_interval_total[i])
        else:
            avg_ig_per_interval_total.append(0)
    
    avg_ig_per_interval_train = []
    for i in range(len(sum_igs_per_interval_train)):
        if samples_per_interval_train[i] != 0:
            avg_ig_per_interval_train.append(sum_igs_per_interval_train[i]/samples_per_interval_train[i])
        else:
            avg_ig_per_interval_train.append(0)
            
    avg_ig_per_interval_test = []
    for i in range(len(sum_igs_per_interval_test)):
        if samples_per_interval_test[i] != 0:
            avg_ig_per_interval_test.append(sum_igs_per_interval_test[i]/samples_per_interval_test[i])
        else:
            avg_ig_per_interval_test.append(0)
            
    avg_ig_per_interval_pos = []
    for i in range(len(sum_igs_per_interval_pos)):
        if samples_per_interval_pos[i] != 0:
            avg_ig_per_interval_pos.append(sum_igs_per_interval_pos[i]/samples_per_interval_pos[i])
        else:
            avg_ig_per_interval_pos.append(0)
            
    avg_ig_per_interval_neg = []
    for i in range(len(sum_igs_per_interval_neg)):
        if samples_per_interval_neg[i] != 0:
            avg_ig_per_interval_neg.append(sum_igs_per_interval_neg[i]/samples_per_interval_neg[i])
        else:
            avg_ig_per_interval_neg.append(0)
    
    x = np.arange(0, 15000, 15000/150)
    plt.subplot(2, 1, 1)
    plt.plot(x, avg_ig_per_interval_total, color = '#979EC5', label = "Total")
    plt.xlabel("Length of the speech")
    plt.ylabel("Average Information Gain")
    plt.title("Information Gain with length of the speech (Train/Test/Total)")
    
    plt.subplot(2, 1, 1)
    plt.plot(x, avg_ig_per_interval_train, color = '#2D00B7', label = "Train dataset")
    
    plt.subplot(2, 1, 1)
    plt.plot(x, avg_ig_per_interval_test, color = '#64EEBA', label = "Test dataset")
    plt.legend()
    plt.ylim(0.0008, 0.0011)
    plt.xlim(0, 3000)
    
    plt.subplot(2, 1, 2)
    x = np.arange(0, 15000, 15000/150)
    plt.plot(x, avg_ig_per_interval_total, color = '#2D00B7', label = "Positive")
    plt.xlabel("Length of the speech")
    plt.ylabel("Average Information Gain")
    plt.title("Information Gain with length of the speech (Positive/Negative)")
    
    plt.subplot(2, 1, 2)
    plt.plot(x, avg_ig_per_interval_neg, color = '#64EEBA', label = "Negative")
    plt.legend()
    plt.ylim(0.0008, 0.0011)
    plt.xlim(0, 3000)
    
    fig.tight_layout()
    plt.show()
    
    return


