# coding=utf-8

# This script contains methods for creating the feature matrices
# and reading them from the txt files when stored. Three type of
# features are included in this script: BoW, with three different
# variations of frequency measurement and the possibility of add
# bigrams and trigrams to the features, Collocations of two words
# within a tuneable parameter window of words, and Collocations 
# around most important words, implemented the same way as normal
# Collocations.

import numpy as np
import pandas as pd
import random
import string
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.collocations import BigramCollocationFinder
from scipy.sparse import csr_matrix
import time

def ExtractWordFeatures(train_dataset, test_dataset, vectorizer_type = "CountVectorizer", ngrams = None, balance_dataset = False, remove_center_interval = None):
    
    # This method extracts the BoW features for the given train and test datasets.
    # It returns X, Y matrices the vectorizer and a list with the feature names.
    # There are three tuneable parameters:
    # - vectorizer_type: three types of measuring the frequence of the words are possible:
    #     - CountVectorizer: to measure the features with the amount of times
    #     they appear on each texts.
    #     - HashingVectorizer: to measure the features with just a measure of presence (1)
    #     or absence of the word.
    #     - TfidfVectorizer: TF-IDF method.
    # - ngrams: to select whether to include only words (None) or also bigrams (2) or trigrams (3).
    # - balance_dataset: set to True to balance the training dataset.
    # - remove_center_interval: format: [-0.2, 0.2]. To remove samples with DW-Nominate inside 
    # the interval.
    
    path_train = "../datasets/train/"+train_dataset
    train_dataset_df = datasetTodf(path_train)
    path_test = "../datasets/test/"+test_dataset
    test_dataset_df = datasetTodf(path_test)
    
    return ExtractWordFeaturesWithDataframes(train_dataset_df, test_dataset_df, vectorizer_type = "CountVectorizer", ngrams = None, balance_dataset = False, remove_center_interval = None)

def datasetTodf(file_path):
    # Reads a text or csv file and creates a pandas dataframe.

    dataset_df = pd.read_csv(file_path, sep = "|", encoding = "latin_1", header = None)
    dataset_df.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    return dataset_df
    
def ExtractWordFeaturesWithDataframes(train_dataset_df, test_dataset_df, vectorizer_type = "CountVectorizer", ngrams = None, balance_dataset = False, remove_center_interval = None):
    # Main logic of the method ExtractWordFeatures.

    (train_speeches, Y_train) = extractTextsAndLabelsFromDf(train_dataset_df, balance_dataset = balance_dataset, remove_center_interval = remove_center_interval)
    (test_speeches, Y_test) = extractTextsAndLabelsFromDf(test_dataset_df, balance_dataset = balance_dataset, remove_center_interval = remove_center_interval)
    
    if vectorizer_type == "CountVectorizer":
        if ngrams != None:
            vectorizer = CountVectorizer(stop_words = 'english', token_pattern = r'[a-zA-Z]+', ngram_range = (1, ngrams))
        else:    
            vectorizer = CountVectorizer(stop_words = 'english', token_pattern = r'[a-zA-Z]+')
    
    if vectorizer_type == "HashingVectorizer":
        vectorizer = CountVectorizer(stop_words = 'english', token_pattern = r'[a-zA-Z]+')
        
    if vectorizer_type == "TfidfVectorizer":
        if ngrams != None:
            vectorizer = TfidfVectorizer(stop_words = 'english', token_pattern = r'[a-zA-Z]+', ngram_range = (1, ngrams))
        else:    
            vectorizer = TfidfVectorizer(stop_words = 'english', token_pattern = r'[a-zA-Z]+')
       
    X_train = vectorizer.fit_transform(train_speeches)
    
    X_test = vectorizer.transform(test_speeches)
    
    if vectorizer_type == "HashingVectorizer":
        transformer = Binarizer().fit(X_train)
        X_train = transformer.transform(X_train)
        transformer = Binarizer().fit(X_test)
        X_test = transformer.transform(X_test)
    
    feature_names = vectorizer.get_feature_names()
    
    return X_train, Y_train, X_test, Y_test, vectorizer, feature_names

def extractTextsAndLabelsFromDf(dataset_df, balance_dataset = False, remove_center_interval = None):

    # This method extract labels and speeches from the total dataframes.

    #Remove rows with DW-nominates close to 0
    if type(remove_center_interval)  != type(None):
        dataset_df['ideology'] = dataset_df['nominate_dim1'].apply(lambda x: 0 if (float(x) > remove_center_interval[0] and float(x) < remove_center_interval[1]) else x)
        dataset_df = dataset_df[dataset_df['ideology'] != 0]
    
    dataset_df['ideology'] = dataset_df['nominate_dim1'].apply(lambda x: 1.0 if (float(x) >= 0) else -1.0)
    
    if balance_dataset:
        positive_rows = len(dataset_df[dataset_df['ideology'] == 1.0])
        negative_rows = len(dataset_df[dataset_df['ideology'] == -1.0])
        if positive_rows > negative_rows:
            n = positive_rows-negative_rows
            
            indices = dataset_df[dataset_df['ideology'] == 1.0].index.values.tolist()
            drop_indices = random.sample(indices, n)
            dataset_df = dataset_df.drop(drop_indices)
        else:
            n = negative_rows-positive_rows
            
            indices = dataset_df[dataset_df['ideology'] == -1.0].index.values.tolist()
            drop_indices = random.sample(indices, n)
            dataset_df = dataset_df.drop(drop_indices)
    
    speeches = dataset_df['speech'].values.tolist()
    Y = dataset_df['ideology'].values.tolist()
    
    return speeches, Y

def ExtractCollocationFeatures(train_dataset, test_dataset, X_train_filename, X_test_filename, window_size, n_features, balance_dataset = False, remove_center_interval = None):

    # This method extract Collocations of two words within the given 
    # window of words as features from the given train and test datasets. 
    # It returns X, Y matrices the vectorizer and a list with the feature names. 
    # It also stores those X matrices in txt files with names X_train_filename and 
    # X_test_filename under the /feature_matrices folder.
    # There are five tuneable parameters:
    # - window_size: size of the window
    # - n_features: number of features considered.
    # - balance_dataset: set to True to balance the training dataset.
    # - remove_center_interval: format: [-0.2, 0.2]. To remove samples with DW-Nominate inside 
    # the interval.

    print("Reading datasets...")
    path_train = "../datasets/train/"
    train_dataset_df = pd.read_csv(path_train+train_dataset, sep = "|", encoding = "latin_1", header=None)
    train_dataset_df.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    
    #Remove rows with DW-nominates close to 0
    if type(remove_center_interval)  != type(None):
        train_dataset_df['ideology'] = train_dataset_df['nominate_dim1'].apply(lambda x: 0 if (float(x) > remove_center_interval[0] and float(x) < remove_center_interval[1]) else x)
        train_dataset_df = train_dataset_df[train_dataset_df['ideology'] != 0]

    train_dataset_df['ideology'] = train_dataset_df['nominate_dim1'].apply(lambda x: 1.0 if (float(x) >= 0) else -1.0)
    
    path_test = "../datasets/test/"
    test_dataset_df = pd.read_csv(path_test+test_dataset, sep = "|", encoding = "latin_1", header=None)
    test_dataset_df.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    
    if balance_dataset == True:
        positive_rows = len(train_dataset_df[train_dataset_df['ideology'] == 1.0])
        negative_rows = len(train_dataset_df[train_dataset_df['ideology'] == -1.0])
        if positive_rows > negative_rows:
            n = positive_rows-negative_rows
            
            indices = train_dataset_df[train_dataset_df['ideology'] == 1.0].index.values.tolist()
            drop_indices = random.sample(indices, n)
            train_dataset_df = train_dataset_df.drop(drop_indices)
        else:
            n = negative_rows-positive_rows
            
            indices = train_dataset_df[train_dataset_df['ideology'] == -1.0].index.values.tolist()
            drop_indices = random.sample(indices, n)
            train_dataset_df = train_dataset_df.drop(drop_indices)
    
    train_speeches = train_dataset_df['speech'].values.tolist()
    Y_train = train_dataset_df['ideology'].values.tolist()

    if type(remove_center_interval)  != type(None):
        test_dataset_df['ideology'] = test_dataset_df['nominate_dim1'].apply(lambda x: 0 if (float(x) > remove_center_interval[0] and float(x) < remove_center_interval[1]) else x)
        test_dataset_df = test_dataset_df[test_dataset_df['ideology'] != 0]

    test_dataset_df['ideology'] = test_dataset_df['nominate_dim1'].apply(lambda x: 1.0 if (float(x) >= 0) else -1.0)
    
    test_speeches = test_dataset_df['speech'].values.tolist()
    Y_test = test_dataset_df['ideology'].values.tolist()
    
    print("Extracting features from train dataset...")
    t_start = time.time()
    
    stop_words = stopwords.words('english')
    
    total_bigrams = {}
    bigrams_per_speech_train = []
    t0 = time.time()
    print(len(train_speeches))
    for i in range(0, len(train_speeches)):
        if (i%1000 == 0):
            print(i)
            t1 = time.time()
            print(str(t1-t0) + " segundos")
            t0 = time.time()
            
        speech = train_speeches[i]
        speech = speech.lower()
        speech = speech.translate(str.maketrans('', '', string.punctuation))
        words = speech.split()
        stop_words = set(stopwords.words('english')) 
        filtered_words = [w for w in words if not w in stop_words]
        
        bcf = BigramCollocationFinder.from_words(filtered_words, window_size = window_size)
        
        for item in bcf.ngram_fd.items():
            if item[0] not in total_bigrams:
                total_bigrams.update({item[0]: item[1]})
            else:
                total_bigrams[item[0]] += item[1]
        
        bigrams_per_speech_train.append(bcf.ngram_fd.items())
    
    print("Total bigrams finded: ", len(total_bigrams))
    
    feature_names = []
    most_frequent_bigrams_sorted = sorted(total_bigrams.items(), key=lambda x: x[1], reverse = True)[:n_features]
    print("Number of features: ", len(most_frequent_bigrams_sorted))
    most_frequent_bigrams = dict(most_frequent_bigrams_sorted)
    
    for i in range(0, len(most_frequent_bigrams_sorted)):
        feature_names.append(most_frequent_bigrams_sorted[i][0]) 
    print(len(feature_names))
    
    order = list(range(0, len(feature_names)))
    collocation_order = dict(zip(feature_names, order))
    
    print("Computing X_train...")
    
    X_train_matrix = np.zeros((len(bigrams_per_speech_train), len(feature_names)))
    
    for i in range(0, len(bigrams_per_speech_train)):
        if (i%1000 == 0):
            print(i)
            t1 = time.time()
            print(str(t1-t0) + " segundos")
            t0 = time.time()
        bigrams_per_speech_i = dict(bigrams_per_speech_train[i])
        for bigram in bigrams_per_speech_i:
            if bigram in most_frequent_bigrams:
                column = collocation_order[bigram]
                X_train_matrix[i][column] = bigrams_per_speech_i[bigram]  
    print("Creating dataframe...")
    X_train_df = pd.DataFrame(X_train_matrix, columns=feature_names)
    t1 = time.time()
    print(str(t1-t0) + " segundos")
    t0 = time.time()
    
    pathX = "../feature_matrices/"
    print("Saving X_train into a txt file...")
    X_train_df.to_csv(pathX+X_train_filename, header=feature_names, index=None, sep=',')
    print("Transforming X_train into a csr_matrix...")
    X_train = csr_matrix(X_train_df)
    
    print("Extracting bigrams from test dataset...")

    bigrams_per_speech_test = []
    t0 = time.time()
    print(len(test_speeches))
    for i in range(0, len(test_speeches)):
        if (i%1000 == 0):
            print(i)
            t1 = time.time()
            print(str(t1-t0) + " segundos")
            t0 = time.time()
            
        speech = test_speeches[i]
        speech = speech.lower()
        speech = speech.translate(str.maketrans('', '', string.punctuation))
        words = speech.split()
        stop_words = set(stopwords.words('english')) 
        filtered_words = [w for w in words if not w in stop_words]
        
        bcf = BigramCollocationFinder.from_words(filtered_words, window_size = window_size)
        bigrams_per_speech_test.append(bcf.ngram_fd.items())
    
    print("Computing X_test...")
    
    X_test_matrix = np.zeros((len(bigrams_per_speech_test), len(feature_names)))
    for i in range(0, len(bigrams_per_speech_test)):
        if (i%1000 == 0):
            print(i)
            t1 = time.time()
            print(str(t1-t0) + " segundos")
            t0 = time.time()
        bigrams_per_speech_i = dict(bigrams_per_speech_test[i])
        for bigram in bigrams_per_speech_i:
            if bigram in most_frequent_bigrams:
                column = collocation_order[bigram]
                X_test_matrix[i][column] = bigrams_per_speech_i[bigram]
                
    print("Creating dataframe...")
    X_test_df = pd.DataFrame(X_test_matrix, columns=feature_names)
    t1 = time.time()
    print(str(t1-t0) + " segundos")
    t0 = time.time()
    
    print("Saving X_test into a txt file...")
    X_test_df.to_csv(pathX+X_test_filename, header=feature_names, index=None, sep=',')
    print("Transforming X_train into a csr_matrix...")
    X_test = csr_matrix(X_test_df)

    t_end = time.time()
    total_time = t_end-t_start
    print("Total time: ")
    print(str(total_time) + " segundos")
    
    return X_train, Y_train, X_test, Y_test, feature_names

def ExtractCollocationsMFWords(train_dataset, test_dataset, X_train_filename, X_test_filename, window_size, n_features, positive_words, negative_words, remove_center = False):
    
    # This method extract Collocations of two words within the given 
    # window around the most important words provided in the lists 
    # positive_words and negative_words, as features from the given 
    # train and test datasets. 
    # It returns X, Y matrices the vectorizer and a list with the feature names. 
    # It also stores those X matrices in txt files with names X_train_filename and 
    # X_test_filename.
    # There are two tuneable parameters:
    # - window_size: size of the window
    # - n_features: number of features considered.

    t_start = time.time()
    positive_words = positive_words[:5]
    negative_words = negative_words[:5]
    print(positive_words)
    print(negative_words)
    
    print("Reading datasets...")
    path_train = "../datasets/train/"
    train_dataset_df = pd.read_csv(path_train+train_dataset, sep = "|", encoding = "latin_1", header=None)
    print(len(train_dataset_df))
    train_dataset_df.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    print(len(train_dataset_df))
    
    #Remove rows with DW-nominates close to 0
    if remove_center == True:
        train_dataset_df['ideology'] = train_dataset_df['nominate_dim1'].apply(lambda x: None if (float(x) > -0.2 and float(x) < 0.2) else x)
    
    train_dataset_df['ideology'] = train_dataset_df['nominate_dim1'].apply(lambda x: 1.0 if (float(x) >= 0.0) else -1.0)
    
    path_test = "../datasets/test/"
    test_dataset_df = pd.read_csv(path_test+test_dataset, sep = "|", encoding = "latin_1", header=None)
    test_dataset_df.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
    test_dataset_df['ideology'] = test_dataset_df['nominate_dim1'].apply(lambda x: 1.0 if (float(x) >= 0) else -1.0)
    
    train_speeches = train_dataset_df['speech'].values.tolist()
    Y_train = train_dataset_df['ideology'].values.tolist()
    
    test_speeches = test_dataset_df['speech'].values.tolist()
    Y_test = test_dataset_df['ideology'].values.tolist()
    
    print("Extracting features from train dataset...")
    
    stop_words = stopwords.words('english')
    total_collocations = {}
    collocations_per_speech_train = []
    t0 = time.time()
    print("Number of speeches: ", len(train_speeches))
    
    for i in range(0, len(train_speeches)):
        if (i%1000 == 0):
            print(i)
            t1 = time.time()
            print(str(t1-t0) + " segundos")
            t0 = time.time()
            
        speech = train_speeches[i]
        speech = speech.lower()
        speech = speech.translate(str.maketrans('', '', string.punctuation))
        words = speech.split()
        stop_words = set(stopwords.words('english')) 
        filtered_words = [w for w in words if not w in stop_words]
        collocations_speech_i = {}
        
        for j in range(0, len(filtered_words)):
            w = filtered_words[j]
            if (w in positive_words) or (w in negative_words):
                for k in range(-4, 5):
                    if k+j >= 0 and k+j < len(filtered_words) and k != 0:
                        collocation_name = (w, filtered_words[k+j])
                        if collocation_name not in total_collocations:
                            total_collocations[collocation_name] = 1
                        else:
                            total_collocations[collocation_name] += 1
                            
                        if collocation_name not in collocations_speech_i:
                            collocations_speech_i[collocation_name] = 1
                        else:
                            collocations_speech_i[collocation_name] += 1
        collocations_per_speech_train.append(collocations_speech_i)
    
    print("Total collocations: ", len(total_collocations))
    
    total_collocations_sorted = sorted(total_collocations.items(), key=lambda x: x[1], reverse = True)[:n_features]
    feature_names = []
    for i in range(0, len(total_collocations_sorted)):
        feature_names.append(total_collocations_sorted[i][0]) 
    print(len(feature_names))
    
    order = list(range(0, len(feature_names)))
    collocation_order = dict(zip(feature_names, order))
    
    print("Computing X_train...")
    X_train_matrix = np.zeros((len(collocations_per_speech_train), len(feature_names)))
    
    for i in range(0, len(collocations_per_speech_train)):
        if (i%1000 == 0):
            print(i)
            t1 = time.time()
            print(str(t1-t0) + " segundos")
            t0 = time.time()
        collocations_speech_i = collocations_per_speech_train[i]
        for collocation in collocations_speech_i:
            if collocation in collocation_order:
                column = collocation_order[collocation]
                X_train_matrix[i][column] = collocations_speech_i[collocation]
    
    print("Creating dataframe...")
    X_train_df = pd.DataFrame(X_train_matrix, columns=feature_names)
    t1 = time.time()
    print(str(t1-t0) + " segundos")
    t0 = time.time()

    pathX = "../feature_matrices/"
    print("Saving X_train into a txt...")
    X_train_df.to_csv(pathX+X_train_filename, header=feature_names, index=None, sep=',')
    t1 = time.time()
    print(str(t1-t0) + " segundos")
    t0 = time.time()
    print("Transforming X_train into a csr_matrix...")
    X_train = csr_matrix(X_train_matrix)
    t1 = time.time()
    print(str(t1-t0) + " segundos")
    t0 = time.time()
    
    print("Extracting features from test dataset...")
    collocations_per_speech_test = []
    t0 = time.time()
    print(len(test_speeches))
    for i in range(0, len(test_speeches)):
        if (i%1000 == 0):
            print(i)
            t1 = time.time()
            print(str(t1-t0) + " segundos")
            t0 = time.time()
            
        speech = test_speeches[i]
        speech = speech.lower()
        speech = speech.translate(str.maketrans('', '', string.punctuation))
        words = speech.split()
        stop_words = set(stopwords.words('english')) 
        filtered_words = [w for w in words if not w in stop_words]
        collocations_speech_i = {}
        
        for j in range(0, len(filtered_words)):
            w = filtered_words[j]
            if w in positive_words or w in negative_words:
                for k in range(-4, 5):
                    if k+j >= 0 and k+j < len(filtered_words) and k != 0:
                        collocation_name = (w, filtered_words[k+j])
                        if collocation_name in total_collocations:                         
                            if collocation_name not in collocations_speech_i:
                                collocations_speech_i[collocation_name] = 1
                            else:
                                collocations_speech_i[collocation_name] += 1
        collocations_per_speech_test.append(collocations_speech_i)
    
    print("Computing X_test...")
    X_test_matrix = np.zeros((len(collocations_per_speech_test), len(feature_names)))
    
    for i in range(0, len(collocations_per_speech_test)):
        if (i%1000 == 0):
            print(i)
            t1 = time.time()
            print(str(t1-t0) + " segundos")
            t0 = time.time()
        collocations_speech_i = collocations_per_speech_test[i]
        for collocation in collocations_speech_i:
            if collocation in collocation_order:
                column = collocation_order[collocation]
                X_test_matrix[i][column] = collocations_speech_i[collocation]
    
    print("Creating dataframe...")
    X_test_df = pd.DataFrame(X_test_matrix, columns=feature_names)  
    t1 = time.time()
    print(str(t1-t0) + " segundos")
    t0 = time.time()
    
    pathX = "../feature_matrices/"
    
    print("Saving X_test into a txt...")
    X_test_df.to_csv(pathX+X_test_filename, header=feature_names, index=None, sep=',')
    t1 = time.time()
    print(str(t1-t0) + " segundos")
    t0 = time.time()
    print("Transforming X_test into a csr_matrix...")
    X_test = csr_matrix(X_test_df)
    t1 = time.time()
    print(str(t1-t0) + " segundos")
    t0 = time.time()
    
    t_end = time.time()
    total_time = t_end-t_start
    print("Total time: ")
    print(str(total_time) + " segundos")
    
    print(X_train_df.loc[X_train_df[feature_names[0]] != 0])
    
    return X_train, Y_train, X_test, Y_test, feature_names, X_train_df, X_test_df

def ReadX(txt_file):

    # This method reads a X matrix file into a csr matrix to be
    # used to training/testing purposes.

    pathX = "../feature_matrices/"
    X_df = pd.read_csv(pathX+txt_file, sep = ",", header = 0)
    feature_names = X_df.columns.values.tolist()
    X = csr_matrix(X_df)
    #X = X_df.values.tolist()
    
    return X, feature_names

        