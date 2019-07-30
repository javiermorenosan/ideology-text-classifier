# coding=utf-8

# In this script we include methods for creating
# the datasets, setting and storing some variables of
# the environment.

import features
import pandas as pd
import re
import unidecode
import string
from scipy.sparse import hstack

def CreateDataset(speakerMap, speeches, HS_members, filename, isTrain, includeNames = False, long_speeches = True):

    # This method creates a dataset file with filename name,
    # given the speeches file, the speakerMap file and 
    # the HS_member file of one congress.
    # There are three tuneable variables:
    # - isTrain: True if the dataset is for training purposes,
    #   to be stored in the train folder and False if the 
    #   dataset is for testing.
    # - includeNames: to remove the names from the corpus.
    # - long_speeches: to remove short speeches from the corpus.

    path = "../source_dataset/"
    speakerMap_df = pd.read_csv(path+speakerMap, sep = "|", error_bad_lines = False)
    speeches_df = pd.read_csv(path+speeches, sep = "|", error_bad_lines = False, encoding = "latin_1")
    HS_members_df = pd.read_csv(path+HS_members, sep = ",", error_bad_lines = False)
    
    speechs_names_df = pd.DataFrame(speakerMap_df['lastname'])
    speechs_names_df['firstname'] =  speakerMap_df['firstname']
    speechs_names_df['speech_id'] =  speakerMap_df['speech_id']
    speechs_names_df = speakerMap_df.merge(speeches_df, on = 'speech_id')
    last_first_speech = speechs_names_df[['lastname', 'firstname', 'speech']].copy()
    
    firstnames = last_first_speech['firstname'].values.tolist()
    lastnames = last_first_speech['lastname'].values.tolist()
    speeches = last_first_speech['speech'].values.tolist()
    bionames = HS_members_df['bioname'].values.tolist()
    nominate_dim1s = HS_members_df['nominate_dim1'].values.tolist()
    nominate_dim2s = HS_members_df['nominate_dim2'].values.tolist()
    
    
    bionames_speech_ordered = []
    nominate_dim1s_speech_ordered = []
    nominate_dim2s_speech_ordered = []
    n_unknowns = 0
    
    print("Joining speeches to names: ")
    
    for i in range(0, len(lastnames)):
        if i%10000 == 0:
            print(i)
        lastname = lastnames[i]
        lastname = unidecode.unidecode(lastname)
        if lastname[0] == 'M' and lastname[1] == 'C':
            list1 = list(lastname)
            list1[1] = 'c'
            lastname = ''.join(list1)
        firstname = firstnames[i].lower().title()
        firstname = unidecode.unidecode(firstname).lower().title()
        for j in range(0, len(bionames)):
            bioname = bionames[j]
            bioname = unidecode.unidecode(bioname)
            if lastname in bioname:
                if firstname in bioname:
                    bionames_speech_ordered.append(bioname)
                    nominate_dim1s_speech_ordered.append(nominate_dim1s[j])
                    nominate_dim2s_speech_ordered.append(nominate_dim2s[j])
                    break
                elif (re.compile(lastname + ', +[ a-zA-z]*'+ firstname[0]).match(bioname) != None):
                    bionames_speech_ordered.append(bioname)
                    nominate_dim1s_speech_ordered.append(nominate_dim1s[j])
                    nominate_dim2s_speech_ordered.append(nominate_dim2s[j])
                    break
                else:
                    next
            if j == len(bionames)-1:
                n_unknowns += 1
                bionames_speech_ordered.append("UNKNOWN")
                nominate_dim1s_speech_ordered.append("UNKNOWN")
                nominate_dim2s_speech_ordered.append("UNKNOWN")
    
    if (includeNames == False):
        #Code to avoid including proper names in the texts. 
        #We first find the words that appear at least once in lowercase in our dataset.
        #Then we extract all words that appear only with first letter in uppercase in out dataset.
        print("Finding words that appears in lowercase at leat once: ")
        
        vocabulary = set()
        
        for i in range(0, len(speeches)):
            speech = speeches[i]
            speech = speech.translate(str.maketrans('', '', string.punctuation))
            if i%1000 == 0:
                print("speech number " + str(i))
            for word in speech.split():
                if word.istitle() or word[0].isupper() or (word in vocabulary):
                    pass
                else:
                    vocabulary.add(word)
                   
        print("Deleting names from the speeches: ")
        
        names = set()
        indices_long = []
        
        for i in range(0, len(speeches)):
            speech = speeches[i]
            speech = speech.translate(str.maketrans('', '', string.punctuation))
            speech_list = []
            if i%1000 == 0:
                print("speech number " + str(i))
            for word in speech.split():
                if (word.istitle() or word[0].isupper()) and (word.lower() not in vocabulary):
                    if word not in names:
                        names.add(word)
                else:
                    speech_list.append(word)
            
            speeches[i] = " ".join(speech_list)
            
            if len(speeches[i])>1000:
                indices_long.append(i)
        
        #print(names)
    
    print(str(n_unknowns) + " unknowns")
    
    bionames_nominate_dim1_nominate_dim2_speech_df = pd.DataFrame(
    {'bioname': bionames_speech_ordered,
     'nominate_dim1': nominate_dim1s_speech_ordered,
     'nominate_dim2': nominate_dim2s_speech_ordered,
     'speech': speeches
    })
    
    if long_speeches:
        bionames_nominate_dim1_nominate_dim2_speech_df = bionames_nominate_dim1_nominate_dim2_speech_df.take(indices_long)
    bionames_nominate_dim1_nominate_dim2_speech_df = bionames_nominate_dim1_nominate_dim2_speech_df[bionames_nominate_dim1_nominate_dim2_speech_df.bioname != 'UNKNOWN']
    bionames_nominate_dim1_nominate_dim2_speech_df = bionames_nominate_dim1_nominate_dim2_speech_df.dropna()
    nominate_dim1_nominate_dim2_speech_df = bionames_nominate_dim1_nominate_dim2_speech_df.drop(columns=['bioname'])
    if isTrain:
        path2 = "../datasets/train/"
    else:
        path2 = "../datasets/test/"
    nominate_dim1_nominate_dim2_speech_df.to_csv(path2+filename, sep='|', index=False)


def ReadEnvironment(remove_center = False):

    # This method was used to read and create feature matrices from a previously
    # created environment in a first stage of the project. This method might be outdated.

    print("Extracting word features without n-grams...", end = "")
    (X_train_w, Y_train, X_test_w, Y_test, vectorizer, feature_names_w) = features.ExtractWordFeatures("speeches_110_dwnominate_nonames.txt", "speeches_112_dwnominate_nonames.txt", remove_center_interval = remove_center)
    print("[DONE]")
    
    print("Extracting word features with 2-grams...", end = "")
    (X_train_w2, Y_train, X_test_w2, Y_test, vectorizer2, feature_names_w2) = features.ExtractWordFeatures("speeches_110_dwnominate_nonames.txt", "speeches_112_dwnominate_nonames.txt", ngrams = 2, remove_center_interval = remove_center)
    print("[DONE]")
    
    print("Reading collocation features...", end = "")
    (X_train_c, feature_names_c) = features.ReadX("X_train_1000_l_2.txt")
    (X_test_c, feature_names_c) = features.ReadX("X_test_1000_l_2.txt")
    print("[DONE]")
    
    if remove_center:
        print("Removing center...", end = "")
        path_train = "../datasets/train/"
        train_dataset_df = pd.read_csv(path_train+"speeches_110_dwnominate_nonames.txt", sep = "|", encoding = "latin_1", header = None)
        train_dataset_df.columns = ['nominate_dim1', 'nominate_dim2', 'speech']
        dw_nominates = train_dataset_df['nominate_dim1'].values.tolist()
        indices = []
        for i in range(0, len(dw_nominates)):
            if float(dw_nominates[i])<-0.2 or float(dw_nominates[i])>0.2:
                indices.append(i)
        X_train_c = [X_train_c[index] for index in indices]
        print("[DONE]")
    
    print("Joining word features without n-grams and collocations...", end = "")
    feature_names_t = feature_names_w+feature_names_c
    X_train_t = hstack((X_train_w, X_train_c))
    X_test_t = hstack((X_test_w, X_test_c))
    print("[DONE]")
    
    return X_train_w, Y_train, X_test_w, Y_test, vectorizer, feature_names_w, X_train_w2, X_test_w2, vectorizer2, feature_names_w2, X_train_c, X_test_c, feature_names_c, X_train_t, X_test_t, feature_names_t

def SetEnvironment():

    # This method was used to create the train and test datasets and 
    # extract some feature matrices from this datasets in a 
    # first stage of the project. This method might be outdated.

    print("Creating train dataset...")
    CreateDataset("110_SpeakerMap.txt", "speeches_110.txt", "HS110_members.csv", True, "train_dataset.csv", includeNames = False)
    
    print("Creating test dataset...")
    CreateDataset("112_SpeakerMap.txt", "speeches_112.txt", "HS112_members.csv", False, "test_dataset.csv", includeNames = False)
    
    print("Extracting word features...")
    (X_train_w, Y_train_w, X_test_w, Y_test_w, vectorizer, BoW_names) = features.ExtractWordFeatures("train_dataset.csv", "test_dataset.csv")
    
    print("Extracting collocation features...")
    (X_train_1000, Y_train_1000, X_test_1000, Y_test_1000, feature_names_1000) = features.ExtractCollocationFeatures("train_dataset.csv", "test_dataset.csv", "X_train_1000.txt", "X_test_1000.txt", 5, 1000) 
    
    print("Creating total matrices...")
    feature_words = vectorizer.get_feature_names()
    total_features = feature_words+feature_names_1000
    X_train_t = hstack((X_train_w, X_train_1000))
    X_test_t = hstack((X_train_w, X_train_1000))
    
    return X_train_w, Y_train_w, X_test_w, Y_test_w, X_train_1000, Y_train_1000, X_test_1000, Y_test_1000, X_train_t, X_test_t, vectorizer, feature_words, feature_names_1000, total_features
        