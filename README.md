# Design of a political ideology text classifier using NLP and Machine Learning techniques
In this project several ML and NLP techniques are used to identify the political ideology of the authors of written texts.

## Abstract
Classifying the ideology of political texts can be a difficult task even for a human being. In this project different techniques based on machine learning (ML) and Natural Language Processing (NLP) are applied and compared to identify the political ideology of certain texts.

For the development of the project, we use the transcripts of the speeches of the different members of the Congress of the United States of as input texts, and classify them between two classes: democrats and republicans.
For this purpose, we carry out several experiments combining different ML and NLP techniques such as the use of different classification algorithms such as Naïve Bayes, Logistic Regression, Decision Trees and Support Vector Machines. In addition, different techniques have been combined for the extraction of features to feed the cited algorithms such as Bag of Words (BoW), Collocations, Collocations around the most important words, bigrams and trigrams of words among others.
An analysis of the most frequent topics among the texts and their impact on ideology has also been carried out using the Latent Dirichlet Allocation (LDA) technique.

After comparing different combinations of the techniques and different years for the creation of the training and testing datasets, we obtain the best of the models used and therefore the model proposed in this Master Thesis as a political ideology text classifier with the combination of: 109th and 111th Congress for the creation of the training dataset and 112th Congress for the testing dataset, the use of Logistic Regression as a classification algorithm, and BoW and bigrams as features of the model. This model provides an accuracy of 76.7%.
## Resources:
- **Parsed speeches from the congresses**: This dataset (Gentzkow, Matthew, Jesse M. Shapiro, 2018 is collected by the Stanford University and contains processed text from the bound and daily editions of the United States Congressional Record. The daily edition has information from the 97th to 114th Congresses. This dataset compiles two types of text files per congress:
    -	*Speeches file*. It contains the texts of the speeches together with an identifier (speech_id) of each of them.
    -   *SpeakerMap file*. It is a table with the fields speakerid, speech_id, lastname, firstname, chamber, state, gender, party, district and nonvoting.
- **Nominate ideology and related data**: This dataset (Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, 2019) contains the ideology data of the members of the congress. This data is composed by two different types of fields:
    - *Biographical fields*. A total of 13 fields. We only need the bioname field. 
    - *Ideological fields*. There are 8 different ideological fields. We only use the nominate_dim1 field.

## Structure of the project
The project is organized in the following folders:
- **source_dataset**: It contains the files of the datasources described in `Resources`.
- **datasets**: It stores the training and testing datasets created from the source datasets.
- **feature_matrices**: In this folder feature matrices are stored in txt fromat when created with the methods in `features.py`.
- **scripts**: It stores all the code developed to carry out the project.

However, in order to not increase considerably the size of the repository, source_dataset, datasets and feature_matrices folders have not been included.

## Structure of the code
This repository contains several scripts containing all the different types of functions that have been used in the development of the project. A quick description of the scripts is given below:
- `environment.py`: In this script we include methods for creating the datasets, setting and storing some variables of the environment, together with some experiments.
- `data_analysis.py`: This script contains functions to analyze in deep the data from the given datasets in terms of Information Gain and lenght of the speeches. Understanding for Information  Gain of a speech the average Information Gain of the words of that particular speech.
- `features.py`: # This script contains methods for creating the feature matrices and reading them from the txt files when stored. Three type of features are included in this script: BoW, with three different variations of frequency measurement and the possibility of add bigrams and trigrams to the features, Collocations of two words within a tuneable parameter window of words, and Collocations  around most important words, implemented the same way as normal Collocations.
- `models.py`: This script contains methods for training algorithms.
- `evaluation.py`:This script contains methods for the evaluation of the models.
- `lda.py`: This script compiles all the methods used in the Latent Dirichlet Allocation analysis of the project.
- `experiments.py`: This script implements several experiments. Each experiment is implemented by one  method. To run a experiment just call the method without arguments. 
Example:
```
experiments.RemoveCenterExperimentBoW()
```
## Requirements
- Python 2.7
## How to run
The scripts of this project do not have any main method, therefore to run any of their function it is necessary to open the python interpreter, import the scripts and call their functions. This can be done with the following comands:
````
$ python

>>> import environment
>>> import data_analysis
>>> import features
>>> import models
>>> import evaluation
>>> import lda
>>> import experiments
````

And an example of function call could be:
````
>>> environment.CreateDataset("110_speakerMap.txt", "speeches_110.txt", "HS110_members.csv", train_dataset.txt, True, includeNames = False, long_speeches = True)
````
## Authors
- Javier Moreno

