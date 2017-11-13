import dataset_functions as dt
import time
from sklearn.model_selection import train_test_split
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import classifier_function as cf
from sklearn.feature_extraction import DictVectorizer
import math
import plot_functions as plf

def splitAndPP(features, label, size):
    vec = DictVectorizer()
    #vec.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size)
    #y_train = np.transpose(y_train)
    #print(X_train)
    #print(y_train)
    #Xtrain = np.concatenate((y_train[np.newaxis, :], X_train),axis=0)
    #print(vec.fit(Xtrain).toarray())

    X_train, moyenne, ecartype = preprocessingtraining(X_train)


    X_test = preprocessingtest(X_test,moyenne,ecartype)

    return X_train, X_test, y_train, y_test


def preprocessingtraining(X):
    moyenne = []
    ecartype = []

    X[:, 0:40] += 1000
    X[:, 0:40] = np.log(X[:, 0:40])
    moyenne.append(np.mean(X[:, 0:40]))
    ecartype.append(np.std(X[:, 0:40]))
    X[:, 0:40] = ((X[:, 0:40]) - moyenne[0]) / ecartype[0]

    X[:, 40:52] = np.log(X[:, 40:52])
    moyenne.append(np.mean(X[:, 40:52]))
    ecartype.append(np.std(X[:, 40:52]))
    X[:, 40:52] = ((X[:, 40:52]) - moyenne[1]) / ecartype[1]

    X[:, 52:180] = np.log(X[:, 52:180])
    moyenne.append(np.mean(X[:, 52:180]))
    ecartype.append(np.std(X[:, 52:180]))
    X[:, 52:180] = ((X[:, 52:180]) - moyenne[2]) / ecartype[2]

    X[:, 180:187] += 1000
    X[:, 180:187] = np.log(X[:, 180:187])
    moyenne.append(np.mean(X[:, 180:187]))
    ecartype.append(np.std(X[:, 180:187]))
    X[:, 180:187] = ((X[:, 180:187]) - moyenne[3]) / ecartype[3]

    X[:, 187:193] += 1000
    X[:, 187:193] = np.log(X[:, 187:193])
    moyenne.append(np.mean(X[:, 187:193]))
    ecartype.append(np.std(X[:, 187:193]))
    X[:, 187:193] = ((X[:, 187:193]) - moyenne[4]) / ecartype[4]
    return X, moyenne, ecartype

def preprocessingtest(X, moyenne, ecartype):
    X[:, 0:40] += 1000
    X[:, 0:40] = np.log(X[:, 0:40])
    X[:, 0:40] = ((X[:, 0:40]) - moyenne[0]) / ecartype[0]

    X[:, 40:52] = np.log(X[:, 40:52])
    X[:, 40:52] = ((X[:, 40:52]) - moyenne[1]) / ecartype[1]

    X[:, 52:180] = np.log(X[:, 52:180])
    X[:, 52:180] = ((X[:, 52:180]) - moyenne[2]) / ecartype[2]

    X[:, 180:187] += 1000
    X[:, 180:187] = np.log(X[:, 180:187])
    X[:, 180:187] = ((X[:, 180:187]) - moyenne[3]) / ecartype[3]

    X[:, 187:193] += 1000
    X[:, 187:193] = np.log(X[:, 187:193])
    X[:, 187:193] = ((X[:, 187:193]) - moyenne[4]) / ecartype[4]

    return X