import dataset_functions as dt
import time
import classifier_function as cf
import librosa as lb
import plot_functions as plf
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC,LinearSVR
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from matplotlib import cm as clm
import pre_processing as pp

# ************ Pense-bête des différents attributs ************

# MFCCS = 40 floats (taille fixable avec n_mfcc)
# chroma = 12 floats
# mel = 128 floats
# contrast = 7 floats
# tonnetz = 6 floats

# *************************************************************

datasetDir = ['Rain', 'Fire crackling','Baby cry','Chainsaw','Clock tick','Dog bark','Helicopter','Person sneeze',
              'Rooster', 'Sea waves']



def mymain():

    features1,labels1 = dt.initDataset("dataset",datasetDir)

    X = features1[:,0:197]
    y = labels1

    table = cf.build_randomforest(X,y,0.15)
    plf.rmseprint('test2.wav')
    plf.plotwaverform('test2.wav')

 # # Compute confusion matrix
    #  cnf_matrix = np.add(confusion_matrix(y_test, y_predict),cnf_matrix)
    # print(cnf_matrix)

    # test=cnf_matrix
    # np.set_printoptions(precision=2)

    # # Plot normalized confusion matrix
    # plt.figure()
    # plf.confusion_matrix(test, classes=datasetDir,title='Normalized confusion matrix : Neural network')
    # print(acc / 50)
    # plt.show()



if __name__ == '__main__':

    mymain()
