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
    cnf_matrix = np.zeros((10, 10))
    acc=0
    for i in range(50):
        X_train, X_test, y_train, y_test = pp.splitAndPP(X, y, 0.15) #Preprocessing


        classifier = RandomForestClassifier(n_estimators=90, max_features="sqrt", criterion="gini",oob_score= False,
                                              max_depth= 15)
        classifier2 = KNeighborsClassifier(n_neighbors=1)
        classifier3= GaussianNB()
        test= MLPClassifier(solver="lbfgs", activation="relu",tol=1e-4 )
        # test= VotingClassifier(estimators=[('rf',classifier),('kn',classifier2),('nt',classifier4)]
        #                            ,voting='soft',weights=[30,15,15],n_jobs=1)
        test.fit(X_train, y_train)
        y_predict = test.predict(X_test)

        acc+=accuracy_score(y_test,y_predict)


        # Compute confusion matrix
        cnf_matrix = np.add(confusion_matrix(y_test, y_predict),cnf_matrix)
        print(cnf_matrix)

    test=cnf_matrix
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plf.confusion_matrix(test, classes=datasetDir,
                          title='Normalized confusion matrix : Neural network')
    print(acc / 50)
    plt.show()



if __name__ == '__main__':

    mymain()
