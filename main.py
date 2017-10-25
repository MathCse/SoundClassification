import dataset_functions as dt
import time
import classifier_function as cf
import librosa as lb
import plot_functions as plf
import numpy as np
from sklearn import preprocessing

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

    X_train = features1[:,0:197]
    y_train = labels1

    X_test,m = dt.soundAnalysis2("pistetest.wav",27)

    y_pred=cf.voting_classifier2(X_train,y_train,X_test)

    plf.plotwaverform2("pistetest.wav",y_pred,m)



if __name__ == '__main__':

    mymain()
