import dataset_functions as dt
import time
import classifier_function as cf
import librosa as lb
import plot_functions as plf
import numpy as np
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

    X_train, X_test, y_train, y_test = pp.splitAndPP(X, y, 0.2) #Preprocessing



if __name__ == '__main__':

    mymain()
