import dataset_functions as dt
import time
import classifier_function as cf
import librosa
import plot_functions as plf
import numpy as np
from sklearn.ensemble import voting_classifier

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

    X = features1[:,0:193]
    y = labels1

    for i in range(10):
        cf.voting_classifier(X,y,0.2)


if __name__ == '__main__':

    mymain()
