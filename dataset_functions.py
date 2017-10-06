import numpy as np
import librosa as lb
import glob
import os
import librosa.display
import pickle


# ************ Pense-bête des différents attributs ************

# MFCCS = 40 floats (taille fixable avec n_mfcc)
# chroma = 6 floats
# mel = 128 floats
# contrast = 7 floats
# tonnetz = 6 floats

# *************************************************************


# Il faut faire une database plus grande pour des test plus objectif
# Il nous faut d'autres labels
labelsDic = {'R': 0, 'F': 1}

#Fonction pour extraire les attributs
def extractFeatures(filename):
    X, sample_rate= librosa.load(filename)
    stft= np.abs(lb.stft(X))
    mfccs= np.mean(lb.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
    chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(lb.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(lb.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(lb.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast, tonnetz

#Fonction pour mettre les divers attributs de tout les sons d'un dossier dans une matrice
def parse_audio_files(filepath,file_ext='*.ogg'):
    features, labels= np.empty((0,193)), np.empty(0)
    for fn in glob.glob(os.path.join('dataset',filepath,file_ext)):
        mfccs, chroma, mel, contrast, tonnetz = extractFeatures(fn) # extraction des attributs d'un son
        extFeatures = np.hstack([mfccs, chroma, mel, contrast, tonnetz]) # Regroupes les attributs du son dans un tab
        features = np.vstack([features, extFeatures]) # Met le tableau d'attributs dans la matrice
        labelName=fn.split('-')[2].split('.')[0] # Decoupe le filename pour récupérer le label
        labels = np.append(labels, [labelsDic[labelName]]) # Convertit le label à la valeur associé du dictionnaire
    return np.array(features),np.array(labels,dtype=np.int)


#Fonction qui ma permis de définir la taille des divers attributs
def testAttribut(filename):
    X, sample_rate = librosa.load(filename)
    stft = np.abs(lb.stft(X))
    tonnetz = np.mean(lb.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    print(len(tonnetz))
    print(tonnetz)

#Cherche la persistence correspondante, si inexistente la crée
def initDataset(filename):
    if os.path.isfile('dataset_features.pkl') and os.path.isfile('dataset_labels.pkl'):
        featureFile = open('dataset_features.pkl', 'rb')  # Ouverture du fichier
        feature = pickle.load(featureFile)  # Lecture du fichier
        featureFile.close()  # Fermeture du fichier

        labelFile = open('dataset_labels.pkl', 'rb')
        label= pickle.load(labelFile)
        labelFile.close()

    else:
        feature,label=parse_audio_files(filename)

        featureFile = open('dataset_features.pkl', 'wb')  # Ouverture du fichier
        pickle.dump(feature,featureFile)  # Ecriture sur le fichier
        featureFile.close() # Fermuture du fichier

        labelFile = open('dataset_labels.pkl', 'wb')
        pickle.dump(label,labelFile)
        labelFile.close()

    return feature,label