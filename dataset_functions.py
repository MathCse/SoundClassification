import numpy as np
import librosa as lb
import glob
import os
import librosa.display
import pickle


# ************ Pense-bête des différents attributs ************

# MFCCS = 40 floats (taille fixable avec n_mfcc)
# chroma = 12 floats
# mel = 128 floats
# contrast = 7 floats
# tonnetz = 6 floats

# *************************************************************


# Il faut faire une database plus grande pour des test plus objectif
# Il nous faut d'autres labels
labelsDic = {'R': 0, 'F': 1}
datasetDic ={'Rain': 0, 'Fire crackling': 1,'Baby cry': 2,'Chainsaw':3,'Clock tick':4,'Dog bark':5,'Helicopter':6,
             'Person sneeze':7,'Rooster':8,'Sea waves':9}

#Fonction pour extraire les attributs
def extractFeatures(filename):
    X, sample_rate= librosa.load(filename)
    stft= np.abs(lb.stft(X))
    mfccs= np.mean(lb.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
    chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(lb.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(lb.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(lb.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    zcr = lb.feature.zero_crossing_rate(y=X)
    rmse = lb.feature.rmse(y=X)

    mean_zcr = np.mean(zcr)
    std_zcr = np.std(zcr)
    mean_rmse = np.mean(rmse)
    std_rmse = np.std(rmse)
    return mfccs, chroma, mel, contrast, tonnetz, mean_zcr, std_zcr, mean_rmse,std_rmse

def extractFeatures2(X,sample_rate):
    stft= np.abs(lb.stft(X))
    mfccs= np.mean(lb.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
    chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(lb.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(lb.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(lb.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    zcr = lb.feature.zero_crossing_rate(y=X)
    rmse = lb.feature.rmse(y=X)

    mean_zcr = np.mean(zcr)
    std_zcr = np.std(zcr)
    mean_rmse = np.mean(rmse)
    std_rmse = np.std(rmse)
    return mfccs, chroma, mel, contrast, tonnetz, mean_zcr, std_zcr, mean_rmse,std_rmse

def parseAudioFiles2(filepath,subdirs,file_ext='*.ogg'):
    features, labels= np.empty((0,197)), np.empty(0)
    for subdir in subdirs:
        for fn in glob.glob(os.path.join(filepath,subdir,file_ext)):
            mfccs, chroma, mel, contrast, tonnetz, mzcr,stdzcr,mrmse,stdrmse= \
                extractFeatures(fn) # extraction des attributs d'un son
            extFeatures = np.hstack([mfccs, chroma, mel, contrast, tonnetz,mzcr,stdzcr,
                                     mrmse,stdrmse ]) # Regroupes les attributs du son dans un tab
            features = np.vstack([features, extFeatures]) # Met le tableau d'attributs dans la matrice
            labels = np.append(labels, [datasetDic[subdir]]) # Convertit le label à la valeur associé du dictionnaire
    return np.array(features),np.array(labels,dtype=np.int)


#Fonction qui ma permis de définir la taille des divers attributs
def testAttribut(filename):
    X, sample_rate = librosa.load(filename)
    stft = np.abs(lb.stft(X))
    chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    print(len(chroma))

def makeAttributPersistence(name,feature):
    featureFile = open(name+'.pkl', 'wb')  # Ouverture du fichier
    pickle.dump(feature, featureFile)  # Ecriture sur le fichier
    featureFile.close()  # Fermuture du fichier

#Cherche la persistence correspondante, si inexistente la crée
def initDataset(maindir,subdirs):
    if os.path.isfile('dataset_features.pkl') and os.path.isfile('dataset_labels.pkl'):
        featureFile = open('dataset_features.pkl', 'rb')  # Ouverture du fichier
        feature = pickle.load(featureFile)  # Lecture du fichier
        featureFile.close()  # Fermeture du fichier

        labelFile = open('dataset_labels.pkl', 'rb')
        label= pickle.load(labelFile)
        labelFile.close()

    else:
        feature,label=parseAudioFiles2(maindir,subdirs)

        featureFile = open('dataset_features.pkl', 'wb')  # Ouverture du fichier
        pickle.dump(feature,featureFile)  # Ecriture sur le fichier
        featureFile.close() # Fermuture du fichier

        labelFile = open('dataset_labels.pkl', 'wb')
        pickle.dump(label,labelFile)
        labelFile.close()

    return feature,label


def soundAnalysis(path,maxDB):
    features= np.empty((0, 197))
    y,sr=lb.load(path)
    m = lb.samples_to_time(lb.effects.split(y=y, top_db=maxDB))

    for sound in m:
        y, sr = lb.load(path,offset=sound[0],duration=sound[1]-sound[0])
        mfccs, chroma, mel, contrast, tonnetz, mzcr, stdzcr, mrmse, stdrmse = \
            extractFeatures2(y,sr)  # extraction des attributs d'un son
        extFeatures = np.hstack([mfccs, chroma, mel, contrast, tonnetz, mzcr, stdzcr,
                                 mrmse, stdrmse])  # Regroupes les attributs du son dans un tab
        features = np.vstack([features, extFeatures])  # Met le tableau d'attributs dans la matrice

    return np.array(features),m

def soundAnalysis2(path,maxDB) :
    features = np.empty((0, 197))
    y, sr = lb.load(path)
    sounds = lb.effects.split(y=y, top_db=maxDB)
    m = lb.samples_to_time(sounds)

    for sound in sounds:
        path_sound = "sound.wav"
        lb.output.write_wav(path_sound, y[sound[0]:sound[1]], sr)

        y_sound, sr_sound = lb.load(path_sound)

        mfccs, chroma, mel, contrast, tonnetz, mzcr, stdzcr, mrmse, stdrmse = \
            extractFeatures2(y_sound, sr_sound)  # extraction des attributs d'un son

        extFeatures = np.hstack([mfccs, chroma, mel, contrast, tonnetz, mzcr, stdzcr,
                                 mrmse, stdrmse])  # Regroupes les attributs du son dans un tab


        features = np.vstack([features, extFeatures])  # Met le tableau d'attributs dans la matrice

        os.remove(path_sound)

    return np.array(features), m