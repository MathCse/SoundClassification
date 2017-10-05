import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa as lb
import glob
import os
import librosa.display
import scipy


#Fonction pour faire un spectrogramme à partir d'un son importé
def importAudioFile(filepath):
    y, sr = lb.load(filepath)
    D= np.abs(lb.stft(y))**2
    S=lb.feature.melspectrogram(S=D)
    S=lb.feature.melspectrogram(y=y,sr=sr,n_mels=128,fmax=8000)

    plt.figure(figsize=(10,4))
    lb.display.specshow(lb.power_to_db(S,ref=np.max),y_axis=('mel'),fmax=8000,x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

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

def parse_audio_files(filepath,file_ext='*.ogg'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(filepath):
        for fn in glob.glob(os.path.join(filepath, file_ext)):
            mfccs, chroma, mel, contrast,tonnetz = extractFeatures(fn)
            extFeatures = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,extFeatures])
            labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int)

def mymain():
    features, labels = parse_audio_files('datasets','dataset1')
    print(features)
    print()
    print(labels)

if __name__ == '__main__':
    mymain()