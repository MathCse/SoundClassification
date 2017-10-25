import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import librosa as lb
import os
import librosa.display
from IPython.display import Image
from sklearn.externals.six import StringIO

datasetDic ={'Rain': 0, 'Fire crackling': 1,'Baby cry': 2,'Chainsaw':3,'Clock tick':4,'Dog bark':5,'Helicopter':6,
             'Person sneeze':7,'Rooster':8,'Sea waves':9}


#Fonction pour faire un spectrogramme à partir d'un son importé
def specAudioFile(filepath):
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

#generateur d'image pour tree (requis graphviz pour utilisation de la cmd dot)
def treetoimg(classifier):
    dot_file = open("iris_decision_tree.dot", 'w')
    tree.export_graphviz(classifier, out_file=dot_file,
                         filled=True, rounded=True,
                         special_characters=True)
    dot_file.close()

    os.system('dot -Tpng iris_decision_tree.dot -o iris_decision_tree.png')
    Image('iris_decision_tree.png')

def plotwaverform(sound):
    y,sr =lb.load(sound)
    m=lb.samples_to_time(lb.effects.split(y=y, top_db=12))
    print(m)
    plt.figure()
    plt.subplot(2,1,1)
    librosa.display.waveplot(y,sr,color='black')
    plt.title("Waveform")
    for line in m:
            plt.axvspan(line[0],line[1],color='green',alpha=0.5,ymin=0.1,ymax=0.9)
    plt.show()

def plotwaverform2(sound,y_pred,m):
    y,sr =lb.load(sound)
    plt.figure()
    ax = plt.subplot(2,1,1)
    librosa.display.waveplot(y,sr,color='black')
    plt.title("Waveform")
    for line,predlabel in zip(m,y_pred):
         plt.axvspan(line[0],line[1],color='green',alpha=0.5,ymin=0.1,ymax=0.9)
         for k,v in datasetDic.items():
             if v == predlabel:
                plt.Axes.text(x=(line[0]+line[1])/2,y=-0.465,s=k,self=ax,color="blue",size=6,horizontalalignment='center')
    plt.show()