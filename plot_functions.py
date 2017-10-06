import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import librosa as lb
import os
import librosa.display
from IPython.display import Image

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