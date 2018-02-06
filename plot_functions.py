import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import librosa as lb
import os
import librosa.display
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from matplotlib import cm as clm
import itertools

datasetDic ={'Rain': 0, 'Fire crack': 1,'Baby cry': 2,'Chainsaw':3,'Clock':4,'Dog bark':5,'Helico':6,
             'sneeze':7,'Rooster':8,'Sea waves':9, 'Non reconnu':10}


#Fonction pour faire un spectrogramme à partir d'un son importé
def specAudioFile(filepath,title=""):
    y, sr = lb.load(filepath)
    D= np.abs(lb.stft(y))**2
    S=lb.feature.melspectrogram(S=D)
    S=lb.feature.melspectrogram(y=y,sr=sr,n_mels=128,fmax=8000)

    plt.figure(figsize=(10,4))
    lb.display.specshow(lb.power_to_db(S,ref=np.max),y_axis='mel',fmax=8000,x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram ' + title)
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
                print(k)
                plt.Axes.text(x=(line[0]+line[1])/2,y=-1,s=k,self=ax,color="blue",size=6,horizontalalignment='center')
    plt.show()



def confusion_matrix(cm,classes,normalize=True,cmap= clm.get_cmap('YlOrBr')  ,title='Confusion Matrix'):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=100)
    # classifier.fit(X_train, y_train)
    # y_predict = classifier.predict(X_test)

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


def printprobatable(table):

    columns= ['Rain', 'Fire crackling','Baby cry','Chainsaw','Clock tick','Dog bark','Helicopter','Person sneeze',
              'Rooster', 'Sea waves']

    fig,ax = plt.subplots()
    ncols,nrows=len(columns),len(table)
    hcell, wcell = 0.1, 0.1
    hpad, wpad = 0.5, 0.5


    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    printtable = ax.table(cellText=table,
            colLabels=columns,
            loc='center')
    printtable.set_fontsize(100)
    printtable.scale(1,1)
    plt.show()

def rmseprint(sound):
    y,sr= lb.load(sound,sr=5000)

    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rmse(S=S)

    plt.figure()
    plt.semilogy(rms.T, label='RMS Energy')
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend(loc='best')
    plt.show()

def rmsederiveeprint(sound):
    y,sr= lb.load(sound,sr=5000)

    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rmse(S=S)

    y = []

    for i in range(len(rms[0])-1):
        y.append((rms[0][i+1]-rms[0][i]))

    plt.figure()
    plt.plot(y)
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend(loc='best')
    plt.axis('auto')
    plt.xlabel('Temps')
    plt.show()


def printdetecgraph(sound,sr=22050):
    y, sr = lb.load(sound, sr=sr)

    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rmse(S=S)

    x = []

    for i in range(len(rms[0])-1):
        x.append((rms[0][i+1]-rms[0][i]))

    plt.figure(1)
    plt.subplot(311)

    plt.semilogy(rms.T, label='RMS Energy')
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend(loc='best')

    plt.subplot(312)
    plt.plot(x)
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend(loc='best')
    plt.axis('auto')

    plt.subplot(313)
    librosa.display.waveplot(y, sr, color='black')


    plt.show()

def regroupevent(events):
    new= [0][4]
    count = 1
    bornesup = 0
    borneinf = 0
    while count < len(events):
        if events[count][1] == events[borneinf][1]:
            bornesup=count
        if events[borneinf][1] != events[count][1]:
            event= np.hstack((events[borneinf][0] + events[bornesup][0]) / 2, events[borneinf][1], events[bornesup][2], events[borneinf][3])
            new=np.vstack(new,event)
            bornesup=count
            borneinf=count
        count+=1
    new.append([(events[borneinf][0] + events[bornesup][0]) / 2, events[borneinf][1], events[bornesup][2],
                          events[borneinf][3]])
    print(new)
    return new


def printanalysis(path,events):
    y,sr =lb.load(path)
    plt.figure()
    ax = plt.subplot(2,1,1)
    librosa.display.waveplot(y,sr,color='black')
    plt.title("Waveform")
    for line in events:
         plt.axvspan(line[2],line[3],color='green',alpha=0.5,ymin=0.1,ymax=0.9)
         for k,v in datasetDic.items():
             if v == line[1]:
                plt.Axes.text(x=(line[2]+line[3])/2,y=-1,s=str(int(line[1])),self=ax,color="blue",size=6,horizontalalignment='center')
    plt.show()