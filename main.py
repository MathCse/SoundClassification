import dataset_functions as dt
import time
import classifier_function as cf



# ************ Pense-bête des différents attributs ************

# MFCCS = 40 floats (taille fixable avec n_mfcc)
# chroma = 6 floats
# mel = 128 floats
# contrast = 7 floats
# tonnetz = 6 floats

# *************************************************************




def mymain():
    features1,labels1 = dt.initDataset("training")
    X = features1[:,0:40]
    y = labels1
    cf.build_decisiontree(X,y,0.2)
    cf.build_knn(X, y,3,0.2)
    cf.build_randomforest(X, y, 0.2)
    cf.build_dummy(X, y, 0.2)
    cf.build_neuralNetwork(X, y, 0.2)
    cf.build_naivebayes(X, y, 0.2)
    cf.build_linearsvm(X, y, 0.2)
    cf.build_rbfsvm(X, y, 0.2)
    cf.build_adaboost(X, y, 0.2)
    cf.build_gaussianprocess(X, y, 0.2)

if __name__ == '__main__':
    start_time = time.time()
    mymain()
    print("--- %s seconds ---" % (time.time() - start_time))