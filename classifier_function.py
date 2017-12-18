from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC,LinearSVR
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
import numpy as np
import pre_processing as pp

# ************ Pense-bête des différents classifiers ************

# Vous trouverez ici toutes les différents classifiers avec des
# parametres plus ou moins standards. Le size représente la por-
# tion de test sur le dataset

# *************************************************************



def build_decisiontree(features,label,size):

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train,y_train)
    y_predict=classifier.predict(X_test)

    print("Decision Tree Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)


def build_knn(features, label, n_number, size):
    X_train, X_test, y_train, y_test = pp.splitAndPP(features, label, 0.15)
    classifier = KNeighborsClassifier(n_neighbors=n_number,weights="distance",algorithm="auto",)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Knn Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_randomforest(features,label,size):
    X_train, X_test, y_train, y_test = pp.splitAndPP(features, label, 0.15)
    classifier=RandomForestClassifier(n_estimators=90, max_features="sqrt", criterion="gini",oob_score= False,
                                      max_depth= 15)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    y_predict_proba= np.around(classifier.predict_proba(X_test), decimals=2)
    print("Random Forest Accuracy : %s" % accuracy_score(y_test, y_predict))
    print(y_predict_proba)
    return y_predict_proba


def rfanalysis(features,label):
    classifier=RandomForestClassifier(n_estimators=90, max_features="sqrt", criterion="gini",oob_score= False,
                                      max_depth= 15)
    classifier.fit(features,label)
    return classifier


def voting_classifier2(X_train,y_train,X_test):
    classifier = RandomForestClassifier(n_estimators=90, max_features="sqrt", criterion="gini",oob_score= False,
                                      max_depth= 15)
    classifier2= KNeighborsClassifier(n_neighbors=2)
    classifier4= MLPClassifier(solver="lbfgs", activation="relu",tol=1e-4 )
    classifier5= SVC(kernel="linear", C=0.001, probability=True)
    test= VotingClassifier(estimators=[('rf',classifier),('kn',classifier2),('nt',classifier4),
                                       ('lsvm',classifier5)]
                           ,voting='soft',weights=[3,3,3,1],flatten_transform=True,n_jobs=1)
    test.fit(X_train,y_train)
    y_predict=test.predict(X_test)
    return y_predict

def build_dummy(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=DummyClassifier()
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Dummy Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_neuralNetwork(features,label,size):
    X_train, X_test, y_train, y_test = pp.splitAndPP(features, label, 0.15)
    classifier=MLPClassifier(solver="lbfgs", activation="relu",tol=1e-4 )
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Neural Network Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_naivebayes(features,label,size):
    X_train, X_test, y_train, y_test = pp.splitAndPP(features, label, 0.15)
    classifier=GaussianNB()
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Naive Bayes Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_linearsvm(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=SVC(kernel="linear", C=0.001 , probability=True)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Linear-svm : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_rbfsvm(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=SVC(gamma='auto', C=5)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("RBF-svm Accuracy: %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_adaboost(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=AdaBoostClassifier()
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("ADAboost Accuracy: %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

#Warning avec cette fonction, faire recherche avant utilisation
def build_quadraticanalysis(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=QuadraticDiscriminantAnalysis()
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("QDA Accuracy: %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_gaussianprocess(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    print("Gaussian process Accuracy: %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def voting_classifier(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier = RandomForestClassifier(n_estimators=90, max_features="sqrt", criterion="gini",oob_score= False,
                                      max_depth= 15)
    classifier2= KNeighborsClassifier(n_neighbors=2)
    # classifier3= GaussianNB()
    classifier4= MLPClassifier(solver="lbfgs", activation="relu",tol=1e-4 )
    classifier5= SVC(kernel="linear", C=0.001, probability=True)
    test= VotingClassifier(estimators=[('rf',classifier),('kn',classifier2),('nt',classifier4),
                                       ('lsvm',classifier5)]
                           ,voting='soft',weights=[3,3,3,1],flatten_transform=True,n_jobs=1)
    test.fit(X_train,y_train)
    y_predict=test.predict(X_test)
    print("Voting Classifier: %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def voting_classifier2(X_train,y_train,X_test):
    classifier = RandomForestClassifier(n_estimators=90, max_features="sqrt", criterion="gini",oob_score= False,
                                      max_depth= 15)
    classifier2= KNeighborsClassifier(n_neighbors=2)
    # classifier3= GaussianNB()
    classifier4= MLPClassifier(solver="lbfgs", activation="relu",tol=1e-4 )
    classifier5= SVC(kernel="linear", C=0.001, probability=True)
    test= VotingClassifier(estimators=[('rf',classifier),('kn',classifier2),('nt',classifier4),
                                       ('lsvm',classifier5)]
                           ,voting='soft',weights=[3,3,3,1],flatten_transform=True,n_jobs=1)
    test.fit(X_train,y_train)
    y_predict=test.predict(X_test)
    return y_predict