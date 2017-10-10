from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier


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

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier = KNeighborsClassifier(n_neighbors=n_number)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Knn Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_randomforest(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=RandomForestClassifier(n_estimators=500)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Random Forest Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_dummy(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=DummyClassifier()
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Dummy Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_neuralNetwork(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=MLPClassifier()
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Neural Network Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_naivebayes(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=GaussianNB()
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Naive Bayes Accuracy : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_linearsvm(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=SVC(kernel="linear", C=0.025)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("Linear-svm : %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)

def build_rbfsvm(features,label,size):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=size, random_state=100)
    classifier=SVC(gamma=2, C=1)
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
    classifier = RandomForestClassifier(n_estimators=90)
    classifier2= KNeighborsClassifier(n_neighbors=2)
    classifier3= GaussianNB()
    classifier4= tree.DecisionTreeClassifier()
    test= VotingClassifier(estimators=[('rf',classifier),('kn',classifier2),('gn',classifier3),('nt',classifier4)]
                           ,voting='soft',weights=[2,2,0.75,0.25],flatten_transform=True)
    test.fit(X_train,y_train)
    y_predict=test.predict(X_test)
    print("Voting Classifier: %s" % accuracy_score(y_test, y_predict))
    return accuracy_score(y_test, y_predict)