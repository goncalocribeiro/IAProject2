import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

# 4.1 Escolha de Features (1 valor)
# numero de letras
# numero de vogais

def features(X):
    
    F = np.zeros((len(X),5))
    for x in range(0,len(X)):
        F[x,0] = len(X[x])
        F[x,1] = #implementar ???
        F[x,2] = #implementar ???
        F[x,3] = #implementar ???
        F[x,4] = #implementar ???

    return F   

# 4.2 Metodo de Aprendizagem (2 valores)
def mytraining(f,Y):
    clf = #implementar
    clf = clf.fit(f, Y)
   
    return clf
    
def mytrainingaux(f,Y,par):
	clf = #implementar
    clf = clf.fit(f, Y)
    
    return clf

def myprediction(f, clf):
    Ypred = clf.predict(f) # os metodos no scikit learn tem todos uma função predict

    return Ypred

