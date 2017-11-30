import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#NEIGHBORS = 0 #numero de palavras

# 4.1 Escolha de Features (1 valor)
# numero de vogais
vogais = ('a','e','i','o','u')
def n_vogais_par(p):
	cont = 0
	for l in range(0,len(p)):
		if p[l] in vogais:
			cont += 1
	return cont%2 == 0

def primeira_letra_vogal(p):
	return p[0] in vogais

def tem_acentuacao(p):
	return all(ord(c) < 128 for c in p)

def features(X):
    #global NEIGHBORS
    #NEIGHBORS = len(X)
    
    #print(X)
    
    F = np.zeros((len(X),5))
    for x in range(0,len(X)):
        F[x,0] = len(X[x]) # tamanho da palavra x em X
        F[x,1] = n_vogais_par(X[x]) # numero de vogais e par
        F[x,2] = primeira_letra_vogal(X[x]) # primeira letra ser vogal
        F[x,3] = tem_acentuacao(X[x]) # ter acentuacao
        #F[x,4] = # implementar ???

    return F   

# 4.2 Metodo de Aprendizagem (2 valores)
def mytraining(f,Y):
    #clf = neighbors.KNeighborsClassifier(n_neighbors=NEIGHBORS, algorithm='auto') #74%, 77%
    clf = tree.DecisionTreeClassifier() #83%, 83%
    #clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') #73%, 79%
    clf = clf.fit(f, Y)
    return clf
    
'''def mytrainingaux(f,Y,par):
    clf = #implementar
    clf = clf.fit(f, Y)
    
    return clf'''

def myprediction(f, clf):
    Ypred = clf.predict(f) # os metodos no scikit learn tem todos uma funcao predict

    return Ypred
