#A030, Goncalo Ribeiro 82303, Andre Mendonca 82304

import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def tem_repetidas(p):
    seen = set()  # O(1) lookups
    for x in p:
        if x not in seen:
            seen.add(x)
        else:
            return True
    return False        
    
def features(X):
    
    F = np.zeros((len(X),6))
    for x in range(0,len(X)):
        F[x,0] = len(X[x]) # tamanho da palavra x em X
        F[x,1] = n_vogais_par(X[x]) # numero de vogais e par
        F[x,2] = primeira_letra_vogal(X[x]) # primeira letra ser vogal
        F[x,3] = tem_acentuacao(X[x]) # ter acentuacao
        F[x,4] = tem_repetidas(X[x]) #tem letras repetidas
        
        #numero impar de letras FALSE
        #comecar e acabar em vogal TRUE
        #ultimas 3 serem palavra de 3
        

    return F   

# 4.2 Metodo de Aprendizagem (2 valores)
def mytraining(f,Y):
    #clf = neighbors.KNeighborsClassifier(n_neighbors=NEIGHBORS, algorithm='auto') #teste1: 74%, teste2: 77%
    clf = tree.DecisionTreeClassifier() #teste1: 84%, teste2: 84% <---------- MELHOR ESCOLHA
    #clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') #teste1: 73%, teste2: 79%
    clf = clf.fit(f, Y)
    return clf
    
'''def mytrainingaux(f,Y,par):
    clf = #implementar
    clf = clf.fit(f, Y)
    
    return clf'''

def myprediction(f, clf):
    Ypred = clf.predict(f) # os metodos no scikit learn tem todos uma funcao predict

    return Ypred
