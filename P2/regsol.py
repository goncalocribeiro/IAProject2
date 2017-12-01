import numpy as np
from sklearn import datasets, tree, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit

def mytraining(X,Y):
    #reg = linear_model.LinearRegression()
    #reg = linear_model.BayesianRidge()
    reg = tree.DecisionTreeRegressor()
    #reg = tree.DecisionTreeRegressor(max_depth=416)
    reg.fit(X,Y)
   
    return reg
    
'''def mytrainingaux(X,Y,par):
    
    reg.fit(X,Y)
                
    return reg'''

def myprediction(X,reg):

    Ypred = reg.predict(X)

    return Ypred
