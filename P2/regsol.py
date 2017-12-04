import numpy as np
from sklearn import datasets, tree, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import time

def mytraining(X,Y):
    #min_samples_split = 8
    #reg = tree.DecisionTreeRegressor(min_samples_split=min_samples_split)
    #reg = tree.DecisionTreeRegressor(max_depth=416)
    #reg = KernelRidge(kernel='rbf', gamma=0.1,alpha=0.1)
    
    reg =  GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1, alpha=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
    
    t0 = time.time()
    
    reg.fit(X,Y)
    
    reg_fit = time.time() - t0
    print("KRR complexity and bandwidth selected and model fitted in %.3f s"
          % reg_fit)  
    
    X_plot = np.linspace(0, 5, 416)[:, None]
    t0 = time.time()
    y_reg = reg.predict(X_plot)
    reg_predict = time.time() - t0
    print("KRR prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], reg_predict))
   
    return reg
    
'''def mytrainingaux(X,Y,par):
    
    reg.fit(X,Y)
                
    return reg'''

def myprediction(X,reg):

    Ypred = reg.predict(X)

    return Ypred
