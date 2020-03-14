"""
    Created on Fri Nov 06 12:32:20 2015
	@author: Javier Carnerero Cano 
    @author: Felix Jimenez Monzon 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import spatial
from sklearn.metrics import mean_squared_error

""" Funcion que implementa los procesos gaussianos """
def gaussian_processes(X_train, y_train, X_test, sigma_eps, l):
    
    dist = spatial.distance.cdist(X_train, X_train, 'euclidean')   
    dist_s = spatial.distance.cdist(X_test, X_train, 'euclidean')    
    K = (1 + (5 ** 0.5 * dist)/l + (5 * dist ** 2)/(3 * l ** 2)) * np.exp(-(5 ** 0.5 * dist)/l)
    K_s = (1 + (5 ** 0.5 * dist_s)/l + (5 * dist_s ** 2)/(3 * l ** 2)) * np.exp(-(5 ** 0.5 * dist_s)/l)
    
    # Posterior distribution of f_star
    m = K_s.dot(np.linalg.inv(K + sigma_eps ** 2 * np.eye(X_train.shape[0]))).dot(y_train) 
    return m

           
""" Funcion que valida los parametros sigma_eps y l de los procesos gaussianos """ 
def validate_gaussian_processes(X_train, y_train, sigma_eps, l, M):
    # This fragment of code runs k-nn with M-fold cross validation
        
    # Obtain the indices for the different folds
    n_tr = X_train.shape[0]        
    """ Vector con elementos de 0 a n_tr - 1 sin repeticion """
    permutation = np.random.permutation(n_tr)
    
    set_indices = {}
    for k in range(M):
        set_indices[k] = []
    k = 0
    """ a cada capeta se le asigna unos indices de X_train """
    for pos in range(n_tr):
        set_indices[k].append(permutation[pos])
        k = (k + 1) % M
        
    # Now, we run the cross-validation process using the GP method 
       
    # Obtain the validation errors
    sq_error_val = np.zeros((len(sigma_eps), len(l)))
    sq_error_val_iter = np.zeros((len(sigma_eps), len(l)))
    
    print " Test mode: %i-fold cross-validation" % (M)    
    print "\n Gaussian Processes"
    print " Kernel used:"
    print " kernel: k(x, x') = exp(-1/(2*l) * ||x - x'||^2) \n"  
    for k in range(M):
        print " Building model for fold %i ..." % (k + 1)
        """ conjunto de validacion """
        val_indices = set_indices[k]
        train_indices = []
        for j in range(M):
            if not j == k:
                """ conjunto de entrenamiento """
                train_indices += set_indices[j] 
        
        for i in range(len(sigma_eps)):
            for j in range(len(l)):        
                sq_error_val_iter[i, j] = mean_squared_error(y_train[val_indices], gaussian_processes(X_train[train_indices, :], y_train[train_indices] \
                                                                           , X_train[val_indices, :], sigma_eps[i], l[j])) ** 0.5 
        sq_error_val += sq_error_val_iter
     
    sq_error_val /= M    
    """ elegimos el minimo de la matriz de errores obtenida, y sus parametros asociados """
    pos_min = np.where(sq_error_val == np.min(sq_error_val[:, :]))
    sigma_eps_opt = np.mean(sigma_eps[pos_min[0]])
    l_opt = np.mean(l[pos_min[1]])
    print "La sigma_eps optima optima es %f y la l optima es %f, que generan un RMSE_validation = %f" % (sigma_eps_opt, l_opt, np.min(sq_error_val[:, :]))
    return sigma_eps_opt, l_opt 
    
    
""" Funcion que preprocesa los datos para calcular los valores nan y normalizar las matrices """
def preprocessing(X_train, X_test, y_train, sigma_eps, l, M):      
         
    ########## Train ###########
         
    """ Creamos las matrices auxiliares X_train_aux, y_train_aux y X_test_aux para predecir 
        los valores nan de la primera columna de X_train.
        # X_train_aux: contiene las columnas 1, 2 y 3 de X_train correspondientes a los valores de la
          columna 0 que no tenian valores nan.
        # y_train_aux: contiene los valores de la columna 0 de X_train que no tenia valores nan
        # X_test_aux: contiene las columnas 1, 2 y 3 de X_train correspondientes a los valores de la
          columna 0 que si tenian valores nan.

    """
    var = 0
    var_2 = 0
    for i in range(X_train.shape[0]):        
        if np.isnan(X_train[i, 0]):
            if var == 0:
                X_test_aux = np.array([X_train[i, range(1, X_train.shape[1])]])
                var = 1
            else:                    
                X_test_aux = np.append(X_test_aux, np.array([X_train[i, range(1, X_train.shape[1])]]), axis = 0)  
        else:
            if var_2 == 0:
                X_train_aux = np.array([X_train[i, range(1, X_train.shape[1])]])
                y_train_aux = np.array([[X_train[i, 0]]])
                var_2 = 1
            else:
                X_train_aux = np.append(X_train_aux, np.array([X_train[i, range(1, X_train.shape[1])]]), axis = 0)
                y_train_aux = np.append(y_train_aux, np.array([[X_train[i, 0]]]), axis = 0)    
                
    """ Normalizamos X_train_aux y X_test_aux con la media y la desviacion tipica de X_train_aux """
    medias_X_train_aux = np.mean(X_train_aux, axis = 0)   
    desvs_X_train_aux = np.std(X_train_aux, axis = 0)  
    X_train_aux = (X_train_aux - medias_X_train_aux) / desvs_X_train_aux
    X_test_aux = (X_test_aux - medias_X_train_aux) / desvs_X_train_aux 
    
    print "\n Imputation of missing data in the first column" 
    sigma_eps_opt, l_opt = validate_gaussian_processes(X_train_aux, y_train_aux, sigma_eps, l, M)    
    y_aux = gaussian_processes(X_train_aux, y_train_aux, X_test_aux, sigma_eps_opt, l_opt)
    
    var = 0
    for i in range(X_train.shape[0]):
        if np.isnan(X_train[i, 0]):
            X_train[i, 0] = y_aux[var] 
            var += 1       
               
    ############ Test #############
    
    var = 0    
    for i in range(X_test.shape[0]): 
        if np.isnan(X_test[i, 0]):
            if var == 0:
                X_test_aux = np.array([X_test[i, range(1, X_test.shape[1])]])          
                var = 1    
            else:
                X_test_aux = np.append(X_test_aux, np.array([X_test[i, range(1, X_test.shape[1])]]), axis = 0)
     
    for i in range(X_test_aux.shape[1]):               
        X_test_aux[:, i] = (X_test_aux[:, i] - medias_X_train_aux[i]) / desvs_X_train_aux[i]    
             
    y_aux = gaussian_processes(X_train_aux, y_train_aux, X_test_aux, sigma_eps_opt, l_opt)    
    
    var = 0
    for i in range(X_test.shape[0]):
        if np.isnan(X_test[i, 0]):
            X_test[i, 0] = y_aux[var] 
            var += 1  
            
    X_train[:, 0] = (X_train[:, 0] - np.mean(y_train_aux)) / np.std(y_train_aux) 
    X_test[:, 0] = (X_test[:, 0] - np.mean(y_train_aux)) / np.std(y_train_aux)   
    X_train[:, range(1, X_train.shape[1])] = (X_train[:, range(1, X_train.shape[1])] - medias_X_train_aux) / desvs_X_train_aux
    X_test[:, range(1, X_test.shape[1])] = (X_test[:, range(1, X_test.shape[1])] - medias_X_train_aux) / desvs_X_train_aux 
    return
        
   
""" Funcion principal del programa """   
def main():
    
    """ Cargamos los datos """
    f_train = 'data_train.csv'
    f_test = 'data_test.csv'    
    output_file = 'y.csv'    
    data_train = pd.read_csv(f_train, header = None)
    X_train = data_train.values[:, : -2]
    y_train = np.array([data_train.values[:, -1]]).T
    X_test = pd.read_csv(f_test, header = None).values[:, : -1]    
    
    M = 10  # number of folds
    
    """
    # GP hyperparameters for imputing the missing data in the first column
    sigma_eps_pre = np.linspace(0.02, 0.042, 60) 
    l_pre = np.linspace(6, 8.5, 60)  
    
    # GP hyperparameters for predicting the y values
    sigma_eps = np.linspace(0.02, 0.042, 60) 
    l = np.linspace(6, 8.5, 60)
    """
    
    # GP hyperparameters for imputing the missing data in the first column
    sigma_eps_pre = np.linspace(0.048, 0.052, 20) 
    l_pre = np.linspace(5.742373, 6.142373, 20)  
    
    # GP hyperparameters for predicting the y values
    sigma_eps = np.linspace(0.029678, 0.034678, 20) 
    l = np.linspace(6.238983, 6.638983, 20)    
    
    """ Preprocesamos los datos para calcular los valores nan y normalizar las matrices """
    preprocessing(X_train, X_test, y_train, sigma_eps_pre, l_pre, M)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X_train[:, 0], y_train, 'b.', markersize = 10) 

    print "\n Prediction of y values"    
    sigma_eps_opt, l_opt = validate_gaussian_processes(X_train, y_train, sigma_eps, l, M)
    
    ######################################################################
    ###########                  TEST                          ###########
    ######################################################################
    y = np.array(gaussian_processes(X_train, y_train, X_test, sigma_eps_opt, l_opt)).flatten()
    
    # Save output file
    with open(output_file,'wb') as f:
            wtr = csv.writer(f, delimiter= ',')
            wtr.writerow(['id', 'prediction'])
            for i, x in enumerate(X_test):
                wtr.writerow([i, y[i]])
                
    print "Fichero guardado"    
    return
   
######################################################################################################
 
if __name__ == "__main__":
    main()

