# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:56:36 2023

@author: rubya
"""

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


#These are the follwoing websites that have been recommended and I have used 
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html



#Extracting the BSP data from each of the win and place market datasets
data1 = pd.read_csv(r"C:\Users\rubya\Desktop\University\Year2\MDM2\REP3\pricesukplace01012023.csv")
data2 = pd.read_csv(r"C:\Users\rubya\Desktop\University\Year2\MDM2\REP3\pricesukwin01012023.csv")

datatotal1 = data1.merge(data2, how='left', on=['EVENT_NAME','SELECTION_ID','EVENT_ID'])
datatotal2 = data2.merge(data1, how='left', on=['EVENT_NAME','SELECTION_ID','EVENT_ID'])


BSP_x = datatotal1['BSP_x']
BSP_x_prob = 1/BSP_x

BSP_y = datatotal2['BSP_x']
BSP_y_prob = 1/BSP_y

BSP_x_new = BSP_x_prob[:,None]
BSP_y_new = BSP_y_prob[:,None]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(BSP_x_new, BSP_y_new)
#BSP_x is now the win market BSP value, BSP_y is now the place market BSP values 



#this is using cross validation method to find the most suitable variables for hidden_layer_size
parameters = {'activation':('relu', 'logistic'), 'hidden_layer_sizes':((5),(10),(20),(5,5),(10,10),(20,20),(5,5,5),(10,10,10),(20,20,20),(5,5,5,5),(10,10,10,10),(20,20,20,20),(5,5,5,5,5),(10,10,10,10,10),(20,20,20,20,20))}
clf = GridSearchCV(estimator=MLPRegressor(), param_grid=parameters, scoring='r2', cv=5)
clf.fit(Xtrain,Ytrain)

#identifying the best parameters out of the ones given
best_params = clf.best_params_
print('**********')
print('best parameters:',best_params)

#assigning the new best parameters variables so they can be graphed and the R2 value calculated 
newactivation = list(best_params.values())[0]
newhiddenlayersize = list(best_params.values())[1]

#applying these new optimal parameters to the neural network to caluclate the R2 value
neural_network = MLPRegressor(activation=str(newactivation),hidden_layer_sizes=newhiddenlayersize, early_stopping=True).fit(Xtrain,Ytrain) 
ypred = neural_network.predict(Xtest)
R2vals = neural_network.score(Xtest,Ytest)

print('************')
print('ypred = ', ypred)
print('**************')
print('R2val=', R2vals)
print('**************')

#plot of the correlation between the actual y values (x axis) and the predicted y values from the machine learning (y axis)
plt.scatter(Ytest,ypred,c='pink')
plt.show()


#finding the mean squared error also
MSE = mean_squared_error(Ytest,ypred)
print('mean squared error:', MSE)
