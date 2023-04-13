from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV #use this for cross validation
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

#print(BSP_x) #this is the win market BSP values
#print(BSP_y) #this is the place market BSP values 

#This is setting up the neural network, the input variables that are passed through can be changed for example hidden_layer_sizes(10,10,10,10,)
neural_network = MLPRegressor(activation='logistic',hidden_layer_sizes=(100,100,100,100,100,100,100,100,100), early_stopping=True).fit(Xtrain,Ytrain) #also try relu for activation
ypred = neural_network.predict(Xtest)
R2vals = neural_network.score(Xtest,Ytest)

print('ypred = ', ypred)
print('**************')
print('R2vals=', R2vals)

plt.scatter(Xtest,Ytest,c='blue')
plt.scatter(Ytest,ypred,c='pink')
plt.show()






#this is using cross validation method to find the most suitable variables for hidden_layer_size
parameters = {'activtion':('relu', 'logistic'), 'hidden_layer_sizes':((10),(10,10),(10,10,10),(10,10,10,10),(10,10,10,10,10),(10,10,10,10,10,10))}
clf = GridSearchCV(estimator=neural_network, param_grid=parameters)


#THE ERROR IS HERE!!!!!!
print('********')
y_score = clf.decision_function(Xtrain)
print('y_score=', y_score)
print('********')
fitting = clf.fit(Xtrain,Ytrain)
print('fitting=', fitting)
print('********')
bestparams = clf.get_params(deep=True)
print('bestparams=', bestparams)
print('********')


ypredcv = clf.predict(Xtest)
R2valscv = clf.score(Xtest,Ytest)
print('ypredcv=', ypredcv)
print('********')
print('R2valscv=', R2valscv)


print(sorted(clf.cv_results_.keys())) # this prints out the optimal values for each of the parameters inputted 


#calculating the mean squared error value of the network
error = mean_squared_error(Ytrain, ypred) #not sure what our y_true and y_pred values are?
print(error)