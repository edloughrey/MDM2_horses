# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:49:39 2023

@author: rubya
"""

import glob
import os
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

folder_name = 'windatasets'
file_type = 'csv'
seperator =','
windataframe = pd.concat([pd.read_csv(f, sep=seperator) for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True)


folder_name = 'placedatasets'
file_type = 'csv'
seperator =','
placedataframe = pd.concat([pd.read_csv(f, sep=seperator) for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True)


datatotal = windataframe.merge(placedataframe, how='left', on=['EVENT_NAME','SELECTION_ID','EVENT_ID'])
#datatotal2 = placedataframe.merge(windataframe, how='left', on=['EVENT_NAME','SELECTION_ID','EVENT_ID'])


BSP_x = datatotal['BSP_x']
BSP_x_prob = 1/BSP_x


BSP_y = datatotal['BSP_y']
BSP_y_prob = 1/BSP_y


BSP_x_new = BSP_x_prob[:,None]
BSP_y_new = BSP_y_prob[:,None]


Xtrain, Xtest, Ytrain, Ytest = train_test_split(BSP_x_new, BSP_y_new)