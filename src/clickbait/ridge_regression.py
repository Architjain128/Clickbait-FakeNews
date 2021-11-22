import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# dataset
from sklearn.datasets import load_boston
# scaling and dataset split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# OLS, Ridge
from sklearn.linear_model import LinearRegression, Ridge
# model evaluation
from sklearn.metrics import r2_score, mean_squared_error
# standarization
from sklearn.preprocessing import StandardScaler


with open("./dataset/vectorized_spacy.json") as f:
    data = json.load(f)
    
x = []
y = []
for p in data:
    x.append(data[p][2])
    y.append(data[p][1])    

# do we need any standarization as we have only one feature(vector) of our model?
scaler = StandardScaler()
x = scaler.fit_transform(x)

# using 30% of data for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

#Fit the model 
ridgereg = Ridge(alpha=0.1,normalize=True)
ridgereg.fit(x_train,y_train)
y_pred = ridgereg.predict(x_test)

mse = 0.0
l = len(y_test)
for i in range(l):
    d = y_test[i]-y_pred[i]
    mse += (d*d)    
mse /= l

print(mse)
