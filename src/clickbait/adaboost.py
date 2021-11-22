import json
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

file = open('dataset/vectorized_spacy.json','r')
data = json.load(file)
x_data = []
y_data = []
for i in data.keys():
    x_data.append(data[i][2])
    y_data.append(data[i][1])
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10)


#Choosing Decision Tree with 1 level as the weak learner
DTR = DecisionTreeRegressor(max_depth=1)
RegModel = AdaBoostRegressor(n_estimators=30, base_estimator=DTR ,learning_rate=1)

#Printing all the parameters of Adaboost
print(RegModel)
 
#Creating the model on Training Data
AB = RegModel.fit(x_train,y_train)
y_pred = AB.predict(x_test)

mse = metrics.mean_absolute_error(y_test,y_pred)
print(mse)
