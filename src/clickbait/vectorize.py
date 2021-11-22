import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
from xgboost import XGBRegressor
import spacy
import pandas as pd


# already pre-trained model for predicting clickbait-intensity
model_path = "stkwxgbr300_20k_0.03390061132923085.pkl"
model = joblib.load(model_path)

nlp = spacy.load("en_core_web_lg")

# for lg the dimensions are 300, checked by adding print statement

if __name__ == '__main__':
    
    ### USE THE BELOW CODE TO READ FROM JSON FILE####
    
    # f = open('../../dataset/clickbait_dataset/processed_string20k.json','r')
    # data = json.load(f)
    # newData = {}
    # c = 0
    # for i in data:
    #     vectors = nlp(data[i][0])
    #     newList = []
    #     for j in vectors.vector:
    #         newList.append(float(j))
    #     newData[i] = []
    #     newData[i].append(data[i][1])
    #     newData[i].append(newList)
        
    #     c += 1
    #     print(c)

    #     # print(type(data), type(data[i]))
    #     # break
    # with open('../../dataset/clickbait_dataset/300dims_vectorized_spacy20k.json', 'w') as f:
    #     json.dump(newData, f)
    
    
    
    ### USE THE BELOW CODE TO READ FROM CSV FILE ####  (Every Time u process the CSV file make sure to manually write 'id' in the first column of the csv file)
    data = pd.read_csv("../../dataset/fakenews_dataset/processed44k.csv") 
    newData = {}
    rows, cols = data.shape
    
    for i in range(rows):
        title_vectors = nlp(data['title'][i])
        text_vectors = nlp(data['text'][i])
        id = str(data['id'][i].item())
        
        title_vec = []
        text_vec = []
        for j in title_vectors.vector:
            title_vec.append(float(j))
            
        for j in text_vectors.vector:
            text_vec.append(float(j))
            
        newData[id] = []
        newData[id].append(data['value'][i].item())   #fake or not      
        newData[id].append(model.predict([title_vec]).item())   #add the clickbait intensity also
        newData[id].append(title_vec)   # vector for clickbait headline
        newData[id].append(text_vec)    # vector for text 
        
        print(i)
        
    with open('../../dataset/fakenews_dataset/300dims_vectorized_spacy44k.json', 'w') as f:
        json.dump(newData, f)
        
        
    