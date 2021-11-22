import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import joblib
from sklearn.tree import DecisionTreeClassifier


f = open('../../dataset/fakenews_dataset/300dims_vectorized_spacy10k.json','r')
data = json.load(f)
p=[]
wX=[]
woX=[]
X1=[]
X2=[]
X3=[]
Y=[]
ID=[]
i=0
for x in data:
    ID.append(x)
    if(len(data[x])!=4):
        print(x)
    Y.append(data[x][0])
    X1.append([data[x][1]])
    X2.append(data[x][2])
    X3.append(data[x][3])

X1=(np.array(X1))
X2=np.array(X2)
X3=np.array(X3)
wX=np.concatenate((X1,X2,X3),axis=1)
woX=np.concatenate((X2,X3),axis=1)


test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(wX, Y, test_size=test_size, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(multi_class='multinomial', random_state=0,max_iter=1000)

dt = DecisionTreeClassifier()

xgbc_params = {'n_estimators': 1000,'max_depth': 10,'learning_rate': 0.01,'booster': 'dart', 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_lambda': 1}
xgbc = XGBClassifier(objective='reg:squarederror',use_label_encoder=False)

rfc = RandomForestClassifier(n_estimators = 200, n_jobs=-1, min_samples_split=0.001)  

lsvm = SVC(kernel='linear', random_state=0) 

abc = AdaBoostClassifier(base_estimator=None,
                                    n_estimators=100, 
                                    learning_rate=1.0, 
                                    algorithm='SAMME.R', 
                                    random_state=69)
# add more estimators to the ensemble
estimators = [
    ('dt', dt),
    ('xgbc',xgbc),
    ('rfc', rfc),
    ('lsvm', lsvm),
    ('adaboost', abc)
]
reg = StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(),n_jobs=-1)
reg.fit(X_train, Y_train)

print('f1_score:',f1_score(Y_test, reg.predict(X_test_std)))
print('accuracy:',accuracy_score(Y_test, reg.predict(X_test_std)))

# joblib.dump(reg, 'stkwxgbr300_5k_palash.pkl')