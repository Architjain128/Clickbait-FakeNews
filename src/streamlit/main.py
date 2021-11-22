import streamlit as st
import json
import time
import torch
import joblib
import pickle
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import streamlit.components.v1 as components

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def bert_vector(data):
    '''
        It transforms the input into a bert vector
    '''
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-roberta-large')
    model = AutoModel.from_pretrained('sentence-transformers/nli-roberta-large')
    encoded_input = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return np.array(sentence_embeddings[0])

def concatenate(cb,title,text):
    '''
        It returns 2 concatenated vectors first one with clickbait score and second one with only title and text.
    '''
    wX=[]
    woX=[]
    X1=[]
    X2=[]
    X3=[]
    X1.append(cb)
    X2.append(title)
    X3.append(text)
    X1=(np.array(X1))
    X2=np.array(X2)
    X3=np.array(X3)
    wX=np.concatenate((X1,X2,X3),axis=1)
    woX=np.concatenate((X2,X3),axis=1)
    return wX,woX

def calculate_cb_score(title_vector):
    '''
        It calculates the clickbait score for the given title
    '''
    return model_predicting_cb.predict(np.array(title_vector).reshape(1,-1))

def calculate_true_val_with_cb(vector_with_cb):
    '''
        It calculates the true value for the given cbscore, title and text.
        It return "Fake"/"Real"
    '''
    val = model_predicting_fakeness_with_cbscore.predict(np.array(vector_with_cb).reshape(1,-1))
    if val[0]==0:
        return "Fake"
    else:
        return "Real"

def calculate_true_val_without_cb(vector_without_cb):
    '''
        It calculates the true value for the given title and text.
        It return "Fake"/"Real".
    '''
    val = model_predicting_fakeness_without_cbscore.predict(np.array(vector_without_cb).reshape(1,-1))
    if val[0]==0:
        return "Fake"
    else:
        return "Real"

def load_model(file):
    '''
        Loads the model from the given file
    '''
    return joblib.load(file)

def time_to_calculate(title,text):
    '''
        It calculates all
    '''
    title_vector=bert_vector(title)
    text_vector=bert_vector(text)
    cb_score = calculate_cb_score(title_vector)
    vector_with_cb,vector_without_cb=concatenate(cb_score,title_vector,text_vector)
    true_val_without_cb=calculate_true_val_without_cb(vector_without_cb)
    true_val_with_cb=calculate_true_val_with_cb(vector_with_cb)
    return cb_score[0],true_val_without_cb,true_val_with_cb

model_predicting_cb=load_model("./models/stkRobert_20k.pkl")
model_predicting_fakeness_with_cbscore=load_model("./models/lr_44k_w.pkl")
model_predicting_fakeness_without_cbscore=load_model("./models/lr_44k_wo.pkl")

def no_page():
    '''
        404 page
    '''
    st.error("### Oops! 404")

def explore_page():
    '''
        Explore page
    '''
    st.write("""## Model Overview """)
    st.image("./public/ire-flow-diagram.png",caption="Workflow of Architecture",use_column_width=True)
    st.write("""
            ## Brief Description
            + We have used a pretrained RoBERTa model to extract the sentence embeddings from the title and text of the article.
            + For clickbait score prediction we have used a stacking model consisting of gradient boosting, ridge regression, random forest, adaboost, extreme gradient boosting as estimators and linear regression as final estimator which has mean squared error of 0.0304.
            + For fakeness prediction we have used a Logistic Regression which has f1 score of 0.9974 with clickbait score as a feature else it has f1 score of 0.9784.
            ## Links
            + Project GitHub Repo
                https://github.com/Architjain128/Clickbait-FakeNews 
            + Dataset used for regression
                https://zenodo.org/record/5530410#.YXrqehzhU2w
            + Dataset used for classification
                https://github.com/laxmimerit/fake-real-news-dataset/tree/main/data
    """)

def show_prediction_page():
    '''
        Prediction page
    '''
    st.write("## User Inputs")
    title= st.text_input("Input your article's title here: ")
    text= st.text_area("Input your article's text here: ")
    run = st.button("Predict")
    if run:
        title=title.strip()
        text=text.strip()
        val1=False
        val2=False
        st.markdown("""---""")
        with st.spinner("Generating scores..."):
            if title and text:
                cb_score,true_val_without_cb,true_val_with_cb=time_to_calculate(title,text)
                st.write('## Predictions')
                st.write("+ Clickbait Score is **{:.02f}/100**".format(cb_score*100))
                st.write("+ This is a **"+ str(true_val_with_cb)+"** news considering Clickbait score")
                st.write("+ This is a **"+ str(true_val_without_cb)+"** news without considering Clickbait score")
            else:
                st.error("Empty input fields")

st.sidebar.header('Navigation')
page = st.sidebar.selectbox("Select the page you want to see", ["Predict","Explore"])
st.sidebar.markdown("---")

st.sidebar.markdown("""
                    + Clickbait Score has mean squared error of 3.04%
                    + Accuracy of classification with clickbait score is of 99.7%.
                    + Accuracy of classification without clickbait score is of 97.93%.
""")

st.title("ClickBait and Fakeness Detector")
st.write("This app predicts the clickbait intensity of the news article and tells whether the news is fake or not.")
if page == "Explore":
    explore_page()
elif page == "Predict":
    show_prediction_page()
else:
    no_page()
