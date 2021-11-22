# Clickbait-FakeNews

## How to run streamlit app
```
> cd ./src/streamlit
> pip install -r requirements.txt
> streamlit run main.py
```
## File structure
+ `documents` : it has all the documents for deliverables 
+ `dataset`
    + `clickbait_dataset` : it has all the dataset used for the regression model used for prediction of clickbait score
    + `fakenews_dataset` : it has all the dataset used for the classification model for labeling the article as fake or not
+ `src`
    + `clickbait` : contains the code for multiple algorithms used for clickbait predication model
        + `models`: all generated models are stored here
    + `fakenews`: contains the code for multiple algorithms used for fake news predication model
        + `models`: all generated models are stored here
    + `streamlit`
        + `requirements.txt`: contains the requirements for the streamlit app
        + `models`: contains final models used for prediction as .pkl file
        + `main.py`: contains the main code for streamlit app
        + `public`: contains media files used in the streamlit app
