# Clickbait-FakeNews

## How to run streamlit app
```
> cd ./src/streamlit
> pip install -r requirements.txt
> streamlit run main.py
```

## Links
+ DEMO : https://www.youtube.com/watch?v=t7kyNycbeRg
+ Presentation Video:  https://www.youtube.com/watch?v=yRhEFS0AUfU 
+ Github Repo : https://github.com/Architjain128/Clickbait-FakeNews
+ Dataset 1: https://zenodo.org/record/5530410#.YXrqehzhU2w
+ Dataset 2: https://github.com/laxmimerit/fake-real-news-dataset/tree/main/data
+ Previous Repo : https://github.com/Architjain128/Clickbait-FakeNews1
## File structure
+ `documents` : it has all the documents for deliverables 
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
