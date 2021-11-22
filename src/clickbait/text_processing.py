import json
import re
import nltk
from nltk.corpus import stopwords
# from spacy.en import English

# nlp = English()

# nlp = 

lemm = nltk.stem.WordNetLemmatizer()

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

stopword = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.split()
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub('[^a-zA-Z0-9]', " ",re.sub('&lt;|&gt;|&amp;|&quot;'," ",text))
    text = text.replace('  ',' ').lstrip().rstrip()

    # text = text.split()
    # after_stopword = []
    # for w in text:
    #     if w not in stopword:
    #         after_stopword.append(w)
    # text = " ".join(after_stopword)
    text =  nltk.WordPunctTokenizer().tokenize(text)
    for k in range(len(text)):
        text[k] = lemm.lemmatize(text[k])

    # text = " ".join(text)
    return text

if __name__ == '__main__':
    with open('dataset/instances1.jsonl', 'r') as json_file:
        json_list = list(json_file)

    data_clickbait = {}
    count = 0
    for i in range(len(json_list)):
        result = json.loads(json_list[i])
        if len(result['postText'][0]) > 0:
            data_clickbait[result['id']] = []
            data_clickbait[result['id']].append(result['postText'])
            count += 1
        if count == 20000:
            break

    with open('dataset/truth1.jsonl', 'r') as json_file:
        json_list1 = list(json_file)

    for i in range(len(json_list1)):
        result = json.loads(json_list1[i])
        if result['id'] in data_clickbait.keys():
            data_clickbait[result['id']].append(result['truthMean'])

    delete_keys = []
    for i in  data_clickbait.keys():
        data_clickbait[i][0] = clean_text(str(data_clickbait[i][0]))
        if len(data_clickbait[i][0]) == 0:
            delete_keys.append(i)

    for i in delete_keys:
        data_clickbait.pop(i)
    with open("dataset/processed_token20k.json", "w") as outfile:
        json.dump(data_clickbait, outfile)
    # newd = clean_text("...")
    # print(len(newd))
    # print(clean_text("..."))
    # print(clean_text("Apple's iOS 9 'App thinning' feature will give your phone's storage a boost"))