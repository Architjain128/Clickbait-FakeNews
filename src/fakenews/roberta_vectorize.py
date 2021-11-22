from transformers import AutoTokenizer, AutoModel
import torch
import json
import pandas as pd

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def giveVector(data):
    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-roberta-large')
    model = AutoModel.from_pretrained('sentence-transformers/nli-roberta-large')

    # Tokenize sentences
    encoded_input = tokenizer(data, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings[0]




# already pre-trained model for predicting clickbait-intensity
# model_path = "stkwxgbr300_20k_0.03390061132923085.pkl"
# model = joblib.load(model_path)

# nlp = spacy.load("en_core_web_lg")

# for lg the dimensions are 300, checked by adding print statement

if __name__ == '__main__':
    
    ### USE THE BELOW CODE TO READ FROM JSON FILE####
    print(giveVector("haha"))
    # f = open('../../dataset/clickbait_dataset/processed_string20k.json','r')
    # data = json.load(f)
    # newData = {}
    # c = 0
    # for i in data:
    #     encoded_input = tokenizer(data[i][0], padding=True, truncation=True, return_tensors='pt')

    #     # Compute token embeddings
    #     with torch.no_grad():
    #         model_output = model(**encoded_input)

    #     # Perform pooling. In this case, max pooling.
    #     sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    #     vectors = sentence_embeddings[0]
    #     newList = []
    #     for j in vectors:
    #         newList.append(float(j))
    #     newData[i] = []
    #     newData[i].append(data[i][1])
    #     newData[i].append(newList)
        
    #     c += 1
    #     print(c)

    #     # print(type(data), type(data[i]))
    #     # break
    # with open('../../dataset/clickbait_dataset/vectorized_bert20k.json', 'w') as f:
    #     json.dump(newData, f)
    
    
    
    # ### USE THE BELOW CODE TO READ FROM CSV FILE ####  (Every Time u process the CSV file make sure to manually write 'id' in the first column of the csv file)
    # data = pd.read_csv("../../dataset/fakenews_dataset/processed_string10k.csv") 
    # newData = {}
    # rows, cols = data.shape
    
    # for i in range(rows):
    #     title_vectors = nlp(data['title'][i])
    #     text_vectors = nlp(data['text'][i])
    #     id = str(data['id'][i].item())
        
    #     title_vec = []
    #     text_vec = []
    #     for j in title_vectors.vector:
    #         title_vec.append(float(j))
            
    #     for j in text_vectors.vector:
    #         text_vec.append(float(j))
            
    #     newData[id] = []
    #     newData[id].append(data['value'][i].item())   #fake or not      
    #     newData[id].append(model.predict([title_vec]).item())   #add the clickbait intensity also
    #     newData[id].append(title_vec)   # vector for clickbait headline
    #     newData[id].append(text_vec)    # vector for text 
        
    #     print(i)
        
    # with open('../../dataset/fakenews_dataset/300dims_vectorized_spacy44k.json', 'w') as f:
    #     json.dump(newData, f)
        
        
    
