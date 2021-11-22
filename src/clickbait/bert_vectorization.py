import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


bert_preprocess = hub.KerasLayer("./bert_en_uncased_preprocess_3")
bert_encoder = hub.KerasLayer("./bert_en_uncased_L-12_H-768_A-12_4")


if __name__ == '__main__':
    
    ### USE THE BELOW CODE TO READ FROM JSON FILE####
    f = open('../../dataset/clickbait_dataset/processed_string10k.json','r')
    data = json.load(f)
    newData = {}
    title = []
    intesity = []
    for i in data:
        title.append(data[i][0])
        intesity.append(data[i][1])
        
        
    text_preprocessed = bert_preprocess(title)
    bert_results = bert_encoder(text_preprocessed)
    
    print('done')
    
    output = bert_results['pooled_output']
    
    for i in range(len(output)):
        newData[i] = []
        newData[i].append(intesity[i])
        newData[i].append(output[i].numpy().tolist())
        print(i)
        
    
    with open('../../dataset/clickbait_dataset/bert_10k.json', 'w') as f:
        json.dump(newData, f)
        
    print(len(output[0]))
    
    
    
    