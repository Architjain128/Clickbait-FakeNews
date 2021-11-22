# This file should be run after running the tf-idf_calc.py file, because we need the tf-idf values for each token of the doc
import json
import math
import time
import numpy as np
from gensim.models import KeyedVectors

start = time.time()
# load the google word2vec model
filename = './dataset/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# list of all the words available in the model
model_words = list(model.index_to_key)

# known from online source or we can print the size of any vector
dim = 300

# for tf-idf scores
with open("./dataset/20k_tf-idf.json") as f:
    data1 = json.load(f)
    
N = len(data1)  # total number of docs

# for click-bait intensities
with open('./dataset/processed_string20k.json') as f:
    data2 = json.load(f)
    
vector_op = {}
   
c = 0 
for id in data1:
    
    final_vec = []
    for _ in range(dim):
        final_vec.append(0.0)
    
    sz = len(data1[id])
    for i in range(sz):
        word = data1[id][i][0]
        tfidf = (math.log(1+data1[id][i][1]))*(math.log(N/data1[id][i][2]))
        
        # for the words not present in the model, we are ignoring them
        if word in model_words:
            vec = model[word]
            for j in range(dim):
                final_vec[j] += (vec[j]*tfidf)
        
    for i in range(dim):
        final_vec[i] /= sz
    
    vector_op[id] = ((data2[id][1],final_vec))   #clickbait intensity and final vector
    c+=1
    print(c)

with open('./dataset/vectorized_word2vec20k.json', 'w') as f:
     f.write(json.dumps(vector_op))
     
end = time.time()
print(end-start)     
