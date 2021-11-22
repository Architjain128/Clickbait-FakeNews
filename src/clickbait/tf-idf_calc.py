# this file should be run before, running the word2vec_vectorizer.py file (This file generates the tf-idf for tokens)
import json

with open("./dataset/processed_string20k.json") as f:
    data = json.load(f)
    
idf = {}
out_dict = {}


# calculating idf first
for id in data:
    word_set = set()
    for word in data[id][0]:
        word_set.add(word)
        
    for word in word_set:
        if idf.get(word) == None:
            idf[word] = 0
            
        idf[word] += 1

# calculating tf and also adding idf of it to it
for id in data:
    keyword_set = set()
    out_dict[id] = []
    word_freq = {}
    
    for word in data[id][0]:
        keyword_set.add(word)
        
        if word_freq.get(word) == None:
            word_freq[word] = 0
            
        word_freq[word] += 1
        
    for word in keyword_set:  #stored in word, tf, idf format
        out_dict[id].append((word,word_freq[word],idf[word]))
        

with open('./dataset/20k_tf-idf.json', 'w') as f:
     f.write(json.dumps(out_dict))
