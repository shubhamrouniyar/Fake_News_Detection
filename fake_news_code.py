import numpy as np 
import pandas as pd
import nltk
nltk.download('stopwords')

train=pd.DataFrame(pd.read_csv('fake news label data for training.csv'))
print(len(train))
train=train.dropna()
train['total']=train['title']+' '+train['author']+train['text']
target=np.array(train['label'])
print(len(train['total'].values))
print(target[0])

import re
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords

#sent=sent_tokenize(str(total))
dict_={}
def calculate_words(document,label):
  for i in range(len(document[0:1500])):
    token=word_tokenize(document[i])
    #print(token)
    stop_words=set(stopwords.words('english'))
    filter_words=[w for w in token if w not in stop_words]
    #print(filter_words)
    punctuation=['?',',','/','!','$','@','#','(',')','.','&',':','`','',';','^','%','_','-',"'"]
    filtered_words=[w for w in filter_words if w not in punctuation]
    #print(filtered_words)

    for w in filtered_words:
      if w in dict_:
        x=label[i]
        dict_.setdefault(w,[]).append(x)
      if w not in dict_:
        x=label[i]
        dict_.setdefault(w,[]).append(x)
   
  return dict_
     
dict_=calculate_words(train['total'].values,target)
#print(dict_)

no_of_words=len(dict_)
count_dict={}
for key,value in dict_.items():
  count_one=0
  count_zero=0
  for one in value:
    if(one==1):
      count_one=count_one+1
    elif(one==0):
      count_zero=count_zero+1
  count_dict.setdefault(key,[]).append(count_one)
  count_dict.setdefault(key,[]).append(count_zero)
#print(count_dict)

import math
no_of_documents=len(train['total'].values)
dict_tfidf={}
def calculate_tfidf(count_dict):
  for key,value in count_dict.items():
    one=count_dict[key][0]
    zero=count_dict[key][1]
    one_value=math.log((no_of_documents+1)/(one+2))
    zero_value=math.log((no_of_documents+1)/(zero+2))
    dict_tfidf.setdefault(key,[]).append(one_value)
    dict_tfidf.setdefault(key,[]).append(zero_value)
  return dict_tfidf
dict_tfidf=calculate_tfidf(count_dict)
#print(dict_tfidf)

prob_dict={}
def calculate_probability(dict_prob):
  for key,value in dict_prob.items():
    prob_for_one=((dict_prob[key][0])/(dict_prob[key][0]+dict_prob[key][1]))
    prob_for_zero=((dict_prob[key][1])/(dict_prob[key][0]+dict_prob[key][1]))
    prob_dict.setdefault(key,[]).append(prob_for_one)
    prob_dict.setdefault(key,[]).append(prob_for_zero)
  return prob_dict
    
prob_dict=calculate_probability(dict_tfidf)
#print(prob_dict)

#for input first tokenize to get words and check with trained model words if present
#calculate the probability of each word and multiply for 1 as well as zero 
#word with highest probability print the result

article=train['total'].values[500]
print(target[500])
def predict(article,prob_dict):
  token_test=word_tokenize(article)
  stop_words_test=set(stopwords.words('english'))
  filter_words_test=[w for w in token_test if w not in stop_words_test]
  punctuation_test=['?',',','/','!','$','@','#','(',')','.','&',':','`','',';','^','%','_','-',"'",'-','""','"']
  filtered_words_x=[w for w in filter_words_test if w not in punctuation_test]
  test_words={}
  filtered_words_test=list(set(filtered_words_x))
  
  for k in range(len(filtered_words_test)):
    for key,value in prob_dict.items():
      if key==filtered_words_test[k]:
        test_words.setdefault(key,[]).append(prob_dict[filtered_words_test[k]][0])
        test_words.setdefault(key,[]).append(prob_dict[filtered_words_test[k]][1])
        break
  final_prob_one=1
  final_prob_zero=1
  #print(list(test_words.keys()))
  #print(list(test_words.values())[0])
  for key,value in test_words.items():
    final_prob_one*=test_words[key][0]
    final_prob_zero*=test_words[key][1]
  print(final_prob_one,final_prob_zero)
  
  if(final_prob_one>final_prob_zero):
    print("Article is True")
  else:
    print("Article is False or Fake")
    
predict(article,prob_dict)

