
# coding: utf-8

# In[7]:


import urllib.request
import csv
import pandas as pd
import numpy as np

tsvin = pd.read_csv('datafile.tsv', delimiter='\t')

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def tokenize(text):
    tokens = word_tokenize(text)
    stems = []
    for item in tokens: stems.append(PorterStemmer().stem(item))
    return stems

corpus = []
tsvin=pd.DataFrame(np.array(tsvin).reshape(499,2))

#Finding all questions
for i in range(0, 499):
    
    n= str(tsvin[1][i])
    n= "     "+(n)
    n= n[:n.find("http")]
    n= n[:n.find("@")]
    n= n[:n.find("mailto")]
    n= n[:n.find("www")]
    #print(n)
    n1=n[:(n.find("?")+1)]
    n3="".join(reversed(n1))
    n4="".join(reversed(str(n3[:max(n3.find("."),n3.find(";"))])))
    review = re.sub('[^a-zA-Z]', ' ', str(tsvin[1][i]))
    if ((n4.find('http') != -1) or (n4.find('com') != -1)):
       review=" "
    else:
        review=n
    review = n4
    review = review.lower()
    review = review.split()
    #ps = PorterStemmer()
    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    #print(corpus)
questions= pd.DataFrame(corpus)
print(questions)
# word tokenize and stem
text = [" ".join(tokenize(txt.lower())) for txt in corpus]
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(text).todense()

# transform the matrix to a pandas df
matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())

# sum over each document (axis=0)
top_words = matrix.sum(axis=0).sort_values(ascending=False)
tags= nltk.pos_tag(top_words.index)

#convert into dataframe
df1=pd.DataFrame(top_words)
df2=pd.DataFrame(tags,index=top_words.index)
result = pd.concat([df1,df2],axis=1)

#print the data table and saving as csv
print(result)
result.to_csv("nlp.csv", encoding='utf-8', index=False)

#print all the adjectives
print(result.loc[result.iloc[:,2] == 'NN'])

