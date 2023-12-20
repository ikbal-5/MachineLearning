#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 23:47:44 2023

@author: ikbalgencarslan
"""

import pandas as pd

data = pd.read_csv("/Users/ikbalgencarslan/Spyder/NLP/gender_classifier.csv",encoding="latin1")
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(axis=0,inplace=True)
data.gender = [ 1 if each == "female" else 0 for each in data.gender]

#cleaning data

import re
first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description) 

description = description.lower() #Büyük harften küçük harfe çevirme 

# stopwords (irrelavent words) gereksiz kelimeler
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords


description = description.split()
#split yerine tokenizer kullanılabilir
#description = nltk.word_tokenizer(description)
# %%
#gereksiz kelimeleri at
description = [word for word in description if not word in set(stopwords.words("english"))]

#lematazation => loved == love kelimeleri köklerine ayırmak
import nltk as nlp
lemma = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description]

description = " ".join(description)

description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",first_description)
    description = description.lower()
    description =nltk.word_tokenize(description)
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
    
#%% bag of words

from sklearn.feature_extraction.text import CountVectorizer
max_features = 5000
count_vectorizer = CountVectorizer(max_features = max_features, stop_words =  "english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() #x
print("En sık kullanılan {} kelimeler: {}".format(max_features, count_vectorizer.get_feature_names_out()))

y = data.iloc[:,0].values
x = sparce_matrix
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.1, random_state=42)


#%% Navie Bayes

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

#prediction
y_pred = nb.predict(x_test)
print("Accuracy:",nb.score(x_test,y_test))


    