# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:23:35 2019

@author: rohan
"""


import pandas as pd
data = pd.read_csv('C:\\Users\\rohan\\Downloads\\zips\\spam.csv', encoding = 'latin-1')
print(data.head())

data = data.drop(data.columns[[2, 3, 4]], axis = 1)
data = data.rename(columns = {'v1': 'class', 'v2': 'text'})
data['length'] = data['text'].apply(len)

import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def pre_process(text):
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]    
    words = ""
    for t in text:
        stemmer = SnowballStemmer('english')
        words += (stemmer.stem(t)) + ' '
    return words

text_features = data['text'].copy()
text_features = text_features.apply(pre_process)
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(text_features)
features = features.toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, data['class'], test_size = 0.2, random_state = 111)

from sklearn.svm import SVC
svc = SVC(kernel = 'sigmoid', gamma = 1.0)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)