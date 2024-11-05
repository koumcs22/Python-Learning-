#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:55:00 2024

@author: koushikdev
"""

# Implementation of Tf-IDF Model: 
    
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

# tokenization
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Existing corpus / we can derive from data set & prepare it using  tokenization  
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]


processed_corpus = []

# Text preprocessing
for i in range(len(corpus)):
    review = re.sub('[^a-zA-Z]', ' ', corpus[i])
    review = review.lower() 
    review = review.split() 
    
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    
    processed_corpus.append(review)

#print(processed_corpus)


# TF IDF Model Here
tfidf_vectorizer = TfidfVectorizer()

# Transform the processed corpus using TF-IDF
X = tfidf_vectorizer.fit_transform(processed_corpus).toarray()

# DataFrame for better visualization
#tfidf_df = pd.DataFrame(X, columns=tfidf_vectorizer.get_feature_names_out())





