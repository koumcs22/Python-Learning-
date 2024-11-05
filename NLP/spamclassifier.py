#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:03:47 2024

@author: koushikdev
"""
# Spam Classifier

"""
Spam Classifier
Poblem Statement : 

@author: koushikdev
"""

import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# tokenization 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#print(pd.__version__)

# Load the dataset
df = pd.read_csv('sms+spam+collection/SMSSpamCollection', sep='\t', names=["label", "message"])

# data Cleaning & data Processing 


#print(len(df))

corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # removing stopwors
    review = ' '.join(review)
    corpus.append(review)

# Bag of Words Model here
from sklearn.feature_extraction.text import CountVectorizer



# Ques: how random select max_features value to select most frequently appears words in Vocabulary
vectorizer = CountVectorizer(max_features=2500) # max_feature  use to Control Vocabulary Size

# To get Vocabulary
#Vocabulary = vectorizer.get_feature_names_out()


X = vectorizer.fit_transform(corpus).toarray() # Analyzes the corpus to build the vocabulary of the most frequent words (up to 2500, as specified).transform: Converts each document in the corpus into a numerical feature vector using the BoW representation.

"""
toarray(): The result of fit_transform is a sparse matrix, which is then converted to a dense array using toarray(). A sparse matrix is a more memory-efficient way to store large matrices with many zero entries, but for ease of use, it is sometimes converted to a dense array.

@author: koushikdev
"""

# Here X value with max_feature  data set is my whole data set for use later

# next we divide in categorical values ham => 0 and spam => 1

# creates a new DataFrame with columns representing the unique categories in df['label']. Each category is converted into a binary column:

y=pd.get_dummies(df['label'])
y=y.iloc[:,1].values # here ILock use for converting the DataFrame into a 1-dimensional array.


# train test split  here 
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# Next predict this output accuracy score using lemmatization & Tf-IDF model
# Also read naiv bayas Theorem & Confusion matrix 
















