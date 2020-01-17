#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:14:34 2019

@author: Rifat

Problem Statement:
    Predict whether a text tweet contains hate speech or not in this classic case of sentiment analysis

Data Source:
    https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/#data_dictionary
"""

import re
import pandas as pd 
pd.set_option("display.max_colwidth", 300)
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk 
import warnings 
from nltk.stem.porter import *
warnings.filterwarnings("ignore", category=DeprecationWarning)
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim


dataset  = pd.read_csv('train.csv')


## DATA OBSERVATION #######################################################
print (dataset.shape)

#All the hate speech are identified in the column 'level';Condition  for hate speech :label = 1
#Check the missing missing values:
dataset.apply(lambda x: sum(x.isnull()))

#Compute the Number of unique values in each column:
dataset.apply(lambda x: len(x.unique()))

#analyze some hate tweets
print (dataset[dataset['label']== 1].head(10))


## DATA CLEANING #######################################################

#Removing Twitter specialcharecter, Punctuations, Numbers, users
dataset['processed_tweet'] = dataset['tweet'].str.replace("(?i)@user", " ").str.replace("[^a-zA-Z#]", " ").str.replace("\s+", " ")
       
#Removing words where length of the word is less than 4
dataset['processed_tweet'] = dataset['processed_tweet'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>3]))     

#tokenizing the text
token_word = dataset['processed_tweet'].apply(lambda x: x.split()) 

# Stemm the token to get the base word
stemmer = PorterStemmer()
token_word = token_word.apply(lambda x: [stemmer.stem(i) for i in x])

#tokenized_tweet is a list. In the following section we will convert the list word into string words
for i in range (len(token_word)):
    token_word[i] = ' '.join(token_word[i])
dataset['processed_tweet'] = token_word

total_words = ' '.join([tweet for tweet in dataset['processed_tweet']])

#analyze HashTag in the tweet
def collect_ht(tweet):
    hts = []
    for i in tweet:
        ht = re.findall(r"#(\w+)", i)
        hts.append(ht)
    return sum(hts,[])

#find all the non Negetive hashtag
nonNeg_ht = collect_ht(dataset['processed_tweet'][dataset['label'] == 0])
#create a list having the totalcount of all the hashtag
nonNeg_ht_list = nltk.FreqDist(nonNeg_ht)
#convert it into Datframe 
nonNeg_ht_dict = pd.DataFrame({'Hashtag': list(nonNeg_ht_list.keys()),'Count': list(nonNeg_ht_list.values())})


#find all the Negetive hashtag
Neg_ht = collect_ht(dataset['processed_tweet'][dataset['label'] == 1])
#create a list having the totalcount of all the hashtag
Neg_ht_list = nltk.FreqDist(Neg_ht)
#convert it into Datframe 
Neg_ht_dict = pd.DataFrame({'Hashtag': list(Neg_ht_list.keys()),'Count': list(Neg_ht_list.values())})

## Data visualization ###########################################
#identify the most frequent words in the tweet
total_words = ' '.join([tweet for tweet in dataset['processed_tweet']])
wordcloud = WordCloud(width=1500, height=1000, random_state=16, max_font_size=110).generate(total_words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bicubic")
plt.axis('off')
plt.show()


#identify non hate tweet
total_words = ' '.join([tweet for tweet in dataset['processed_tweet'][dataset['label'] == 0]])
wordcloud = WordCloud(width=1500, height=1000, random_state=16, max_font_size=110).generate(total_words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bicubic")
plt.axis('off')
plt.show()


#identify  hate tweet
total_words = ' '.join([tweet for tweet in dataset['processed_tweet'][dataset['label'] == 1]])
wordcloud = WordCloud(width=1500, height=1000, random_state=16, max_font_size=110).generate(total_words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bicubic")
plt.axis('off')
plt.show()


# selecting top 10 most frequent hashtags     
nonNeg_ht_dict = nonNeg_ht_dict.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(20,8))
axis = sns.barplot(data=nonNeg_ht_dict, x= "Hashtag", y = "Count")
axis.set(ylabel = 'Count')
plt.show()


# selecting top 20 most frequent hashtags
Neg_ht_dict = Neg_ht_dict.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=Neg_ht_dict, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


 ## Run models and evaluate ##########################################################
   
#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


#Bag-of-Words Features
vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = vectorizer.fit_transform(dataset['processed_tweet'] )

#TF-IDF Features
tfidf_vector = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tf_idf = tfidf_vector.fit_transform(dataset['processed_tweet'] )

#Bag-of-Words Features
# splitting data into training and test set
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow, dataset['label'], test_size = .25, random_state =20)


# training the model
lregressor = LogisticRegression()
lregressor.fit(X_train_bow, y_train_bow) 

y_pred = lregressor.predict_proba(X_test_bow) 
y_pred_int = y_pred[:,1] >= 0.3 # 
y_pred_int = y_pred_int.astype(np.int)

print ("\nLogistic Regression Model f1_score (Bag-of-Words Features):\n")
f1_score(y_test_bow, y_pred_int, average='micro') 
#print ('f1_score:', f1_score)



##TF-IDF Features
X_train, X_test, y_train, y_test = train_test_split(tf_idf, dataset['label'], test_size = .25, random_state =20)

lregressor.fit(X_train, y_train)

y_pred = lregressor.predict_proba(X_test)
y_pred_int = y_pred[:,1] >= 0.3 
y_pred_int = y_pred_int.astype(np.int)

print ("\nLogistic Regression Model f1_score (TF-IDF Features):\n")
f1_score(y_test, y_pred_int,  average='micro')
#print ('f1_score:', f1_score)

##Support Vector Machine


from sklearn.svm import SVC

#bag of word features
svcregressor = SVC(kernel='linear', probability=True).fit(X_train_bow, y_train_bow)

y_pred = svcregressor.predict_proba(X_test_bow) 
y_pred_int = y_pred[:,1] >= 0.3 
y_pred_int = y_pred_int.astype(np.int)

print ("\nSupport Vector Machine Model f1_score (Bag-of-Words Features):\n")
f1_score(y_test_bow, y_pred_int,  average='micro') 
#print ('f1_score:', f1_score)
#TD-IDF feature
svcregressor = SVC(kernel='linear', probability=True).fit(X_train, y_train)

prediction = svcregressor.predict_proba(X_test)
y_pred_int = y_pred[:,1] >= 0.3 
y_pred_int = y_pred_int.astype(np.int)
print ("\nSupport Vector Machine Regression Model f1_score (TF-IDF Features):\n")
f1_score(y_test, y_pred_int,  average='micro')
#print ('f1_score:', f1_score)

##3. RandomForest
from sklearn.ensemble import RandomForestClassifier

#bag of word feature
randomforest = RandomForestClassifier(n_estimators=400, random_state=11).fit(X_train_bow, y_train_bow)

y_pred = randomforest.predict(X_test_bow)
print ("\nRandomForest Machine Model f1_score (Bag-of-Words Features):\n")
f1_score(y_test_bow, y_pred, average='micro')
#print ('f1_score:', f1_score)

#td-idf
randomforest = RandomForestClassifier(n_estimators=400, random_state=11).fit(X_train, y_train)
print ("\nRandomForest Machine Model f1_score (TF-IDF Features):\n")
y_pred = randomforest.predict(X_test)
result= f1_score(y_test, y_pred, average='micro')
print ('f1_score:', result)



## Findings ###############################################
"""
We first processd the data and create two new features 1.Bag-of-Words Features and 2. TF-IDF Features.
Then We  defined some models  and apply those model to the Bag-of-Words Features and 2. TF-IDF Features.
Then valide the results with the test set using f1_score.In the following we will share the test results of each model and 
see which model works best for this dataset.

1. Logistic Regression Model f1_score (Bag-of-Words Features): 0.9449380553122263
   Logistic Regression Model f1_score (TF-IDF Features): 0.945688900012514 

2. Support Vector Machine Regression Model f1_score (Bag-of-Words Features): 0.0.9471905894130898
   Support Vector Machine Regression Model f1_score (TF-IDF Features):  0.9475660117632336
  
3. RandomForest Machine Model f1_score (Bag-of-Words Features) : 0.9388061569265423
   RandomForest Machine Model f1_score (TF-IDF Features): 0.951820798398198

After analyzing results we find all the algorithm has very good F1-ascore (94%) whcich indicates very good precision and recall.