#!/usr/bin/env python
# coding: utf-8

# ## Ontario research project fund predictor 
# 
# The outline of this notebool includes:
# * 1. import the data
# * 2. split the data into train and test set
# * 3. make tf-idf for training dataset and use it for test set 
# * 4. build a logistic regression on training tfidf data 
# * 5. evaluate the model
# 
# First, lets download all the required packages. 
# 

# In[1]:


import os
import sys
import pandas as pd
import numpy as np
import nltk

from pyprojroot import here
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report


# # 1. Import the data
# 
# I imported the data which was cleaned in data_cleaning.ipynb.

# In[2]:


# Read in the data in the previous step
df = pd.read_pickle('..\data\processed\data_clean.pkl')
len(df)


# In[3]:


#df.info()


# # 2. Split the data to train-test set

# Here the data was splited to train and test set. 

# In[4]:


y6=df['two_labela_ontario_commitment']
x=df['project_title']


# In[5]:


def split_data(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    return (x_train,x_test,y_train,y_test)


# In[6]:


x_train,x_test,y_train6,y_test6 =split_data(x,y6)


# In[7]:


len(x_train),len(x_test)


# # 3.1. Make Tf-idf for training
# 
# I used the TfidfVectorizer to convert the words to matrix of TF_IDF features.

# In[8]:


#nltk.download('stopwords')


# In[9]:


tfidfvectorizer = TfidfVectorizer(analyzer='word', lowercase=True, max_df=0.9,min_df=2,ngram_range=(1,1),stop_words='english')


# In[10]:


tfidfvectorizer.fit(x_train)
tfidf_train = tfidfvectorizer.transform(x_train)


# In[11]:


tfidf_test  = tfidfvectorizer.transform(x_test)


# In[12]:


tfidf_train.shape,tfidf_test.shape


# In[13]:


countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
count_wm=countvectorizer.fit_transform(x_train)


# In[14]:


count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()


# In[15]:


tfidf_train.shape, tfidf_test.shape


# In[16]:


len(tfidf_tokens)


# In[17]:


len(set(tfidf_tokens))


# In[18]:


tfidf_tokens[100:]


# In[19]:


#print(tfidf_train) 


# In[20]:


len(list(tfidfvectorizer.vocabulary_.keys())),len(set(list(tfidfvectorizer.vocabulary_.keys())))


# # 4. Model

# # 4.1 Logistic regression on two categories prediction and tf-idf 

# In[21]:


y_train6.shape, tfidf_train.shape


# In[22]:


# logistic regression
logistic_model = LogisticRegression()
logistic_model.fit(tfidf_train, y_train6)


# In[23]:


logistic_model.score(tfidf_train,y_train6)


# In[24]:


logistic_model.score(tfidf_test,y_test6)


# In[25]:


y_pred = logistic_model.predict(tfidf_test)


# In[26]:


accuracy = accuracy_score(y_test6,y_pred)*100
accuracy


# In[27]:


confusion_mat = confusion_matrix(y_test6,y_pred)
confusion_mat


# In[28]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20)}
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid)


# In[29]:


rs_log_reg.fit(tfidf_train, y_train6)


# In[30]:


rs_log_reg.best_params_


# # 5. Evaluate the model

# In[31]:


rs_log_reg.score(tfidf_train, y_train6)


# In[32]:


y_preds6=rs_log_reg.predict(tfidf_test)
print(classification_report(y_test6, y_preds6))

