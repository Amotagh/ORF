#!/usr/bin/env python
# coding: utf-8

# ## Data cleaning 
# # Introduction
# This notebook go through all necessary steps to clean the data and make it ready for exlanatory data analysis. 
# 1. Import the data
# 2. Explore the data and keep the necessasry columns
# 3. Change the format of object to date and integar

# # 1. Import the data

# In[1]:


#Import libraries
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# Load the data
data = pd.read_csv(r"C:\Users\Atieh\Documents\OntarioResearchFund\data\raw\orf_ri_april_6_2022.csv")
data.T


# # 2. Make a new DataFrame
# I keep below columns for furhter investigarion. 
# 1. project title
# 2. project description
# 3. approval date
# 4. lead research institution
# 5. city
# 6. ontario commitment
# 7. total project cost
# 8. keyword

# In[2]:


df = data [['Project Title','Project Description','Area Primary','Discipline Primary','Approval Date','Lead Research Institution','City','Ontario Commitment','Total Project Costs','Keyword']]


# In[3]:


df.T


# In[4]:


df.info()


# I removed the spaces in the title of each columns.

# In[5]:


df.columns = df.columns.str.replace(' ', '_')


# In[6]:


df.columns


# # 3. Change the object to datetime

# In[7]:


import datetime


# In[8]:


df['Approval_Date'] = pd.to_datetime(df['Approval_Date'])


# In[9]:


df.info()


# In[10]:


df.T


# I added two new columns for the year and month of approval date. 

# In[11]:


df['year'], df['month'] = df['Approval_Date'].dt.year, df['Approval_Date'].dt.month


# In[12]:


df.T


# # 4. Convert obj to integer

# In[13]:


df['Ontario_Commitment']=df['Ontario_Commitment'].str.replace('[\$\,\.]','').astype(int)
df['Total_Project_Costs']=df['Total_Project_Costs'].str.replace('[\$\,\.]','').astype(int)


# In[14]:


df.info()


# In[15]:


df.loc[:,'City']


# # 4. Cleaning the text columns

# I lower cased all the columns' name. 

# In[16]:


df.info()


# In[17]:


df.columns


# In[18]:


df.columns=df.columns.str.lower()


# In[19]:


df.columns


# In[20]:


df.city = df.city.str.strip()
df.lead_research_institution=df.lead_research_institution.str.strip()


# In[21]:


df.info()


# In[22]:


import string
string.punctuation


# I removed all the ponctutation.

# In[23]:


for i in df.columns:
     if df[i].dtype == 'object':
        df[i]=df[i].str.lower()
        df[i]=df[i].str.replace('[^\w\s]','')


# In[24]:


df.T


# In[25]:


df.info()


# In[26]:


# I created a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(df.project_title)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = df.index
data_dtm.head()


# In[27]:


dtmt=data_dtm.transpose()
dtmt


# In[28]:


# Find the top 30 words said by each comedian
top_dict = {}
for c in dtmt.columns:
    top = dtmt[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict


# In[29]:


for comedian, top_words in top_dict.items():
    #print(index)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('---')


# In[30]:


# Look at the most common top words --> add them to the stop word list
from collections import Counter

# Let's first pull out the top 30 words for each comedian
words = []
for i in dtmt.columns:
    top = [word for (word, count) in top_dict[i]]
    for t in top:
        words.append(t)
        
words


# In[31]:


Counter(words).most_common()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


df.to_pickle(r'C:\Users\Atieh\Documents\OntarioResearchFund\data\processed\data_clean.pkl')

