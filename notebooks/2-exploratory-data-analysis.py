#!/usr/bin/env python
# coding: utf-8

# ## Exploratory data analysis
# After the data cleaning step to put the data in the right format, it is time to explore the data to find any pattern and trend. 

# In[8]:


import matplotlib.pyplot as plt
import pandas as pd


# In[9]:


# Read in the data in the previous step
df = pd.read_pickle(r'C:\Users\Atieh\Documents\OntarioResearchFund\data\processed\data_clean.pkl')
df.head()


# In[10]:


# sort the data base on date
df = df.sort_values(by="approval_date")
df


# # Graphs

# In[11]:


df.groupby(['year'])['ontario_commitment','total_project_costs'].sum().plot()


# In[57]:



fig,ax=plt.subplots()
df.groupby(['city'])['total_project_costs'].sum().plot()
ax.set_xticks(np.arange(len(df.groupby(['city'])['total_project_costs'].sum().index)))
ax.set_xticklabels(df.groupby(['city'])['total_project_costs'].sum().index,rotation=90)
ax.set_ylabel("Total Project Costs")
plt.show()


# In[54]:


len(df.groupby(['city'])['total_project_costs'].sum().index)


# In[16]:


df['city'].unique()


# In[70]:


fig,ax=plt.subplots()

df.groupby(['lead_research_institution'])['total_project_costs'].sum().plot()
plt.tick_params(which='minor', labelsize=__)
ax.set_xticks(np.arange(len(df.groupby(['lead_research_institution'])['total_project_costs'].sum().index)))
ax.set_xticklabels(df.groupby(['lead_research_institution'])['total_project_costs'].sum().index,rotation=90)

#ax.set_xticklabels(df.lead_research_institution,rotation=90)
ax.set_ylabel("Total Project Costs")
ax.set_xlabel("Lead Research Institution")
plt.show()


# In[66]:


np.arange(len(df.groupby(['lead_research_institution'])['total_project_costs'].sum().index))


# In[73]:


df.groupby(['lead_research_institution'])['total_project_costs'].count().tail(30)


# In[71]:


df.groupby(['area_primary'])['total_project_costs'].sum().plot()


# In[74]:


df.groupby(['area_primary'])['total_project_costs'].count()


# In[75]:


df.groupby(['discipline_primary'])['total_project_costs'].sum().plot()


# In[76]:


df.groupby(['discipline_primary'])['total_project_costs'].count()


# In[77]:


df.groupby(['discipline_primary'])['total_project_costs'].sum()


# In[84]:


df['project_title'][0]


# # Tokens

# In[124]:


tokens=df['project_title'][0].split()


# In[134]:


tokens=[]
for text in df['project_title']:
    tokens.append(list(set(text.split())))


# In[135]:


len(tokens)


# In[138]:


tokens[3130:3134]


# In[164]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[165]:


vectorizer = TfidfVectorizer(min_df=2)


# In[166]:


tfidf = vectorizer.fit_transform(df['project_title'])


# In[167]:


#fit change and fit the model
#transform just return the value of known model
type(tfidf)


# In[168]:


tfidf=tfidf.toarray()
type(tfidf)


# In[169]:


tfidf.shape


# In[170]:


words=vectorizer.get_feature_names()


# In[171]:


words[:20]


# In[181]:


tfidf[:,200:400]


# In[173]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[182]:


plt.figure(figsize=(20,4))
_=plt.imshow(tfidf)


# In[186]:


np.nonzero(tfidf)


# # Group project based on low or high total project costs

# In[192]:


df.total_project_costs.min(),df.total_project_costs.max(),df.total_project_costs.mean()


# In[195]:


df['total_project_costs'] > df.total_project_costs.mean()


# In[198]:


df["y"] = (df['total_project_costs'] > df.total_project_costs.mean()).astype(int)


# # Model

# In[200]:


from sklearn.model_selection import train_test_split


# In[201]:


x=tfidf
y=df['y']


# In[202]:


x_train,x_test,y_train,y_test =train_test_split(x,y)


# In[203]:


len(x_train),len(x_test)


# In[208]:


x_train.shape,y_train.shape


# In[209]:


from sklearn.linear_model import LogisticRegression


# In[211]:


model = LogisticRegression()


# In[212]:


model.fit(x_train,y_train)


# In[213]:


model.predict_proba(x_test)


# In[215]:


y_predict =[int(p[1] >0.5) for p in model.predict_proba(x_test)]


# In[218]:


model.coef_


# In[219]:


model.coef_.shape


# In[220]:


coef = model.coef_.reshape(-1)


# In[222]:


coef.shape


# In[223]:


np.argmax(coef)


# In[227]:


idx = np.argsort(coef)[-10:]


# In[231]:


idx


# In[229]:


words[312]


# In[232]:


words=np.array(words)


# In[233]:


words[idx]


# In[ ]:




