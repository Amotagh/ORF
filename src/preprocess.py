#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd


# In[54]:


def clean_text(project_title):
    project_title={'project_title':[project_title]}
    df=pd.DataFrame(project_title)
    df.project_title=df.project_title.str.lower()
    df.project_title=df.project_title.str.strip()
    df.project_title=df.project_title.str.replace('[^\w\s]','')
    text=df.project_title[0]
    return text
    

