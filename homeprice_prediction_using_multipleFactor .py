#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


df=pd.read_csv(r"C:\Users\uniqu\Downloads\archive\multi_factor_homeprices.csv")


# In[3]:


df


# In[4]:


import math
df.bedrooms.median()


# In[5]:


df.bedrooms=df.bedrooms.fillna(df.bedrooms.median())
df


# In[7]:


model=linear_model.LinearRegression()
model.fit(df.drop('price', axis='columns'), df.price)


# In[8]:


model.predict([[3000, 3, 40]])


# In[ ]:




