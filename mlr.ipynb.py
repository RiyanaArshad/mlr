#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


import os
print(os.listdir(r'C:\Users\hp\Downloads'))


# In[9]:


dataset=pd.read_csv(r'C:\Users\hp\Desktop\MLR\data.csv')

# In[13]:


print(dataset.columns)
print(dataset.shape)


# In[10]:


dataset


# In[11]:


X=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]


# In[14]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
X=np.array(ct.fit_transform(X))


# In[15]:


X


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/2)


# In[17]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[18]:


y_pred = regressor.predict(X_test)


# In[19]:


y_pred


# In[20]:


regressor.score(X,y)

