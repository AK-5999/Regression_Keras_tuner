#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import tensorflow as tf


# In[5]:


from tensorflow import keras


# In[6]:


from tensorflow.keras import layers


# In[7]:


from kerastuner.tuners import RandomSearch


# In[8]:


df=pd.read_csv('E:\COLLINS\Rajdeep\DeepLearning\ANN\AI_Quality.csv')


# In[9]:


df.head()


# In[13]:


X = df.iloc[:,0:-1]


# In[14]:


X


# In[15]:


Y = df.iloc[:,-1]


# In[16]:


Y


# In[17]:


df.info()


# In[18]:


df.describe()


# In[19]:


import seaborn as sns


# In[20]:


sns.pairplot(data=df, x_vars=["T","TM","Tm","SLP","H","VV","V","VM"], y_vars = "PM 2.5")


# # Hence, given data is problem of linear Regression
# 
# letu's start build a neural network

# In[21]:


def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model


# In[22]:


build_model


# In[23]:


tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='project',
    project_name='Air Quality Index')


# In[24]:


tuner


# In[25]:


tuner.search_space_summary()


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[27]:


X_train


# In[28]:


X_test


# In[30]:


y_train


# In[31]:


y_test


# In[32]:


tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test))


# In[33]:


tuner.results_summary()


# In[ ]:




