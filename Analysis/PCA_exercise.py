#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# # The Data
# Let's work with the cancer data set since it had so many features

# In[4]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()


# In[5]:


cancer


# In[6]:


df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head()


# # PCA Visualization
# As we've noticed before it is difficult to visualize high dimensional data, we can use PCA to find the first two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot. Before we do this though, we'll need to scale our data so that each feature has a single unit variance.
# 

# In[8]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)


# PCA with Scikit Learn uses a very similar process to other preprocessing functions that come with Scikit Learn. We instantiate a PCA object, find the principal components using the fit method, then apply the rotation and dimensionality reduction by calling transform().

# In[9]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)


# Now we can transform this data to its first 2 principal components.

# In[10]:


x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)


# Let's plot these two dimensions out!

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# # Interpreting the components
# Unfortunately, with great power of dimensionality reduction, comes the cost of NOT being able to easily understand what these components represent.
# The components correspond to combinations of the original features, the components thamselves are stored as an attribute of the fitted PCA object:

# In[13]:


pca.components_


# In this matrix array, each row represents a principal component, and each column relates back to the original features. We can visualize this relationship with a heatmap.

# In[15]:


df_comp= pd.DataFrame(pca.components_, columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


# In[ ]:




