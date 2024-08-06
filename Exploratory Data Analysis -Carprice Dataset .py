#!/usr/bin/env python
# coding: utf-8

# In[56]:


# Importing Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from scipy import stats


# #  Exploratory Data Analysis -Carprice Dataset

# Problem Description
# 
# A Chinese automobile company Teclov_chinese aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. Essentially, the company wants to know:
# 
# • Which variables are significant in predicting the price of a car
# 
# 
# • How well those variables describe the price of a car 
# 
# - Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the American market.

# # Load Data

# In[5]:


df = pd.read_csv(r"D:\carprice.csv")
df.head()


# # Check & Understand the data by info & describing
# 

# In[4]:


df.info()


# In[9]:


df.describe().T


# # Extract Data as per requriment

# In[39]:


x = df.iloc[:,[0,1,2,4,5,6,13,16,18,22,23,24]]
x


# # Check for missing values 
# 

# In[40]:


plt.figure(figsize=(10, 6))

sns.heatmap(x.isnull(), cbar=False, cmap='viridis')

plt.show()


# # Imputing data

# In[41]:


x['fueltype'].fillna(x['fueltype'].mode()[0], inplace=True)
x['doornumber'].fillna(x['doornumber'].mode()[0], inplace=True)
x['stroke'].fillna(x['stroke'].median(), inplace=True)
x['citympg'].fillna(x['citympg'].median(), inplace=True)
x['highwaympg'].fillna(x['highwaympg'].median(), inplace=True)


# In[42]:


x.isnull().sum()


# # Detecting Outliers

# In[45]:


plt.figure(figsize=(10,6))

sns.boxplot(x)

plt.show()


# In[47]:


sns.boxplot(x['stroke'])


# In[52]:


Q1 = x['stroke'].quantile(0.25)

Q3 = x['stroke'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['stroke'] = np.where(x['stroke'] > upper_bound, upper_bound, x['stroke'])

x['stroke'] = np.where(x['stroke'] < lower_bound, lower_bound, x['stroke'])

sns.boxplot(x['stroke'])

plt.title("Identify Outliers in stroke")

plt.show()


# In[48]:


sns.boxplot(x['citympg'])


# In[64]:


Q1 = x['citympg'].quantile(0.25)

Q3 = x['citympg'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['citympg'] = np.where(x['citympg'] > upper_bound, upper_bound, x['citympg'])

x['citympg'] = np.where(x['citympg'] < lower_bound, lower_bound, x['citympg'])

sns.boxplot(x['citympg'])

plt.title("Identify Outliers in citympg")

plt.show()


# In[49]:


sns.boxplot(x['highwaympg'])


# In[65]:


Q1 = x['highwaympg'].quantile(0.25)

Q3 = x['highwaympg'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['highwaympg'] = np.where(x['highwaympg'] > upper_bound, upper_bound, x['highwaympg'])

x['highwaympg'] = np.where(x['highwaympg'] < lower_bound, lower_bound, x['highwaympg'])

sns.boxplot(x['highwaympg'])

plt.title("Identify Outliers in highwaympg")

plt.show()


# In[66]:


sns.boxplot(x['price'])


# In[68]:


Q1 = x['price'].quantile(0.25)

Q3 = x['price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['price'] = np.where(x['price'] > upper_bound, upper_bound, x['price'])

x['price'] = np.where(x['price'] < lower_bound, lower_bound, x['price'])

sns.boxplot(x['price'])

plt.title("Identify Outliers in price")

plt.show()


# In[69]:


plt.figure(figsize=(10,6))

sns.boxplot(x)

plt.show()


# # Examine data insights visually

# Symboling --Its assigned insurance risk rating, A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe. 		
# 
# 

# In[93]:


# Pairplot to visualize relationships between variables

sns.pairplot(x[['stroke', 'citympg', 'highwaympg', 'price']], height=4, aspect=1)


# In[95]:


# Correlation

correlation_matrix = df[["symboling",'stroke','citympg','highwaympg','price']].corr()
correlation_matrix


# In[96]:


# Heatmap

sns.heatmap(correlation_matrix,cmap='coolwarm',linewidths = 0.5, annot=True, square=True )


# In[97]:


# Boxplot to visualize distribution of price by categorical variables

plt.figure(figsize=(10, 6))
sns.boxplot(x='fueltype', y='price', data=x)
sns.boxplot(x='carbody', y='price', data=x)
sns.boxplot(x='drivewheel', y='price', data=x)


# In[98]:


# Scatterplot to visualize relationship between continuous variables and price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='stroke', y='price', data=x)
sns.scatterplot(x='citympg', y='price', data=x)
sns.scatterplot(x='highwaympg', y='price', data=x)


# Conclusion Now that we have understood and gained insight into the dataset ie performed an Exploratory Data Analysis, So let’s summarize what we have learnt in this case study.
# 
# We have extensively covered pre-processing steps required to analyze data We have covered Null value imputation methods We have also covered step by step analyzing techniques such as Bivariate analysis, Multivariate analysis, etc.

# In[ ]:




