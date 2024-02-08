#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[92]:


df=pd.read_csv("C:\\Users\\ANKIT\\Desktop\\House Rent Prediction\\hyd_v2.csv.zip")


# In[106]:


plt.scatter(df.property_size, df.rent_amount)


# In[93]:


#filling of unfilled spaces in the data frame
df.balconies.median()
df.balconies=df.balconies.fillna(df.balconies.median())


# In[94]:


df['type_bhk'].isnull().sum().sum()


# In[95]:


df.type_bhk=df.type_bhk.fillna(df.type_bhk.median())


# In[96]:


df['type_bhk'].isnull().sum().sum()


# In[97]:


#depended variable
x=df[['property_size', 'property_age','type_bhk', 'balconies', 'bathroom']]
#independent variable
y=df['rent_amount']


# In[98]:


#splitting of traing set and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[99]:


model=linear_model.LinearRegression()


# In[100]:


model.fit(x_train, y_train)


# In[101]:


#User input
print("WELCOME HOUSE RENT PREDICTER !!!")
print("Enter property area (in sq.feet): ")
area=float(input())
print("\nEnter property age (in years): ")
age=int(input())
print("\nEnter BHK: ")
bhk=int(input())
print("\nEnter number of balconies: ")
balconies=int(input())
print("\nEnter number of bathrooms: ")
bathrooms=int(input())


# In[103]:


rent=model.predict([[area, age, bhk, balconies, bathrooms]])


# In[107]:


print("Rent per month: ", rent)


# In[ ]:





# In[ ]:




