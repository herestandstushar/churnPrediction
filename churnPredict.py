#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,  StandardScaler


# In[2]:


ds = pd.read_csv('BankChurners.csv')


# In[3]:


ds.head()


# In[4]:


ds.isnull().sum()


# In[5]:


ds.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],axis= 1, inplace =True)


# In[6]:


ds.head()


# In[7]:


fig = plt.figure(figsize = (15, 15))
sns.heatmap(ds.corr(), annot= True)
plt.show()


# In[8]:


ds.corr()


# In[9]:


ds.head()


# In[10]:


arr = ['Attrition_Flag','Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
ds['Income_Category'].unique()


# In[11]:


Encoder= LabelEncoder()
for i in arr:
    ds[i] = Encoder.fit_transform(ds[i])


# In[12]:


ds.head()


# In[13]:


ds.drop(['CLIENTNUM'],axis=1,inplace=True)


# In[14]:


ds.head()


# In[15]:


plt.scatter(ds['Attrition_Flag'], ds['Avg_Utilization_Ratio'])


# In[16]:


x=ds.drop(["Attrition_Flag"], axis=1)
y=ds['Attrition_Flag']


# In[17]:


model = LogisticRegression()


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)


# In[19]:


model.fit(x_train,y_train)


# In[20]:


model.score(x_test,y_test)


# In[21]:


y_pred = model.predict(x_test)


# In[22]:


from sklearn.metrics import accuracy_score


# In[23]:


print(accuracy_score(y_test,y_pred)*100 ,'%')


# In[24]:


from sklearn import svm


# In[25]:


model2 = svm.SVC()


# In[26]:


model2.fit(x_train,y_train)


# In[27]:


y_pred2= model2.predict(x_test)


# In[28]:


print(accuracy_score(y_test,y_pred2)*100, '%')


# In[29]:


from sklearn import tree
model3=tree.DecisionTreeClassifier()


# In[30]:


model3.fit(x_train,y_train)


# In[34]:


y_pred3 = model3.predict(x_test)
print(accuracy_score(y_test,y_pred3)*100,'%')


# In[32]:


from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(max_depth = 2, random_state =0)


# In[33]:


model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)  


# In[35]:


print(accuracy_score(y_test,y_pred4)*100, '%')


# In[ ]:




