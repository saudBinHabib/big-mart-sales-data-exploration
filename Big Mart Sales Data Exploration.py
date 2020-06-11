#!/usr/bin/env python
# coding: utf-8

# ## importing the packages. 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# In[2]:


# importing the datasets.


# In[3]:


df = pd.read_csv('dataset/train.csv')


# ### Exploring the Training Dataset.

# In[4]:


df.head()


# In[5]:


df.info()


# #### Column item _weight and outlet_size contains null values.

# In[ ]:





# In[6]:


df.describe()


# In[7]:


df.hist(figsize=(20,15))


# ## Let's see corelation between targets and features.

# In[8]:


corr_matrix = df.corr()
corr_matrix['Item_Outlet_Sales']


# ### From the correlation above we can see that the Item_MRP seems to have a good correlation with targeted Item_Outlet_Sales and looks like other columns are not very useful for prediction of target value
# 
# 

# 
# ### So Lets start checking columns relation with Target Item_Outlet_Sales Price

# In[9]:


df.Item_Fat_Content.value_counts()


# ### In the above Value Counts we can see we have two main categories Low Fat and Regular the remaining LF and low fat are belongs to Low Fat and reg belongs to Regular.

# #### So we can replace the wrong categories with the correct one.

# In[10]:


df.Item_Fat_Content = df.Item_Fat_Content.replace('LF','Low Fat')
df.Item_Fat_Content = df.Item_Fat_Content.replace('reg','Regular')
df.Item_Fat_Content = df.Item_Fat_Content.replace('low fat','Low Fat')


# In[11]:


df.Item_Fat_Content.value_counts()


# ### Since ITEM_WEIGHT column correlation strength is very low so we can drop it

# In[12]:


df.Item_Identifier.value_counts()


# ### From above output we can say that ITEM_IDENTIFIER should be categorical columns

# ## For further data processing we need to convert column types into their correct types

# In[13]:


df.Item_Identifier = df.Item_Identifier.astype('category')
df.Item_Fat_Content = df.Item_Fat_Content.astype('category')
df.Item_Type = df.Item_Type.astype('category')
df.Outlet_Identifier = df.Outlet_Identifier.astype('category')
df.Outlet_Establishment_Year = df.Outlet_Establishment_Year.astype('int64')

df.Outlet_Type = df.Outlet_Type.astype('category')
df.Outlet_Location_Type = df.Outlet_Location_Type.astype('category')
df.Outlet_Size = df.Outlet_Size.astype('category')


# ### Now lets explore Item_MRP column. Correlation strength of this column with target column is very high so we need to exploit this column for further infomation about target column

# In[14]:


fig,axes=plt.subplots(1,1,figsize=(12,8))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)


# ### Item_MRP column contain prices which are in clusters so it would be better if we convert this columnn into bins for further processing

# ### so for that lets describe the dataset first to see the clusters.

# In[15]:


df.describe()


# In[16]:


fig,axes=plt.subplots(1,1,figsize=(10,8))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)
plt.plot([69,69],[0,5000])
plt.plot([137,137],[0,9000])
plt.plot([203,203],[0,10000])


# ### We can use these perpendicular lines to divide data into proper bins. So from above graph we got out bin value.

# In[17]:


df['Item_MRP_BIN'] = pd.cut(df.Item_MRP, bins=[25, 69, 137, 203, 270], labels = ['a', 'b', 'c', 'd'], right = True)


# In[18]:


df.head()


# ### Now Explore the other columns

# In[19]:


fig, axes = plt.subplots(3, 1, figsize = (25, 17))
sns.scatterplot(x = 'Item_Visibility', y = 'Item_Outlet_Sales', hue = 'Item_MRP_BIN', ax = axes[0], data = df)
sns.boxplot(x = 'Item_Type', y = 'Item_Outlet_Sales', ax = axes[1], data = df)
sns.boxplot(x = 'Outlet_Identifier', y = 'Item_Outlet_Sales', ax = axes[2], data = df)


# In[20]:


fig, axes = plt.subplots(2, 2, figsize = (20, 16))
sns.boxplot(x = 'Outlet_Establishment_Year', y = 'Item_Outlet_Sales', ax = axes[0, 0], data = df)
sns.boxplot(x = 'Outlet_Size', y = 'Item_Outlet_Sales', ax = axes[0, 1], data = df)
sns.boxplot(x = 'Outlet_Location_Type', y = 'Item_Outlet_Sales', ax = axes[1, 0], data = df)
sns.boxplot(x = 'Outlet_Type', y = 'Item_Outlet_Sales', ax = axes[1, 1], data = df)


# In[ ]:





# ### From above plots we can say that we can drop ITEM_VISIBILiTY along with ITEM_WEIGHT . Further more both of these column have very low correlation strength with target column

# ### From above plots we can say that we can drop ITEM_VISIBILiTY along with ITEM_WEIGHT . Further more both of these column have very low correlation strength with target column

# In[21]:


attributes=['Item_MRP', 'Item_MRP_BIN', 'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Establishment_Year', 'Outlet_Identifier', 'Item_Type', 'Item_Outlet_Sales']


# In[22]:


fig, axes = plt.subplots(2, 2, figsize = (20,16))
sns.boxplot(x ='Outlet_Establishment_Year', y = 'Item_Outlet_Sales', hue = 'Outlet_Size', ax=axes[0,0], data = df)
sns.boxplot(x = 'Outlet_Size',y = 'Item_Outlet_Sales', hue = 'Outlet_Size', ax = axes[0,1], data = df)
sns.boxplot(x = 'Outlet_Location_Type',y = 'Item_Outlet_Sales', hue = 'Outlet_Size', ax = axes[1,0], data = df)
sns.boxplot(x = 'Outlet_Type', y = 'Item_Outlet_Sales', hue = 'Outlet_Size', ax=axes[1,1], data = df)


# ### Exploration of more attributes in the data

# In[23]:


data = df[attributes]


# In[24]:


data.info()


# In[25]:


fig, axes = plt.subplots(1, 1, figsize = (8, 6))

sns.boxplot(y = 'Item_Outlet_Sales', hue = 'Outlet_Type', x = 'Outlet_Location_Type', data = data)


# In[26]:


data[data.Outlet_Size.isnull()]


# ### from the above data we can see that, when OUTLET_TYPE = supermarket type 1 and OUTLET_LOCATION_TYPE is Tier 2 then outlet size is null furthermore when OUTLET_TYPE = Grocery store and OUTLET_LOCATION_TYPE is Tier 3 then outlet size is always null

# In[27]:


data.groupby('Outlet_Type').get_group('Grocery Store')['Outlet_Location_Type'].value_counts()


# In[28]:


data.groupby('Outlet_Type').get_group('Grocery Store')


# In[29]:


data.groupby(['Outlet_Location_Type', 'Outlet_Type'])['Outlet_Size'].value_counts()


# In[30]:


(data.Outlet_Identifier == 'OUT010').value_counts()


# In[31]:


data.groupby('Outlet_Size').Outlet_Identifier.value_counts()


# ### Tier 1 have small and medium size shop. Tier 2 have small and (missing 1) type shop. Tier 3 have 2-medium and 1 high and (missing 2) shop

# ### Tier 2 will have medium size shop in missing 1 and Tier 3 will be high or medium size shop

# In[32]:


data.head()


# In[33]:


def function_replacing_null_Values(x):
    if x == 'OUT010' :
        return 'High'
    elif x == 'OUT045' :
        return 'Medium'
    elif x == 'OUT017' :
        return 'Medium'
    elif x == 'OUT013' :
        return 'High'
    elif x == 'OUT046' :
        return 'Small'
    elif x == 'OUT035' :
        return 'Small'
    elif x == 'OUT019' :
        return 'Small'
    elif x == 'OUT027' :
        return 'Medium'
    elif x == 'OUT049' :
        return 'Medium'
    elif x == 'OUT018' :
        return 'Medium'


# In[34]:


# using Function to fill null values.

data['Outlet_Size'] = data.Outlet_Identifier.apply(function_replacing_null_Values)


# In[35]:


data.head()


# In[36]:


# checking the dataframe, either we have complete dataset or do we have some null values in dataset.

data.info()


# In[37]:


# changing the data type of the Outlet_Size to categorical type.

data.Outlet_Size = data.Outlet_Size.astype('category')


# In[38]:


data.Outlet_Size.unique()


# ## Now lets explore other OUTLIERS for the target column.

# In[39]:


data.head()


# In[40]:


sns.boxplot(x='Item_MRP_BIN',y='Item_Outlet_Sales',data=data)


# In[41]:


data[data.Item_MRP_BIN=='b'].Item_Outlet_Sales.max()


# In[42]:


data[data.Item_Outlet_Sales==7158.6816]


# In[43]:


data=data.drop(index=7796)
data.groupby('Item_MRP_BIN').get_group('b')['Item_Outlet_Sales'].max()


# In[44]:


sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',data=data)


# In[45]:


sns.boxplot(x = 'Outlet_Location_Type', y = 'Item_Outlet_Sales', data = data)


# In[46]:


data[data.Outlet_Location_Type == 'Tier 1'].Item_Outlet_Sales.max()


# In[47]:


data[data['Item_Outlet_Sales'] == 9779.9362]


# In[48]:


data.drop(index=4289, inplace = True)


# In[49]:


sns.boxplot(x='Outlet_Size', y = 'Item_Outlet_Sales', data = data)


# In[50]:


sns.boxplot(x = 'Outlet_Establishment_Year', y = 'Item_Outlet_Sales', data = data)


# In[51]:


# now lets see the data types of the columns

data.info()


# In[52]:


# as we can see from the above table that Outlet_Establishment_Year have data type of int64,
# we can change it to the categorical type as we can see it in the graphical representation above.


data.Outlet_Establishment_Year = data.Outlet_Establishment_Year.astype('category')


# In[ ]:





# ### Now lets remove the Item_MRP column and make the Item_MRP_BIN column to Item_MRP column for better categorical data.

# In[53]:


data = data[['Item_MRP_BIN', 'Outlet_Type', 'Outlet_Location_Type',
       'Outlet_Size', 'Outlet_Establishment_Year', 'Outlet_Identifier',
       'Item_Type', 'Item_Outlet_Sales']]


# In[54]:


data.columns = ['Item_MRP', 'Outlet_Type', 'Outlet_Location_Type',
       'Outlet_Size', 'Outlet_Establishment_Year', 'Outlet_Identifier',
       'Item_Type', 'Item_Outlet_Sales']


# In[55]:


data.head()


# ## Lets create the dataset for training the ML model.

# In[56]:


data_label = data.Item_Outlet_Sales
dataset = pd.get_dummies(data.iloc[:,0:6])


# In[57]:


# adding the labeled output data in the dataset.

dataset['Item_Outlet_Sales'] = data_label


# In[58]:


# checking the shape of the dataset.

dataset.shape


# In[59]:


#checking the top 5 rows of the dataset.

dataset.head(5)


# In[60]:


dataset.to_csv('dataset/dataset_for_ML.csv')


# In[ ]:




