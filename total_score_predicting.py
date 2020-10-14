#!/usr/bin/env python
# coding: utf-8

# #### Importing the Dataset

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("ipl2017.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# #### Handling Missing Value

# In[5]:


df.isnull().sum()


# #### Dropping Unnecessary Columns

# In[6]:


y=df['total']
x=df.drop(['total','batsman','bowler','mid','date'],axis=1)


# In[7]:


x.head()


# In[8]:


x.dtypes


# In[9]:


x=pd.get_dummies(x,columns=['bat_team','bowl_team','venue'],drop_first=True)
x.head()


# In[10]:


x.shape


# #### Converting Categorical String Columns To Numerical Columns

# #### using One-Hot Encoding

# In[11]:


from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
encoder.fit_transform(df.venue.values.reshape(-1,1)).toarray()
encoder.fit_transform(df.bat_team.values.reshape(-1,1)).toarray()
encoder.fit_transform(df.bowl_team.values.reshape(-1,1)).toarray()
encoder.fit_transform(df.batsman.values.reshape(-1,1)).toarray()
encoder.fit_transform(df.bowler.values.reshape(-1,1)).toarray()


# #### Train_Test_split

# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)


# #### Data Visualisation

# ####       before feature scaling 

# In[13]:


import matplotlib.pyplot as plt
x.iloc[:,0:7].hist(figsize=(16,8))
plt.show()


# In[14]:


import seaborn as sns
sns.pairplot(x.iloc[:,0:7])
plt.show()


# In[15]:


x.iloc[:,0:7].boxplot(figsize=(16,8))
plt.show()


# #### Feature Scaling Using StandardScaler

# In[16]:


type(x_train)


# In[17]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# #### Data Visualisation after Scaling 

# In[18]:


train=pd.DataFrame(x_train,columns=x.columns)
test=pd.DataFrame(x_test,columns=x.columns)


# In[19]:


print("Train data set boxplot")
train.iloc[:,0:7].boxplot(figsize=(16,6))
plt.show()


# In[20]:


print("Test Dataset boxplot")
test.iloc[:,0:7].boxplot(figsize=(16,6))
plt.show()


# In[21]:


print("Train dataset Histogram")
train.iloc[:,0:7].hist(figsize=(16,8))
plt.show()


# In[22]:


print("Test Dataset boxplot")
test.iloc[:,0:7].hist(figsize=(16,8))
plt.show()


# #### Building a model On "Total" columns using RandomForestRegressor

# In[23]:


# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_jobs=-1)
# estimators = np.arange(100,200, 20)
# scores_train = []
# scores_test = []
# for n in estimators:
#     model.set_params(n_estimators=n)
#     model.fit(x_train, y_train)
#     scores_train.append(model.score(x_train, y_train))
#     scores_test.append(model.score(x_test, y_test))
# plt.title("Effect of n_estimators")
# plt.xlabel("n_estimator")
# plt.ylabel("score_train")
# plt.plot(estimators, scores_train)
# plt.title("Effect of n_estimators")
# plt.xlabel("n_estimator")
# plt.ylabel("score_test")
# plt.plot(estimators, scores_test)


# #### above process is time conusuming give good efficiency at n_estimator=140

# In[24]:


from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor( n_estimators=140)
forest.fit(x_train,y_train)
print("Score on Train data:",forest.score(x_train,y_train))
print("Score on Test data :",forest.score(x_test,y_test))


# #### Accuracy_score , Confusion_metrix ,Classification_report

# In[25]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
y_pred=forest.predict(x_test)
y_pred=y_pred.round()
print("accuracy_score : ",accuracy_score(y_test,y_pred))
# print("classification_report : ",classification_report(y_test,y_pred))
print("confusion matrix :")
print(confusion_matrix(y_test,y_pred))


# #### Prediction over_new Dataset

# In[26]:


match={ 'runs':[146,42], 'wickets':[2,2], 'overs':[12.4,6.1],'runs_last_5':[47,39], 'wickets_last_5':[1,2],
       'striker':[52,3], 'non-striker':[6,1],
       'bat_team_Deccan Chargers':[0,0], 'bat_team_Delhi Daredevils':[0,0],
       'bat_team_Gujarat Lions'  :[0,1], 'bat_team_Kings XI Punjab':[1,0],
       'bat_team_Kochi Tuskers Kerala':[0,0], 'bat_team_Kolkata Knight Riders':[0,0],
       'bat_team_Mumbai Indians':[0,1], 'bat_team_Pune Warriors':[0,0],
       'bat_team_Rajasthan Royals':[0,0], 'bat_team_Rising Pune Supergiant':[0,0],
       'bat_team_Rising Pune Supergiants':[0,0],
       'bat_team_Royal Challengers Bangalore':[0,0], 'bat_team_Sunrisers Hyderabad':[0,1],
       'bowl_team_Deccan Chargers':[0,0], 'bowl_team_Delhi Daredevils':[0,0],
       'bowl_team_Gujarat Lions':[0,0], 'bowl_team_Kings XI Punjab':[0,1],
       'bowl_team_Kochi Tuskers Kerala':[0,0], 'bowl_team_Kolkata Knight Riders':[0,0],
       'bowl_team_Mumbai Indians':[1,0], 'bowl_team_Pune Warriors':[0,0],
       'bowl_team_Rajasthan Royals':[0,0], 'bowl_team_Rising Pune Supergiant':[0,0],
       'bowl_team_Rising Pune Supergiants':[0,0],
       'bowl_team_Royal Challengers Bangalore':[0,0],
       'bowl_team_Sunrisers Hyderabad':[0,0],'venue_Brabourne Stadium':[0,0],
       'venue_Buffalo Park':[0,0], 'venue_De Beers Diamond Oval':[0,0],
       'venue_Dr DY Patil Sports Academy':[0,0],
       'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':[0,0],
       'venue_Dubai International Cricket Stadium':[0,0], 'venue_Eden Gardens':[0,1],
       'venue_Feroz Shah Kotla':[0,0], 'venue_Green Park':[0,0],
       'venue_Himachal Pradesh Cricket Association Stadium':[0,0],
       'venue_Holkar Cricket Stadium':[0,0],
       'venue_JSCA International Stadium Complex':[0,0], 'venue_Kingsmead':[0,0],
       'venue_M Chinnaswamy Stadium':[0,0], 'venue_MA Chidambaram Stadium, Chepauk':[0,0],
       'venue_Maharashtra Cricket Association Stadium':[0,0], 'venue_Nehru Stadium':[0,0],
       'venue_New Wanderers Stadium':[0,0], 'venue_Newlands':[0,0],
       'venue_OUTsurance Oval':[0,0,],
       'venue_Punjab Cricket Association IS Bindra Stadium, Mohali':[0,0],
       'venue_Punjab Cricket Association Stadium, Mohali':[0,0],
       'venue_Rajiv Gandhi International Stadium, Uppal':[0,1],
       'venue_Sardar Patel Stadium, Motera':[0,0],
       'venue_Saurashtra Cricket Association Stadium':[0,0],
       'venue_Sawai Mansingh Stadium':[0,0],
       'venue_Shaheed Veer Narayan Singh International Stadium':[0,0],
       'venue_Sharjah Cricket Stadium':[0,0], 'venue_Sheikh Zayed Stadium':[0,0],
       'venue_St Georges Park':[0,0], 'venue_Subrata Roy Sahara Stadium':[0,0],
       'venue_SuperSport Park':[0,0],
       'venue_Vidarbha Cricket Association Stadium, Jamtha':[0,0],
       'venue_Wankhede Stadium':[0,0]}


# In[27]:


new_match=pd.DataFrame(match)


# In[28]:


new_match


# In[29]:


predict_score=forest.predict(new_match)
print("Predicted Score : ",predict_score)


# In[ ]:




