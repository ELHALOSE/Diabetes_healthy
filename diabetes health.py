#!/usr/bin/env python
# coding: utf-8

# <img><src>![image-2.png](attachment:image-2.png)</src></img>

# <b>Definition of data sate: </b>Diabetes is a chronic health condition that affects how your body turns food into energy. There are three main types of diabetes: type 1, type 2, and gestational diabetes.
# <hr><b>Goals: </b>we create model classifay human of health  and predict the state of human diabetes by used ML
# <hr>
# <h4>df1:</h4>
# diabetes_binary_health_indicators_BRFSS2021.csv is a clean dataset of 236,378 survey responses to the CDC's BRFSS2021. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is not balanced.
# 
# <h4>df2:</h4>
# diabetes_binary_5050split_health_indicators_BRFSS2021 is a clean dataset of 67,136 survey responses to the CDC's BRFSS2021. It has an equal 50-50 split of respondents with no diabetes and with either prediabetes or diabetes. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is balanced.
# 
# <h4>df3:</h4>
# diabetes_012_health_indicators_BRFSS2021 is a clean dataset of 236,378 survey responses to the CDC's BRFSS2021. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is not balanced.
#     

# <!-- # Import library -->
# <h1 align=center  style=background-color:DodgerBlue> Import library</h1>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import KFold , cross_val_score


from sklearn. linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

import pickle


# In[2]:


df1=pd.read_csv("diabetes_binary_health_indicators_BRFSS2021.csv")
df2=pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2021.csv")
df3=pd.read_csv("diabetes_012_health_indicators_BRFSS2021.csv")


# <!-- # PreProcessing -->
# <h1 align=center  style=background-color:DodgerBlue> PreProcessing</h1>

# In[3]:


# show all columns
pd.set_option('display.max_columns', None) 


# In[4]:
#
#
# display(df1.head(3))
# display(df2.head(3))
# display(df3.head(3))


# In[5]:


df1.info()
print("-"*100)
df2.info()
print("-"*100)
df3.info()


# In[6]:


print(f"dataframe is sum duplicated:-\n df1: {df1.duplicated().sum()} \n df2: {df2.duplicated().sum()} \n df3: {df3.duplicated().sum()}")


# In[7]:


# replace name feature 
df3['Diabetes_012'] = df3['Diabetes_012'].replace({1: 1, 2: 1, 0: 0})
df3.rename(columns={'Diabetes_012': 'Diabetes_binary'}, inplace=True)


# In[8]:


# combine 3 dataset
df = pd.concat([df1, df2, df3], ignore_index=True)


# In[9]:


df.sample(5)


# In[10]:


df.info()


# dataset isnot null value 
# 

# In[11]:


df.describe().T


# In[12]:


def summary(df):
    summary_df = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary_df['missing_value'] = df.isna().sum()
    summary_df['unique'] = df.nunique().values
    summary_df['count'] = df.count().values
    return summary_df

summary(df).style.background_gradient(cmap='Purples')


# In[13]:


# show columns values
for col in df.columns:
    print(df[col].value_counts())


# In[14]:


# convert datatype to int 
df = df.astype(int)


# In[15]:


df.info()


# In[16]:


df.columns


# In[17]:


df=df.drop(["Education","Income","NoDocbcCost","Veggies","Age"],axis=1)


# In[18]:


# #show feature effect on target
# sns.heatmap(data=df,annot=True)


# ## Data ratio

# In[19]:


sns.countplot(x="Diabetes_binary",data=df)


# <!-- Before marge 
# Not_Diabetes: 84.6%
# Diabetes: 15.3%
# 
# 
# After Marge 
# Not_Diabetes: 80%
# Diabetes: 15.3% -->

# In[20]:


# # unimbalance
# from imblearn.over_sampling import RandomOverSampler
# ous=RandomOverSampler(random_state=42)
# x_res,y_res=ous.fit_resample(x,y)


# In[21]:


# # After and Before check imbalance data
# from collections import Counter 
# print('Original dataset shape {}'.format(Counter(y)))

# print('Resampled dataset shape {}'.format(Counter(y_res)))


# <!-- # EDA -->
# <h1 align=center  style=background-color:DodgerBlue> Exploratory Data Analysis</h1>

# In[22]:


df.columns


# In[23]:


# from matplotlib import pyplot as plt

# def f_importances(coef, names, top=-1):
#     imp = coef
#     imp, names = zip(*sorted(list(zip(imp, names))))

#     # Show all features
#     if top == -1:
#         top = len(names)

#     plt.barh(range(top), imp[::-1][0:top], align='center')
#     plt.yticks(range(top), names[::-1][0:top])
#     plt.title('feature importances')
#     plt.show()

# # whatever your features are called
# features_names = ['PhysHlth', 'MentHlth', 'GenHlth', 'Sex', 'Age', 'AnyHealthcare', 'DiffWalk',
#        'HvyAlcoholConsump', 'Fruits', 'PhysActivity','HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',]

# f_importances(abs(dt.feature_importances_), features_names, top=15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <!-- # Model -->
# <h1 align=center  style=background-color:DodgerBlue> Modeling</h1>

# In[24]:


# Split Data
x=df.iloc [:,1:]
y=df[["Diabetes_binary"]]


# In[25]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,train_size=0.8,random_state=42)


# In[26]:


# # standerscaler
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.fit_transform(X_test)


# In[27]:


# Train model
LR=LogisticRegression()
LR.fit(X_train, Y_train)


# In[28]:


# # Feature selection 

# FeatureSelection= SelectFromModel(estimator=LR) # make sure that thismodel is well-defined
# FeatureSelection.fit(X_train,Y_train)
# FeatureSelection.get_support()


# In[ ]:





# ## KNN Model

# In[29]:


# knn=KNeighborsClassifier(n_neighbors=10)
# knn.fit(X_train, Y_train)


# ## Decision Tree Model

# In[30]:

#
# dt=DecisionTreeClassifier(max_depth=5,max_features=7)
# dt.fit(X_train, Y_train)
#

# In[ ]:





# ## SVC Model

# In[31]:


# svc=SVC(kernel='linear',C=1)


# In[32]:


# svc.fit(x,y)


# In[33]:


# # feedback model(accurcy)
# print(classification_report(Y_test, y_pred_svc))


# ## XGBoost Model

# In[34]:


# !pip install xgboost



# In[35]:


# xgb=XGBClassifier( max_depth=4, n_estimators=6, learning_rate=0.4)


# In[36]:


# # Gridsearch
# from sklearn.model_selection import GridSearchCV
# parameter={"n_estimators":np.arange(10,15),"max_depth":np.arange(1,10)}
# GS=GridSearchCV(XGBClassifier(),parameter)
# GS.fit(x,y)


# In[37]:


# xgb.fit(X_train,Y_train)


# ## RandomForect Model

# In[38]:

#
# # Randomforect
# rf = RandomForestClassifier(n_estimators= 18 , max_depth= 5 , max_features= 7)
# rf.fit(X_train,Y_train)
#

# In[39]:

#
# # AUC XGB
# from sklearn.metrics import plot_roc_curve
# plot_roc_curve(xgb,X_test,Y_test)


# # Voting

# In[40]:

#
# clf1=LogisticRegression()
# clf2=XGBClassifier(missing=10 , max_depth=4,  n_estimators=6,learning_rate=0.4)
# clf3= DecisionTreeClassifier(    max_depth= 3 , max_features= 4)
# clf4=KNeighborsClassifier(n_neighbors=10)
# clf5=RandomForestClassifier(n_estimators= 5 , max_depth= 3 , max_features= 3)


# In[41]:
#
#
# v_clf=VotingClassifier(estimators=[('LR',clf1),('xgb',clf2),('DT',clf3),('knn',clf4),('rf',clf5)],voting ="hard")


# In[42]:


# v_clf.fit(X_train , Y_train)
# print (v_clf.score(X_train , Y_train))
# print (v_clf.score(X_test, Y_test))


# In[43]:


# y_pred_vot=v_clf.predict(X_test)


# <!-- # Report -->
# <h1 align=center  style=background-color:DodgerBlue> Report </h1>

# ### LR

# In[44]:


# Y_prediction
y_pred=LR.predict(X_test)
CM=confusion_matrix(Y_test, y_pred)
print(CM)

sns.heatmap(CM,annot=True)


# In[45]:


# feedback LR model
print(classification_report(Y_test, y_pred))


# ### KNN

# In[46]:


# y_pred_knn=knn.predict(X_test)
# print("acuuricy score: ", accuracy_score(y_pred_knn,Y_test))
# print(confusion_matrix(y_pred_knn,Y_test))


# ### DT

# In[47]:

#
# y_pred_dt=dt.predict(X_test)
# print (classification_report(Y_test , y_pred_dt))
#

# ### SVC

# In[48]:


# y_pred_svc=svc.predict(X_test)
# CM=confusion_matrix(Y_test, y_pred_svc)
# print(CM)
# sns.heatmap(CM,annot=True)


# ### XGB

# In[49]:

#
# y_pred_xgb=xgb.predict(X_test)
# print (classification_report(Y_test , y_pred_xgb))


# ### RF

# In[50]:

#
# y_pred_rf=rf.predict(X_test)
# print (classification_report(Y_test , y_pred_rf))


# # Cross Validation

# ### LR

# In[51]:


#cross validation
from sklearn.model_selection import KFold , cross_val_score
K_fold = KFold(n_splits= 3 , shuffle  = True ,random_state= 42)
scoring = "accuracy"
score = cross_val_score(LR ,X_train, Y_train, cv = K_fold , scoring= scoring)
print (score)


# ### RF

# In[52]:


# #cross validation
# from sklearn.model_selection import KFold , cross_val_score
# K_fold = KFold(n_splits= 3 , shuffle  = True ,random_state= 42)
# scoring = "accuracy"
# score = cross_val_score(rf ,X_train, Y_train, cv = K_fold , scoring= scoring)
# print (score)


# ### xgb

# In[53]:

#
# #cross validation
# from sklearn.model_selection import KFold , cross_val_score
# K_fold = KFold(n_splits= 3 , shuffle  = True ,random_state= 42)
# scoring = "accuracy"
# score = cross_val_score(xgb ,X_train, Y_train, cv = K_fold , scoring= scoring)
# print (score)


# ### DT

# In[54]:

#
# #cross validation
# from sklearn.model_selection import KFold , cross_val_score
# K_fold = KFold(n_splits= 3 , shuffle  = True ,random_state= 42)
# scoring = "accuracy"
# score = cross_val_score(dt ,X_train, Y_train, cv = K_fold , scoring= scoring)
# print (score)


# <hr>

# <!-- # Evaluation -->
# <h1 align=center  style=background-color:DodgerBlue> Evaluation</h1>

# In[55]:

#
# evaluation=pd.DataFrame({
#     "Model":['LR',"RF","XGB","DT",],"Accurcy":[accuracy_score(Y_test, y_pred),accuracy_score(Y_test , y_pred_rf),accuracy_score(Y_test , y_pred_xgb),accuracy_score(Y_test , y_pred_dt)]
# })
# evaluation


# <hr>

# <!-- # Test -->
# <h1 align=center  style=background-color:DodgerBlue> Test</h1>


# In[56]:


print(x.head(10))
print("************************************************************************")
print(y.head(10))


# In[57]:


human_1 = [0,1,1,15,1,0,0,0,1,0,1,5,10,20,0,0] # 0
print (LR.predict([human_1]))


human_2 = [0,0,1,24,1,0,0,0,0,0,1,3,0,0,1,1] # 4
print (LR.predict([human_2]))

human_3 = [0,1,1,29,0,1,1,1,1,0,1,5,0,30,1,1] # 3
print (LR.predict([human_3]))


# In[ ]:





# In[58]:


pickle.dump(LR,open("Model.pkl","wb"))


# In[ ]:





# In[ ]:




