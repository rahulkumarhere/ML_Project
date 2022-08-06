#!/usr/bin/env python
# coding: utf-8

# 
**NOTES: * This Jupyter notebook includes both Project Notes I and Project Notes II.**
# In[ ]:





# ## Importing Libraries and Data  set
NOTE - There were few categories in categorical variable where binning was done in excel itself because the same excel file needs to be used in Tableau itself.
Below is the details of binning done:
    Prefered Category - Mobile to Mobile Phone
prefered login device - Mobile to Mobile Phone

Prefered Payment methods - Cash on Delivery to COD
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


churn_df = pd.read_excel('E Commerce Dataset.xlsx', 'E Comm')


# In[3]:


churn_df.head()


# ## EDA

# In[4]:


churn_df.shape


# In[5]:


churn_df.info()


# In[6]:


churn_df.describe().T


# In[7]:


churn_df.columns = churn_df.columns.str.strip().str.replace(' ', '_')
churn_df.head()


# In[8]:


churn_df.columns


# In[ ]:





# ### Univariate and Bivariate analysis

# In[9]:


plt.figure(figsize=(6,5))
churn_df['Churn'].value_counts(normalize=True).plot(kind='bar')
plt.ylabel('Percentage_Count')
plt.xlabel('Churn_Flag')
Index = [0,1]
plt.xticks(Index, ['Not_Churn', 'Churn'])
plt.title('Churn Flag Vs Percentage Count', weight = 'bold')


# In[10]:


sns.distplot(churn_df.Tenure)


# In[11]:


sns.distplot(churn_df.WarehouseToHome)


# In[12]:


sns.distplot(churn_df.HourSpendOnApp)


# In[13]:


sns.distplot(churn_df.NumberOfDeviceRegistered)


# In[14]:


sns.distplot(churn_df.NumberOfAddress)


# In[15]:


sns.distplot(churn_df.Complain)


# In[16]:


sns.distplot(churn_df.OrderAmountHikeFromlastYear)


# In[17]:


sns.distplot(churn_df.CouponUsed)


# In[18]:


sns.distplot(churn_df.OrderCount)


# In[19]:


sns.distplot(churn_df.DaySinceLastOrder)


# In[20]:


sns.distplot(churn_df.CashbackAmount)


# In[21]:


churn_df.CashbackAmount.skew()


# In[22]:


plt.figure(figsize = (12, 7))
sns.heatmap(churn_df.corr(), cbar = True, annot=True, cmap = 'rocket')


# In[23]:


churn_df.drop('CustomerID', axis=1, inplace=True)
churn_df.head()


# In[24]:


plt.figure(figsize = (12, 10))
sns.heatmap(churn_df.corr(), cbar = True, annot=True, cmap = 'rocket')


# In[25]:


# calc_vif(X).sort_values(by = 'VIF', ascending = False)


# In[26]:


churn_df.drop('CouponUsed', axis=1, inplace=True)
churn_df.head()


# ## Missing Value Treatment

# In[27]:


churn_df.dtypes


# In[28]:


churn_df.isnull().sum()


# In[29]:


print('% of null value in the data set: {} '.format((churn_df.isnull().sum().sum())/churn_df.Churn.count()))


# ### Imputing missing values

# In[30]:


median=churn_df['Tenure'].median()
churn_df['Tenure']=churn_df['Tenure'].fillna(median)


# In[31]:


median=churn_df['WarehouseToHome'].median()
churn_df['WarehouseToHome']=churn_df['WarehouseToHome'].fillna(median)


# In[32]:


median=churn_df['HourSpendOnApp'].median()
churn_df['HourSpendOnApp']=churn_df['HourSpendOnApp'].fillna(median)


# In[33]:


median=churn_df['OrderAmountHikeFromlastYear'].median()
churn_df['OrderAmountHikeFromlastYear']=churn_df['OrderAmountHikeFromlastYear'].fillna(median)


# In[34]:


median=churn_df['OrderCount'].median()
churn_df['OrderCount']=churn_df['OrderCount'].fillna(median)


# In[35]:


median=churn_df['DaySinceLastOrder'].median()
churn_df['DaySinceLastOrder']=churn_df['DaySinceLastOrder'].fillna(median)


# In[36]:


churn_df.isnull().sum().sum()


# In[37]:


churn_df.dtypes


# ## Outlier check and Treatment

# In[38]:


plt.figure(figsize=(20,6))
churn_df.iloc[:,1:12].boxplot()


# In[39]:


plt.figure(figsize=(20,6))
churn_df.iloc[:,11:19].boxplot()


# In[40]:


churn_df['Churn'].value_counts(normalize=True)


# In[41]:


def capping_outliers(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[42]:


for i in churn_df.columns:
    if churn_df[i].dtype != 'object' and i != 'Churn':
        lr, ur = capping_outliers(churn_df[i])
        churn_df[i]=np.where(churn_df[i]>ur,ur,churn_df[i])
        churn_df[i]=np.where(churn_df[i]<lr,lr,churn_df[i])


# In[43]:


plt.figure(figsize=(20,6))
churn_df.iloc[:,1:12].boxplot()


# In[44]:


plt.figure(figsize=(20,6))
churn_df.iloc[:,11:19].boxplot()


# ## Variable transformation

# ### Encoding Object type variables

# In[45]:


for i in churn_df.columns:
    if churn_df[i].dtype== 'object':
        print('\n')
        print('Feature: ', i)
        print(pd.Categorical(churn_df[i].unique()))
        print(pd.Categorical(churn_df[i].unique()).codes)
        churn_df[i]=pd.Categorical(churn_df[i]).codes


# In[ ]:





# ### Creating new variable, 'Recency' from 'DaySinceLastOrder'

# In[46]:


churn_df.DaySinceLastOrder.max() # Mximum days since last order is 14.5


# In[47]:


churn_df['Recency'] = 100 - churn_df.DaySinceLastOrder # taking 100 as a constant value to get recency.


# In[48]:


churn_df.head()


# ## Checking Data imbalanced

# In[49]:


churn_df['Churn'].value_counts(normalize = True)


# <span style='color:green'> The data is imbalanced as the target variable 'Churn' has large difference in proportion of both the classes ( 0 and 1, or Not churn and Churn). Thus SMOTE can be used during model building to balance the target classes

# ## Clustering

# In[50]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[51]:


cluster_df = churn_df.drop('Churn', axis = 1)


# In[52]:


wardlink=linkage(cluster_df, method= 'ward')


# In[53]:


# dend=dendrogram(wardlink)


# In[54]:


dend=dendrogram(wardlink, truncate_mode='lastp', p=10)


# In[55]:


from scipy.cluster.hierarchy import fcluster


# In[56]:



clusters = fcluster(wardlink, 3, criterion='maxclust')
clusters


# In[57]:


cluster_df['clusters'] = clusters


# In[58]:


cluster_df.clusters.value_counts().sort_index()


# In[59]:


aggdata=cluster_df[['Tenure','HourSpendOnApp','NumberOfDeviceRegistered','NumberOfAddress','Complain','OrderAmountHikeFromlastYear','OrderCount','Recency','clusters']].groupby('clusters').mean()
aggdata['Freq']=cluster_df['clusters'].value_counts().sort_index()
aggdata.T


# In[60]:


cluster_df


# # <span style = 'color: red'>**Project-Notes II**

# ## Splitting the data

# In[1]:


X=churn_df.drop('Churn', axis=1)
Y=churn_df[['Churn']]


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, stratify = Y, random_state=1)


# ## Scaling the data

# In[64]:


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled=sc.transform(x_test)


# ### Feature Selection

# In[65]:


cor = churn_df.corr()


# In[66]:


#Correlation with output variable
cor_target = abs(cor["Churn"])
cor_target


# In[67]:


#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.05]
relevant_features


# ## Model building and interpretation.

# ## LDA Model

# In[68]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[69]:


LDA_clf=LinearDiscriminantAnalysis()
LDA_model=LDA_clf.fit(x_train_scaled, y_train)
LDA_clf


# ### Train and Test Accuracy

# In[70]:


print('Accuracy for train data is: {}'.format(LDA_model.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(LDA_model.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for train data

# In[71]:


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


# In[72]:


y_predict=LDA_model.predict(x_train_scaled)
sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[73]:


y_predict=LDA_model.predict(x_train_scaled)
print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[74]:


y_predict=LDA_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[75]:


y_predict=LDA_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC Curve for Train and Test data

# In[76]:


probs=LDA_model.predict_proba(x_train_scaled)
probs= probs[: ,1]
lda_train_auc=roc_auc_score(y_train,probs)
print('AUC for train data : {}'.format(lda_train_auc))

train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

plt.plot([0,1], [0,1])
plt.plot(train_fpr, train_tpr)


# In[77]:


probs=LDA_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
lda_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(lda_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### Logistic Model

# In[78]:


from sklearn.linear_model import LogisticRegression


# In[79]:


Logistic_clf=LogisticRegression(random_state=1)
Logistic_model=Logistic_clf.fit(x_train_scaled, y_train)
Logistic_clf


# ### Accuracy for Train and Test data

# In[80]:


print('Accuracy for train data is: {}'.format(Logistic_model.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(Logistic_model.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for train data

# In[81]:


y_predict=Logistic_model.predict(x_train_scaled)
sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[82]:


y_predict=Logistic_model.predict(x_train_scaled)
print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[83]:


y_predict=Logistic_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[84]:


y_predict=Logistic_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for train data

# In[85]:


probs=Logistic_model.predict_proba(x_train_scaled)
probs= probs[: ,1]
logistic_train_auc=roc_auc_score(y_train,probs)
print('AUC for train data : {}'.format(logistic_train_auc))

train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

plt.plot([0,1], [0,1])
plt.plot(train_fpr, train_tpr)


# In[86]:


probs=Logistic_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
logistic_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(logistic_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ## KNN Model

# In[87]:


from sklearn.neighbors import KNeighborsClassifier

Knn_clf=KNeighborsClassifier()
Knn_clf.fit(x_train_scaled,y_train)


# ### Accuracy for Train and Test data

# In[88]:


print('Accuracy for train data is: {}'.format(Knn_clf.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(Knn_clf.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for train data

# In[89]:


y_predict=Knn_clf.predict(x_train_scaled)
sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[90]:


y_predict=Knn_clf.predict(x_train_scaled)
print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[91]:


y_predict=Knn_clf.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[92]:


y_predict=Knn_clf.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for train data

# In[93]:


probs=Knn_clf.predict_proba(x_train_scaled)
probs= probs[: ,1]
knn_train_auc=roc_auc_score(y_train,probs)
print('AUC for train data : {}'.format(knn_train_auc))

train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

plt.plot([0,1], [0,1])
plt.plot(train_fpr, train_tpr)


# In[94]:


probs=Knn_clf.predict_proba(x_test_scaled)
probs= probs[: ,1]
knn_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(knn_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### Naive Bayes model
# 

# In[95]:


from sklearn.naive_bayes import GaussianNB

NB_clf=GaussianNB()
NB_clf.fit(x_train_scaled, y_train)


# ### Accuracy for Train and Test data

# In[96]:


print('Accuracy for train data is: {}'.format(NB_clf.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(NB_clf.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for train data
# 

# In[97]:


y_predict=NB_clf.predict(x_train_scaled)
sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[98]:


y_predict=NB_clf.predict(x_train_scaled)
print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[99]:


y_predict=NB_clf.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[100]:


y_predict=NB_clf.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for Train and Test data

# In[101]:


probs=NB_clf.predict_proba(x_train_scaled)
probs= probs[: ,1]
NB_train_auc=roc_auc_score(y_train,probs)
print('AUC for train data : {}'.format(NB_train_auc))

train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

plt.plot([0,1], [0,1])
plt.plot(train_fpr, train_tpr)


# In[102]:


probs=NB_clf.predict_proba(x_test_scaled)
probs= probs[: ,1]
NB_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(NB_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### SVM model

# In[103]:


from sklearn import svm


# In[104]:


SVM_clf=svm.SVC(random_state=1, probability=True) # Here probability attribute is set to True to enable Probability as its disabled by default
SVM_clf.fit(x_train_scaled, y_train)


# ### Accuracy for Train and Test data

# In[105]:


print('Accuracy for train data is: {}'.format(SVM_clf.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(SVM_clf.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for train data

# In[106]:


y_predict=SVM_clf.predict(x_train_scaled)
sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[107]:


y_predict=SVM_clf.predict(x_train_scaled)
print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[108]:


y_predict=SVM_clf.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[109]:


y_predict=SVM_clf.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for Train and Test data

# In[110]:


probs=SVM_clf.predict_proba(x_train_scaled)
probs= probs[: ,1]
svm_train_auc=roc_auc_score(y_train,probs)
print('AUC for train data : {}'.format(svm_train_auc))

train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

plt.plot([0,1], [0,1])
plt.plot(train_fpr, train_tpr)


# In[111]:


probs=SVM_clf.predict_proba(x_test_scaled)
probs= probs[: ,1]
svm_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(svm_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs, pos_label=1)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# # Model Tuning

# ## 1.Ensembling Modelling

# ### ✔ <span style='color:red'> Here we will implement two different Ensembling techniques, namely, a.) Bagging (Parallel Ensemble) and b.) Boosting (Here model are buid in a serial manner)
#     
# ### ✔<span style='color:red'> Ensemble Technique is used to reduce error in prediction by combining multiple model to get a better output

# In[112]:


import warnings
warnings.filterwarnings("ignore")


# ### a. Bagging (using Random Forest)
# 

# In[113]:



from sklearn.ensemble import RandomForestClassifier

RF_model=RandomForestClassifier(n_estimators=100,random_state=1)
RF_model.fit(x_train_scaled, y_train)


# In[114]:


print('Accuracy for train data is: {}'.format(RF_model.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(RF_model.score(x_test_scaled, y_test)))


# 

# ### Hyper-parameter tuning Random Forest

# In[115]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [6,7,10],
    'max_features': [4, 6],
    'min_samples_leaf': [30, 50, 100],
    'min_samples_split': [50, 70,150, 300],
    'n_estimators': [150,200,350, 500]
}

RF_model = RandomForestClassifier(random_state=1)

grid_search = GridSearchCV(estimator = RF_model, param_grid = param_grid, cv = 5)


# In[116]:


grid_search.fit(x_train_scaled, y_train)
grid_search.best_params_


# In[120]:


best_grid=grid_search.best_estimator_


# In[121]:


print('Accuracy for train data is: {}'.format(best_grid.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(best_grid.score(x_test_scaled, y_test)))


# In[122]:


# Confusion Matrix and Classification report for train data


# In[123]:


# y_predict=best_grid.predict(x_train_scaled)
# sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
#             ,cmap='plasma');
# plt.xlabel('Predicted');
# plt.ylabel('Actuals',rotation=0);


# In[124]:


# # y_predict=best_grid.predict(x_train_scaled)
# print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[125]:


y_predict=best_grid.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[126]:


y_predict=best_grid.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# In[127]:


# AUC and ROC curve for train data


# In[128]:


# probs=best_grid.predict_proba(x_train_scaled)
# probs= probs[: ,1]
# best_grid_train_auc=roc_auc_score(y_train,probs)
# print('AUC for train data : {}'.format(best_grid_train_auc))

# train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

# plt.plot([0,1], [0,1])
# plt.plot(train_fpr, train_tpr)


# ### AUC and ROC curve for test data
# 

# In[129]:


probs=best_grid.predict_proba(x_test_scaled)
probs= probs[: ,1]
best_grid_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(best_grid_test_auc))

train_fpr, train_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(train_fpr, train_tpr)


# ### b. Boosting (XGBoost)

# In[130]:


import xgboost as xgb
XGB_model=xgb.XGBClassifier(random_state=1, learning_rate=0.01)
XGB_model.fit(x_train_scaled, y_train)


# In[131]:


x_test=np.array(x_test)

print('Accuracy for train data is: {}'.format(XGB_model.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(XGB_model.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for test data

# In[132]:


y_predict=XGB_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[133]:


y_predict=XGB_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for test data

# In[134]:


probs=XGB_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
xgb_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(xgb_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### Hyper parameter tuning XGBoost model

# #### ✔ <span style='color:red'>If your learning rate is set too low, training will progress very slowly as you are making very tiny updates to the weights in your network. However, if your learning rate is set too high, it can cause undesirable divergent behavior in your loss function

# In[157]:


param_grid={'learning_rate':[0.00001, 0.0001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]}
XGB_model=xgb.XGBClassifier(random_state=1)
grid_search=GridSearchCV(estimator=XGB_model, param_grid=param_grid, cv=5)


# In[158]:


grid_search.fit(x_train_scaled, y_train)


# In[159]:


grid_search.best_params_


# In[160]:


XGB_model_best=xgb.XGBClassifier(learning_rate=0.5, random_state=1)
XGB_model_best.fit(x_train_scaled, y_train)


# In[161]:


print('Accuracy for train data is: {}'.format(XGB_model_best.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(XGB_model_best.score(x_test_scaled, y_test)))


# In[162]:


#  Confusion Matrix and Classification report for train data


# In[163]:


# y_predict=XGB_model_best.predict(x_train_scaled)
# sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
#             ,cmap='plasma');
# plt.xlabel('Predicted');
# plt.ylabel('Actuals',rotation=0);


# In[164]:


# y_predict=XGB_model_best.predict(x_train_scaled)
# print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[165]:


y_predict=XGB_model_best.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[166]:


y_predict=XGB_model_best.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for test data

# In[167]:


probs=XGB_model_best.predict_proba(x_test_scaled)
probs= probs[: ,1]
logistic_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(logistic_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ## 3.Model Tuning via Hyper parameter

# ### LDA model tuning

# #### ✔<span style='color:red'> 'SVD' is the default solver in LDA model. We will try tuning it along with 'tol' (tolerance rate)

# In[168]:


from sklearn.model_selection import GridSearchCV
param_grid={
    'solver':['svd', 'lsqr', 'eigen'],
    'tol':[0.00001, 0.0001, 0.001, 0.01, 0.1]
}
LDA_clf=LinearDiscriminantAnalysis(shrinkage='auto')
grid_search=GridSearchCV(estimator=LDA_clf, param_grid=param_grid, cv=8)


# In[169]:


grid_search.fit(x_train_scaled, y_train)
grid_search.best_params_


# In[170]:


LDA_clf=LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr', tol = 1e-05)
LDA_model=LDA_clf.fit(x_train_scaled, y_train)
LDA_model


# In[171]:


print('Accuracy for train data is: {}'.format(LDA_model.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(LDA_model.score(x_test_scaled, y_test)))


# In[172]:


#Confusion Matrix and Classification report for train data


# In[173]:


# y_predict=LDA_model.predict(x_train_scaled)
# sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
#             ,cmap='plasma');
# plt.xlabel('Predicted');
# plt.ylabel('Actuals',rotation=0);


# In[174]:


# y_predict=LDA_model.predict(x_train_scaled)
# print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data
# 

# In[175]:


y_predict=LDA_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[176]:


y_predict=LDA_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# In[177]:


# AUC and ROC curve for train data


# In[178]:


# probs=LDA_model.predict_proba(x_train_scaled)
# probs= probs[: ,1]
# lda_train_auc=roc_auc_score(y_train,probs)
# print('AUC for train data : {}'.format(lda_train_auc))

# train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

# plt.plot([0,1], [0,1])
# plt.plot(train_fpr, train_tpr)


# ### AUC and ROC curve for test data

# In[179]:


probs=LDA_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
lda_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(lda_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### Tuninng Logistic model
# 

# In[180]:


param_grid={
    'C':[0.01, 0.1, 1.0, 10, 50, 100, 500, 1000],
    'solver':['lbfgs', 'sag', 'saga', 'newton-cg'],
    'tol':[0.00001, 0.0001, 0.001, 0.01, 0.1]
}
Logistic_clf=LogisticRegression(random_state=1)
grid_search=GridSearchCV(estimator=Logistic_clf, param_grid=param_grid, cv=8)


# In[181]:


grid_search.fit(x_train_scaled, y_train)
grid_search.best_params_


# In[182]:


Logistic_clf=LogisticRegression(random_state=1, C=1.0, solver='sag', tol=0.01)
Logistic_model=Logistic_clf.fit(x_train_scaled, y_train)
Logistic_clf


# In[183]:


print('Accuracy for train data is: {}'.format(Logistic_model.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(Logistic_model.score(x_test_scaled, y_test)))


# In[184]:


# Confusion Matrix and Classification report for train data


# In[186]:


# y_predict=Logistic_model.predict(x_train_scaled)
# sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
#             ,cmap='plasma');
# plt.xlabel('Predicted');
# plt.ylabel('Actuals',rotation=0);


# In[187]:


# y_predict=Logistic_model.predict(x_train_scaled)
# print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[188]:


y_predict=Logistic_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[189]:


y_predict=Logistic_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# In[190]:


# AUC and ROC curve for train data


# In[191]:


# probs=Logistic_model.predict_proba(x_train_scaled)
# probs= probs[: ,1]
# logistic_train_auc=roc_auc_score(y_train,probs)
# print('AUC for train data : {}'.format(logistic_train_auc))

# train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

# plt.plot([0,1], [0,1])
# plt.plot(train_fpr, train_tpr)


# ### AUC and ROC curve for test data

# In[192]:


probs=Logistic_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
logistic_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(logistic_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### TUNING KNN MODEL

# In[193]:


param_grid={
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p':[1,2]
}
Knn_clf=KNeighborsClassifier()
grid_search=GridSearchCV(estimator=Knn_clf, param_grid=param_grid, cv=10)


# In[194]:


grid_search.fit(x_train_scaled, y_train)
grid_search.best_params_


# In[195]:


Knn_clf=KNeighborsClassifier(algorithm='auto', p=1)
Knn_clf.fit(x_train_scaled,y_train)


# In[196]:


print('Accuracy for train data is: {}'.format(Knn_clf.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(Knn_clf.score(x_test_scaled, y_test)))


# In[197]:


# Confusion Matrix and Classification report for train data


# In[198]:


# y_predict=Knn_clf.predict(x_train_scaled)
# sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
#             ,cmap='plasma');
# plt.xlabel('Predicted');
# plt.ylabel('Actuals',rotation=0);


# In[199]:


# y_predict=Knn_clf.predict(x_train_scaled)
# print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[200]:


y_predict=Knn_clf.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[201]:


y_predict=Knn_clf.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# In[202]:



# AUC and ROC curve for train data


# In[203]:


# probs=Knn_clf.predict_proba(x_train_scaled)
# probs= probs[: ,1]
# knn_train_auc=roc_auc_score(y_train,probs)
# print('AUC for train data : {}'.format(knn_train_auc))

# train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

# plt.plot([0,1], [0,1])
# plt.plot(train_fpr, train_tpr)


# 
# ### AUC and ROC curve for test data

# In[204]:


probs=Knn_clf.predict_proba(x_test_scaled)
probs= probs[: ,1]
knn_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(knn_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### TUNING NAIVE BAYES

# In[205]:


param_grid={
    'var_smoothing':[1e-16,1e-18,1e-14,1e-12, 1e-10, 1e-09]
}
NB_clf=GaussianNB()
grid_search=GridSearchCV(estimator=NB_clf, param_grid=param_grid, cv=10)


# In[206]:


grid_search.fit(x_train_scaled, y_train)
grid_search.best_params_


# In[207]:


NB_clf=GaussianNB(var_smoothing=1e-16)
NB_clf.fit(x_train_scaled,y_train)


# In[208]:


print('Accuracy for train data is: {}'.format(NB_clf.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(NB_clf.score(x_test_scaled, y_test)))


# In[209]:


#  Confusion Matrix and Classification report for train data


# In[210]:


# y_predict=NB_clf.predict(x_train_scaled)
# sns.heatmap((confusion_matrix(y_train,y_predict)),annot=True,fmt='.5g'
#             ,cmap='plasma');
# plt.xlabel('Predicted');
# plt.ylabel('Actuals',rotation=0);


# In[211]:


# y_predict=NB_clf.predict(x_train_scaled)
# print(classification_report(y_train, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[212]:


y_predict=NB_clf.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[213]:


y_predict=NB_clf.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# In[ ]:


### AUC and ROC curve for train data


# In[214]:


# probs=NB_clf.predict_proba(x_train_scaled)
# probs= probs[: ,1]
# nb_train_auc=roc_auc_score(y_train,probs)
# print('AUC for train data : {}'.format(nb_train_auc))

# train_fpr, train_tpr, threshold = roc_curve(y_train, probs)

# plt.plot([0,1], [0,1])
# plt.plot(train_fpr, train_tpr)


# ### AUC and ROC curve for test data

# In[215]:


probs=NB_clf.predict_proba(x_test_scaled)
probs= probs[: ,1]
nb_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(nb_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### TUNING SVM MODEL

# In[216]:


param_grid={
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'tol':[0.00001, 0.0001, 0.001, 0.01, 0.1]
}
SVM_tune=svm.SVC(random_state=1, probability=True)
grid_search=GridSearchCV(estimator=SVM_tune, param_grid=param_grid, cv=10)


# In[217]:


grid_search.fit(x_train_scaled, y_train)
grid_search.best_params_

# default kernel value is rbf and default tol value is 1e-03


# #### ✔ <span style = 'color:red'> default kernel value is rbf and default tol value is 1e-03

# In[218]:


SVM_tune=svm.SVC(kernel='poly', tol=1e-05,random_state=1, probability=True)
SVM_tune.fit(x_train_scaled,y_train)


# In[219]:


print('Accuracy for train data is: {}'.format(SVM_tune.score(x_train_scaled, y_train)))
print('Accuracy for test data is: {}'.format(SVM_tune.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for test data

# In[220]:


y_predict=SVM_tune.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[221]:


y_predict=SVM_tune.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for test data

# In[222]:


probs=SVM_tune.predict_proba(x_test_scaled)
probs= probs[: ,1]
svm_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(svm_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ## 4. Model Tuning via SMOTE (Balancing target classes)

# #### ✔<span style = 'color: red'> SMOTE is an Oversampling technique where new minor target class data point are formulated synthetically.
# 
# #### ✔<span style = 'color: red'> This technique is often helpful in case model is not able to learn enough the embedded signal or insights for Minor target class due to comparetively much lesser data for the minor target class. In those cases SMOTE do helps. <span/>

# In[223]:


from imblearn.over_sampling import SMOTE


# In[224]:


x_train=np.asarray(x_train_scaled)
y_train=np.asarray(y_train)


# In[225]:


sm = SMOTE(random_state=1)
x_train_blncd, y_train_blncd = sm.fit_sample(x_train_scaled, y_train.ravel())


# ### Logistic regression with SMOTE

# In[226]:


logistic_blncd_clf=LogisticRegression()
logistic_blncd_model=logistic_blncd_clf.fit(x_train_blncd, y_train_blncd)


# In[227]:


print('Accuracy for train data is: {}'.format(logistic_blncd_model.score(x_train_blncd, y_train_blncd)))
print('Accuracy for test data is: {}'.format(logistic_blncd_model.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for test data

# In[228]:


y_predict=logistic_blncd_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[229]:


y_predict=logistic_blncd_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for test data

# In[230]:


probs=logistic_blncd_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
logistic_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(logistic_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### LDA model with SMOTE

# In[231]:


lda_blncd_clf=LinearDiscriminantAnalysis()
lda_blncd_model=lda_blncd_clf.fit(x_train_blncd, y_train_blncd)


# In[232]:


print('Accuracy for train data is: {}'.format(lda_blncd_model.score(x_train_blncd, y_train_blncd)))
print('Accuracy for test data is: {}'.format(lda_blncd_model.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for test data

# In[233]:


y_predict=lda_blncd_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[234]:


y_predict=lda_blncd_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for test data

# In[235]:


probs=lda_blncd_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
lda_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(lda_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### KNN model with SMOTE

# In[236]:


knn_blncd_clf=KNeighborsClassifier()
knn_blncd_model=knn_blncd_clf.fit(x_train_blncd, y_train_blncd)


# In[237]:


print('Accuracy for train data is: {}'.format(knn_blncd_model.score(x_train_blncd, y_train_blncd)))
print('Accuracy for test data is: {}'.format(knn_blncd_model.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for test data

# In[238]:


y_predict=knn_blncd_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[239]:


y_predict=knn_blncd_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for test data

# In[240]:


probs=knn_blncd_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
knn_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(knn_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### Naive Bayes model with SMOTE

# In[241]:


nb_blncd_clf=GaussianNB()
nb_blncd_model=nb_blncd_clf.fit(x_train_blncd, y_train_blncd)


# In[242]:


print('Accuracy for train data is: {}'.format(nb_blncd_model.score(x_train_blncd, y_train_blncd)))
print('Accuracy for test data is: {}'.format(nb_blncd_model.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for test data

# In[243]:


y_predict=nb_blncd_model.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[244]:


y_predict=nb_blncd_model.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC curve for test data

# In[245]:


probs=nb_blncd_model.predict_proba(x_test_scaled)
probs= probs[: ,1]
nb_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(nb_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ### SVM with SMOTE

# In[246]:


SVM_blncd_clf= svm.SVC(random_state=1, probability=True)
SVM_blncd_clf.fit(x_train_blncd, y_train_blncd)


# In[247]:


print('Accuracy for train data is: {}'.format(SVM_blncd_clf.score(x_train_blncd, y_train_blncd)))
print('Accuracy for test data is: {}'.format(SVM_blncd_clf.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for test data

# In[248]:


y_predict=SVM_blncd_clf.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[249]:


y_predict=SVM_blncd_clf.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# 
# ### AUC and ROC curve for test data

# In[250]:


probs=SVM_blncd_clf.predict_proba(x_test_scaled)
probs= probs[: ,1]
svm_test_auc=roc_auc_score(y_test,probs)
print('AUC for test data : {}'.format(svm_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# ## <span style = 'color: red'> Model Comparison (Selecting most optimum model)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ## Most Optimum Model : SVM with SMOTE applied

# #### ✔ <span style = 'color:red'> from above details, SVM model with SMOTE applied is the best ML model in this case.

# In[251]:


SVM_blncd_clf= svm.SVC(random_state=1, probability=True)
SVM_blncd_clf.fit(x_train_blncd, y_train_blncd)


# In[252]:


print('Accuracy for train data is: {}'.format(SVM_blncd_clf.score(x_train_blncd, y_train_blncd)))
print('Accuracy for test data is: {}'.format(SVM_blncd_clf.score(x_test_scaled, y_test)))


# ### Confusion Matrix and Classification report for train data

# In[253]:


y_predict=SVM_blncd_clf.predict(x_train_blncd)
sns.heatmap((confusion_matrix(y_train_blncd,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[254]:


y_predict=SVM_blncd_clf.predict(x_train_blncd)
print(classification_report(y_train_blncd, y_predict))


# ### Confusion Matrix and Classification report for test data

# In[255]:


y_predict=SVM_blncd_clf.predict(x_test_scaled)
sns.heatmap((confusion_matrix(y_test,y_predict)),annot=True,fmt='.5g'
            ,cmap='plasma');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[256]:


y_predict=SVM_blncd_clf.predict(x_test_scaled)
print(classification_report(y_test, y_predict))


# ### AUC and ROC Curve for Train and Test data

# In[257]:


probs=SVM_blncd_clf.predict_proba(x_train_blncd)
probs= probs[: ,1]
svm_train_auc=roc_auc_score(y_train_blncd,probs)
print('AUC for train data : {}'.format(svm_train_auc))

train_fpr, train_tpr, threshold = roc_curve(y_train_blncd, probs, pos_label=1)

plt.plot([0,1], [0,1])
plt.plot(train_fpr, train_tpr)


# In[258]:


probs=SVM_blncd_clf.predict_proba(x_test_scaled)
probs= probs[: ,1]
svm_test_auc=roc_auc_score(y_test,probs)
print('AUC for train data : {}'.format(svm_test_auc))

test_fpr, test_tpr, threshold = roc_curve(y_test, probs, pos_label=1)

plt.plot([0,1], [0,1])
plt.plot(test_fpr, test_tpr)


# # <span style='color:red'> *Thanks You!* 😊 </span>
# ## <span style = 'color: red'>**~ Rahul Kumar**
