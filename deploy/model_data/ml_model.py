import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn import svm
from imblearn.over_sampling import SMOTE

churn_df = pd.read_excel('E Commerce Dataset.xlsx', 'E Comm')
churn_df.drop('CustomerID', axis=1, inplace=True)

# TREATING MULTICOLLINEARITY:-
churn_df.drop('CouponUsed', axis=1, inplace=True)

# Imputing missing values
median = churn_df['Tenure'].median()
churn_df['Tenure'] = churn_df['Tenure'].fillna(median)
median = churn_df['WarehouseToHome'].median()
churn_df['WarehouseToHome'] = churn_df['WarehouseToHome'].fillna(median)
median = churn_df['HourSpendOnApp'].median()
churn_df['HourSpendOnApp'] = churn_df['HourSpendOnApp'].fillna(median)
median = churn_df['OrderAmountHikeFromlastYear'].median()
churn_df['OrderAmountHikeFromlastYear'] = churn_df['OrderAmountHikeFromlastYear'].fillna(median)
median = churn_df['OrderCount'].median()
churn_df['OrderCount'] = churn_df['OrderCount'].fillna(median)
median = churn_df['DaySinceLastOrder'].median()
churn_df['DaySinceLastOrder'] = churn_df['DaySinceLastOrder'].fillna(median)


# Outlier treatment:-
def capping_outliers(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range
for i in churn_df.columns:
    if churn_df[i].dtype != 'object' and i != 'Churn':
        lr, ur = capping_outliers(churn_df[i])
        churn_df[i]=np.where(churn_df[i]>ur,ur,churn_df[i])
        churn_df[i]=np.where(churn_df[i]<lr,lr,churn_df[i])

# Variable transformation:-
for i in churn_df.columns:
    if churn_df[i].dtype== 'object':
        print('\n')
        print('Feature: ', i)
        print(pd.Categorical(churn_df[i].unique()))
        print(pd.Categorical(churn_df[i].unique()).codes)
        churn_df[i]=pd.Categorical(churn_df[i]).codes

X = churn_df.drop('Churn', axis=1)
print('NULL TOTAL:- ', X.isnull().sum())
Y = churn_df[['Churn']]
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, stratify = Y, random_state=1)

sc=StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# Model training############################################################

# Balancing data using SMOTE
x_train=np.asarray(x_train_scaled)
y_train=np.asarray(y_train)
sm = SMOTE(random_state=1)
x_train_blncd, y_train_blncd = sm.fit_resample(x_train_scaled, y_train.ravel())

SVM_blncd_clf = svm.SVC(random_state=1, probability=True)
SVM_blncd_clf.fit(x_train_blncd, y_train_blncd)

pickle.dump(SVM_blncd_clf, open("ml_model.sav", "wb"))
pickle.dump(sc, open("scalar.sav", "wb"))