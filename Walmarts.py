#Walmart stores sales

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dt = DecisionTreeRegressor()
rf = RandomForestRegressor(random_state=1)

train = pd.read_csv("C:/Users/Riya/Desktop/walmart/train.csv")
#print(train)

#print(train.isnull().sum())

features = pd.read_csv("C:/Users/Riya/Desktop/walmart/features.csv")
# print(features)

# print(features.isnull().sum())

stores = pd.read_csv("C:/Users/Riya/Desktop/walmart/stores.csv")
# print(stores)

# print(stores.isnull().sum())

# print(features.describe())

features['CPI'].fillna((features['CPI'].mean()), inplace=True)
features['Unemployment'].fillna((features['Unemployment'].median()), inplace=True)
features['MarkDown1'] = features['MarkDown1'].fillna(0)
features['MarkDown2'] = features['MarkDown2'].fillna(0)
features['MarkDown3'] = features['MarkDown3'].fillna(0)
features['MarkDown4'] = features['MarkDown4'].fillna(0)
features['MarkDown5'] = features['MarkDown5'].fillna(0)

# print(features.isnull().sum())

# print(train.describe())

# print(train[train.Weekly_Sales<0])
# print(train[train.Weekly_Sales>=0])

# train.info()
# print(train.dtypes)

train['Date'] = pd.to_datetime(train.Date)
# print(train.dtypes)

train["IsHoliday"] = train["IsHoliday"].astype(int)
# print(train)
# print(train.dtypes)

train['Year']=train['Date'].dt.year
train['Month']=train['Date'].dt.month
train['Week']=train['Date'].dt.isocalendar().week
train['Day']=train['Date'].dt.day
train['n_days']=(train['Date'].dt.date-train['Date'].dt.date.min()).apply(lambda x:x.days)

# print("Holiday")
# print(train[train['IsHoliday']==True]['Weekly_Sales'].describe())
# print("Non-Holiday")
# print(train[train['IsHoliday']==False]['Weekly_Sales'].describe())

# print(features.dtypes)
# features.describe()
features['Date'] = pd.to_datetime(features.Date)
# print(features.dtypes)

features["IsHoliday"] = features["IsHoliday"].astype(int)
# features.dtypes


# print(stores['Type'].value_counts())

stores = stores.merge(features,on='Store',how='left')
# print(stores)

train  = train.merge(stores,on=['Store','Date','IsHoliday'],how='left')
# print(train)

# train.info()

# print("The shape of stores data set is: ", stores.shape)
# print("The unique value of store is: ", stores['Store'].unique())
# print("The unique value of Type is: ", stores['Type'].unique())

train['Is_month_end'] = np.where(train.Day > 22, 1, 0)
train['Is_month_start'] = np.where(train.Day<7,1,0)
train['Is_month_end'] = train['Is_month_end'].astype('bool')
train['Is_month_start'] = train['Is_month_start'].astype('bool')
train['CPI_category'] = pd.cut(train['CPI'],bins=[120,140,160,180,200,220])
train['Unemployment_category'] = pd.cut(train['Unemployment'],bins=[4,6,8,10,12,14,16])
train['fuel_price_category'] = pd.cut(train['Fuel_Price'],bins=[0,2.5,3,3.5,4,4.5])
train['Temperature_category'] = pd.cut(train['Temperature'],bins=[0,20,40,60,80,100])
# train.info()

storetype_values = {'A':3, 'B':2, 'C':1}
train['Type_Numeric'] = train.Type.map(storetype_values)
train = train.drop(['Date', 'Temperature','Fuel_Price', 'Type','Temperature_category','fuel_price_category','Unemployment_category','CPI_category','Is_month_start','Is_month_end','n_days', 'MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Month', 'Day' ], axis=1)

input_cols = train.columns.to_list()
input_cols.remove('Weekly_Sales')
target_col = 'Weekly_Sales'

x = train[input_cols].copy()
y = train[target_col].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print("Decision Tree : " ,dt.score(x_test,y_test))

rf.fit(x_train, y_train)
rf_pred= rf.predict(x_train)
print("Random Forest : " ,rf.score(x_test,y_test))

'''
OUTPUT 
Decision Tree :  0.9690253210969264
Random Forest :  0.9785753931205371

Process finished with exit code 0
'''
