import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 10)


# read csv file
df = pd.read_csv("C:\\Users\\82109\\OneDrive\\문서\\software\\3-1\\data science\\Termproject\\Crime.csv")
'''
# missing value 확인
print(df.isna().sum())
print()
'''
# Dispatch Date/Time의 missing value를 같은 row의 Start_Date_Time으로
# End_Date_Time은 같은 row의 Start_Date_Time으로
df['Dispatch Date / Time'] = np.where(pd.notnull(df['Dispatch Date / Time']) == True, df['Dispatch Date / Time'], df['Start_Date_Time'])
df['Crime Name1'].fillna(method='ffill', inplace=True)
df['Crime Name2'].fillna(method='ffill', inplace=True)
df['Crime Name3'].fillna(method='ffill', inplace=True)
df['End_Date_Time'] = np.where(pd.notnull(df['End_Date_Time']) == True, df['End_Date_Time'], df['Start_Date_Time'])


pt = df[['Place','Start_Date_Time']]

# - 를 기준으로 place를 place와 place_detail로 split
ptd = pt.copy()
ptd[['Place','Place_Detail']] = pt['Place'].str.split(' - ', n=1, expand=True)

# : 를 기준으로 start_date_time을 start_date, start_time, start_ampm으로 split
ptd[['Start_Date','Start_Time','Start_AMPM']] = pt['Start_Date_Time'].str.split(' ', n=2, expand=True)

# /를 기준으로 start_date를 start_day, start_month, start_year로 spilt
ptdd = ptd.copy()
ptdd[['Start_Month','Start_Day','Start_Year']] = ptd['Start_Date'].str.split('/', n=2, expand=True)
ptdd[['Start_Hour','Minute','Second']] = ptd['Start_Time'].str.split(':', n=2, expand=True)

# place_detail의 missing value를 같은 row의 place로
ptdd['Place_Detail'] = np.where(pd.notnull(ptdd['Place_Detail']) == True, ptdd['Place_Detail'], ptdd['Place'])

# feature engineering
ptdd.drop(['Place','Start_Date_Time','Start_Date','Start_Time','Start_Year','Start_Day','Minute','Second'], axis=1, inplace=True)

# dawn : 2~7
# morning : 7~12
# day : 12~18
# evening : 18~22
# night : 22~2
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='12')&(ptdd['Start_AMPM']=='AM'), 'night', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='01')&(ptdd['Start_AMPM']=='AM'), 'night', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='02')&(ptdd['Start_AMPM']=='AM'), 'dawn', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='03')&(ptdd['Start_AMPM']=='AM'), 'dawn', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='04')&(ptdd['Start_AMPM']=='AM'), 'dawn', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='05')&(ptdd['Start_AMPM']=='AM'), 'dawn', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='06')&(ptdd['Start_AMPM']=='AM'), 'dawn', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='07')&(ptdd['Start_AMPM']=='AM'), 'morning', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='08')&(ptdd['Start_AMPM']=='AM'), 'morning', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='09')&(ptdd['Start_AMPM']=='AM'), 'morning', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='10')&(ptdd['Start_AMPM']=='AM'), 'morning', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='11')&(ptdd['Start_AMPM']=='AM'), 'morning', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='12')&(ptdd['Start_AMPM']=='PM'), 'day', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='01')&(ptdd['Start_AMPM']=='PM'), 'day', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='02')&(ptdd['Start_AMPM']=='PM'), 'day', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='03')&(ptdd['Start_AMPM']=='PM'), 'day', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='04')&(ptdd['Start_AMPM']=='PM'), 'day', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='05')&(ptdd['Start_AMPM']=='PM'), 'day', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='06')&(ptdd['Start_AMPM']=='PM'), 'evening', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='07')&(ptdd['Start_AMPM']=='PM'), 'evening', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='08')&(ptdd['Start_AMPM']=='PM'), 'evening', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='09')&(ptdd['Start_AMPM']=='PM'), 'evening', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='10')&(ptdd['Start_AMPM']=='PM'), 'night', ptdd['Start_Hour'])
ptdd['Start_Hour'] = np.where((ptdd['Start_Hour']=='11')&(ptdd['Start_AMPM']=='PM'), 'night', ptdd['Start_Hour'])

### First dataset
# Place_Detail, Start_Hour, Start_Month, Police District data
# Target : Place_Detail
ptdd.drop(['Start_AMPM'], axis=1, inplace=True)
ptdd['Police District Number'] = df['Police District Number']

### Second dataset
# Crime Name1, Start_Hour, State data
# Target : Crime Name1
css = ptdd.copy()
css['Crime Name1'] = df['Crime Name1']
css['State'] = df['State']
css.drop(['Start_Month','Place_Detail','Police District Number'], axis=1, inplace=True)
css.dropna(axis=0, inplace=True)

### Third dataset







# Label Encoding
def label(col):
    encoder = preprocessing.LabelEncoder()
    col = encoder.fit_transform(col)
    return col

# Ordinal Encoding
def ordinal(col):
    encoder = preprocessing.OrdinalEncoder()
    col = encoder.fit_transform(col)
    return col

# Standard Scaling
def standard(train, test):
    scaler = preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    return train, test

# MinMax Scaling
def minmax(train, test):
    scaler = preprocessing.MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    return train, test

# Robust Scaling
def robust(train, test):
    scaler = preprocessing.RobustScaler()
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    return train, test


# K-Nearest Neighbors
def knn(x_train, x_test, y_train, y_test, n):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train, y_train)
    predict = knn.predict(x_test)
    score = knn.score(x_test, y_test)
    print(score)
    print()


# KFold evaluation
def kfold(n, split, x, y):
    knn = KNeighborsClassifier(n_neighbors=n)
    kf = KFold(n_splits=split, shuffle=True)

    for train_index, test_index in kf.split(x) :
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        scaler = preprocessing.StandardScaler()
        x_train_scale = scaler.fit_transform(x_train)
        x_test_scale = scaler.fit_transform(x_test)

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        knn.fit(x_train, y_train)
        prediction = knn.predict(x_test)
        print(prediction)

    score = cross_val_score(knn,x,y,cv=5)
    print('scores mean : {}'.format(np.mean(score)))
    print()


# Gradient Boosting
def boosting(x_train, x_test, y_train, y_test):
    regressor = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1)
    regressor.fit(x_train, y_train)
    predict = regressor.predict(x_test)
    print(predict)
    score = regressor.score(x_test, y_test)
    print(score)
    print()
    
  



### First dataset

## Label encoding
ptdd['Place_Detail'] = label(ptdd['Place_Detail'])
ptdd['Start_Hour'] = label(ptdd['Start_Hour'])
ptdd['Police District Number'] = label(ptdd['Police District Number'])

X = ptdd[['Start_Hour', 'Start_Month', 'Police District Number']]
y = ptdd['Place_Detail']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)

# Standard
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)

# MinMax
X_train_scale, X_test_scale = MinMax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)

# Robust
X_train_scale, X_test_scale = Robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)


## Ordinal encoding
ptdd['Place_Detail'] = ordinal(ptdd['Place_Detail'])
ptdd['Start_Hour'] = ordinal(ptdd['Start_Hour'])
ptdd['Police District Number'] = ordinal(ptdd['Police District Number'])

X = ptdd[['Start_Hour', 'Start_Month', 'Police District Number']]
y = ptdd['Place_Detail']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)

# Standard
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)

# MinMax
X_train_scale, X_test_scale = MinMax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)

# Robust
X_train_scale, X_test_scale = Robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)




### Second dataset

## label encoding
css['Start_Hour'] = label(css['Start_Hour'])
css['Crime Name1'] = label(css['Crime Name1'])
css['State'] = label(css['State'])

X = css[['Start_Hour','State']]
y = css['Crime Name1']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)

# Standard
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)

# MinMax
X_train_scale, X_test_scale = MinMax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)

# Robust
X_train_scale, X_test_scale = Robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)


## Ordinal encoding
css['Start_Hour'] = ordinal(css['Start_Hour'])
css['Crime Name1'] = ordinal(css['Crime Name1'])
css['State'] = ordinal(css['State'])

X = css[['Start_Hour','State']]
y = css['Crime Name1']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)

# Standard
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)

# MinMax
X_train_scale, X_test_scale = MinMax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)

# Robust
X_train_scale, X_test_scale = Robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)








