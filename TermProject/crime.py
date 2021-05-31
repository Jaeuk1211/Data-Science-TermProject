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
df = pd.read_csv("TermProject/Crime.csv")

# missing value 확인
print(df.isna().sum())
print()


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
# Place_Detail, Start_Hour, Start_Month, Police District Number data
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
# Latitude, Longitude, Start_Hour
# Target : Start_Hour
lls = ptdd.copy()
lls['Latitude'] = df['Latitude']
lls['Longitude'] = df['Longitude']
lls.drop(['Start_Month','Place_Detail','Police District Number'], axis=1, inplace=True)


<<<<<<< HEAD
### Police District Number = 3D
d3 = ptdd.copy()
d3['Crime Name1'] = df['Crime Name1']
d3 = d3[d3['Police District Number']=='3D']
d3.drop(['Place_Detail', 'Start_Month','Police District Number'], axis=1, inplace=True)




=======
##########################################
# Scatter Plot

# drop the 0 value
sc = df[['Latitude', 'Longitude', 'Crime Name1']]
sc = sc[sc['Latitude'] != 0]
sc = sc[sc['Longitude'] != 0]

# drop the nan value
sc.dropna(axis=0, inplace=True)

# Classify Dataframes by the Crime category
property = sc.copy()
person = sc.copy()
society = sc.copy()
other = sc.copy()

property = sc[sc['Crime Name1'] == 'Crime Against Property']
person = sc[sc['Crime Name1'] == 'Crime Against Person']
society = sc[sc['Crime Name1'] == 'Crime Against Society']
other = sc[sc['Crime Name1'] == 'Other']

# plot data size = 1/100
property = property.loc[:2500,:]
person = person.loc[:2500,:]
society = society.loc[:2500,:]
other = other.loc[:2500,:]

# Plot the Scatter
plt.scatter(property['Latitude'], property['Longitude'], label='property',c='r')
plt.scatter(person['Latitude'], person['Longitude'], label='person',c='g')
plt.scatter(society['Latitude'], society['Longitude'], label='society',c='b')
plt.scatter(other['Latitude'], other['Longitude'], label='other',c='yellow')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
>>>>>>> 33dda5edc3e459c2a85dff616fc028ceb0ace733


##### bar chart
crime_name = sns.countplot(x='Crime Name1', data=df, order=df['Crime Name1'].value_counts().index)
crime_name.set_title('Number of crime name', fontsize=15)
plt.setp(crime_name.get_xticklabels(),fontsize=6)
plt.show()

df['Crime Name2'].dropna(axis=0, inplace=True)
name2 = sns.countplot(y='Crime Name2', data=df, order=df['Crime Name2'].value_counts().index)
name2.set_title('Number of crime name', fontsize=15)
plt.setp(name2.get_yticklabels(),fontsize=5)
plt.show()

month = sns.countplot(x='Start_Month', data=ptdd, order=ptdd['Start_Month'].value_counts().index)
month.set_title('Number of crime month', fontsize=15)
plt.show()

time = sns.countplot(x='Start_Hour', data=ptdd, order=ptdd['Start_Hour'].value_counts().index)
time.set_title('Number of crime time zone', fontsize=15)
plt.show()

df['City'].dropna(axis=0, inplace=True)
city = sns.countplot(y='City', data=df, order=df['City'].value_counts().index)
city.set_title('Number of crime city', fontsize=15)
plt.setp(city.get_yticklabels(),fontsize=5)
plt.show()

place_de = sns.countplot(y='Place_Detail', data=ptdd, order=ptdd['Place_Detail'].value_counts().index)
place_de.set_title('Number of crime place', fontsize=15)
plt.setp(place_de.get_yticklabels(),fontsize=5)
plt.show()

police = sns.countplot(x='Police District Number', data=df, order=df['Police District Number'].value_counts().index)
police.set_title('Number of crime police district number', fontsize=15)
plt.show()



### Function definition

# Label Encoding
def label(col):
    encoder = preprocessing.LabelEncoder()
    col = encoder.fit_transform(col)
    cg = encoder.classes_
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
    print('KNN test accuracy : ', score)
    return predict


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
    

# Random Forest
def forest(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(max_depth = 8,
                                  max_features = 3,
                                  min_samples_split = 5,
                                  n_estimators = 100)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print("RandomForestClassifier test accuracy : ", rf.score(x_test,y_test))
    print()
    return y_pred




### First dataset
print('***** Predict place with month, hour, police district number *****')
print()

## Label encoding
print('*** Label encoding ***')
ptdd['Place_Detail'] = label(ptdd['Place_Detail'])
ptdd['Start_Hour'] = label(ptdd['Start_Hour'])
ptdd['Police District Number'] = label(ptdd['Police District Number'])

X = ptdd[['Start_Hour', 'Start_Month', 'Police District Number']]
y = ptdd['Place_Detail']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()

# Standard
print('** Standard Scaling **')
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# MinMax
print('** MinMax Scaling **')
X_train_scale, X_test_scale = minmax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# Robust
print('** Robust Scaling **')
X_train_scale, X_test_scale = robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()


## Ordinal encoding
print('*** Ordinal encoding ***')
ptdd['Place_Detail'] = ordinal(ptdd['Place_Detail'])
ptdd['Start_Hour'] = ordinal(ptdd['Start_Hour'])
ptdd['Police District Number'] = ordinal(ptdd['Police District Number'])

X = ptdd[['Start_Hour', 'Start_Month', 'Police District Number']]
y = ptdd['Place_Detail']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()

# Standard
print('** Standard Scaling **')
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# MinMax
print('** MinMax Scaling **')
X_train_scale, X_test_scale = minmax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# Robust
print('** Robust Scaling **')
X_train_scale, X_test_scale = robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()




### Second dataset
print('***** Predict Crime type with hour, state *****')
print()

## label encoding
print('*** Label encoding ***')
css['Start_Hour'] = label(css['Start_Hour'])
css['Crime Name1'] = label(css['Crime Name1'])
css['State'] = label(css['State'])

X = css[['Start_Hour','State']]
y = css['Crime Name1']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()

# Standard
print('** Standard Scaling **')
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# MinMax
print('** MinMax Scaling **')
X_train_scale, X_test_scale = minmax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# Robust
print('** Robust Scaling **')
X_train_scale, X_test_scale = robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()


## Ordinal encoding
print('*** Ordinal encoding ***')
css['Start_Hour'] = ordinal(css['Start_Hour'])
css['Crime Name1'] = ordinal(css['Crime Name1'])
css['State'] = ordinal(css['State'])

X = css[['Start_Hour','State']]
y = css['Crime Name1']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()

# Standard
print('** Standard Scaling **')
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# MinMax
print('** MinMax Scaling **')
X_train_scale, X_test_scale = minmax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# Robust
print('** Robust Scaling **')
X_train_scale, X_test_scale = robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()




### Third dataset
print('***** Predict Crime time zone with latitude, longitude *****')
print()

## Label encoding
print('*** Label encoding ***')
lls['Start_Hour'] = label(lls['Start_Hour'])

X = lls[['Latitude','Longitude']]
y = lls['Start_Hour']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()

# Standard
print('** Standard Scaling **')
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# MinMax
print('** MinMax Scaling **')
X_train_scale, X_test_scale = minmax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# Robust
print('** Robust Scaling **')
X_train_scale, X_test_scale = robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()



## Ordinal encoding
print('*** Ordinal encoding ***')
lls['Start_Hour'] = ordinal(lls['Start_Hour'])

X = lls[['Latitude','Longitude']]
y = lls['Start_Hour']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()

# Standard
print('** Standard Scaling **')
X_train_scale, X_test_scale = standard(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# MinMax
print('** MinMax Scaling **')
X_train_scale, X_test_scale = minmax(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()

# Robust
print('** Robust Scaling **')
X_train_scale, X_test_scale = robust(X_train, X_test)
knn(X_train_scale, X_test_scale, y_train, y_test, 100)
print()
forest(X_train_scale, X_test_scale, y_train, y_test)
print()



### 3D
d3['Start_Hour'] = label(d3['Start_Hour'])

X = d3[['Start_Hour']]
y = d3['Crime Name1']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)
X_train_scale, X_test_scale = standard(X_train, X_test)

result = knn(X_train_scale, X_test_scale, y_train, y_test, 100)
df_result = pd.DataFrame(result, columns=['Crime Name'])

d3_name = sns.countplot(x='Crime Name', data=df_result, order=df_result['Crime Name'].value_counts().index)
d3_name.set_title('Number of crime name', fontsize=15)
plt.show()









