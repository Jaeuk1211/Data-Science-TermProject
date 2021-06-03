import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 10)


# Read csv file
df = pd.read_csv("C:\\Users\\82109\\OneDrive\\문서\\software\\3-1\\data science\\Termproject\\Crime.csv")

# Check missing value
print(df.isna().sum())
print()

# Drop rows where Latitude or Longitude is 0
df = df[df['Latitude'] != 0]
df = df[df['Longitude'] != 0]

# Fill missing values of Dispatch Date/Time with Start_Date_Time of the same row
# Fill missing values of End_Date_Time with Start_Date_Time of the same row
df['Dispatch Date / Time'] = np.where(pd.notnull(df['Dispatch Date / Time']) == True, df['Dispatch Date / Time'], df['Start_Date_Time'])
df['Crime Name1'].fillna(method='ffill', inplace=True)
df['Crime Name2'].fillna(method='ffill', inplace=True)
df['Crime Name3'].fillna(method='ffill', inplace=True)
df['End_Date_Time'] = np.where(pd.notnull(df['End_Date_Time']) == True, df['End_Date_Time'], df['Start_Date_Time'])


pt = df[['Place','Start_Date_Time']]

# Split Place to Place and Place_Detail by '-'
ptd = pt.copy()
ptd[['Place','Place_Detail']] = pt['Place'].str.split(' - ', n=1, expand=True)

# Split Start_Date_Time to Start_Date, Start_Time, Start_AMPM by ':'
ptd[['Start_Date','Start_Time','Start_AMPM']] = pt['Start_Date_Time'].str.split(' ', n=2, expand=True)

# Split Start_Date to Start_Day, Start_Month, Start_Year by '/'
ptdd = ptd.copy()
ptdd[['Start_Month','Start_Day','Start_Year']] = ptd['Start_Date'].str.split('/', n=2, expand=True)
ptdd[['Start_Hour','Minute','Second']] = ptd['Start_Time'].str.split(':', n=2, expand=True)

# Fill missing values of Place_Detail with Place of the same row
ptdd['Place_Detail'] = np.where(pd.notnull(ptdd['Place_Detail']) == True, ptdd['Place_Detail'], ptdd['Place'])

# Feature engineering
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

### Police District Number = 3D
d3 = ptdd.copy()
d3['Crime Name1'] = df['Crime Name1']
d3 = d3[d3['Police District Number']=='3D']
d3.drop(['Place_Detail', 'Start_Month','Police District Number'], axis=1, inplace=True)



##### Scatter Plot

# 1.Whole data by crime name
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
plt.legend()
plt.show()

# 2. Whole data & 3D district
threed = df[['Latitude', 'Longitude', 'Police District Number','Crime Name1']]
threed = threed[threed['Police District Number'] == '3D']

# Plot the Scatter
plt.scatter(df['Latitude'], df['Longitude'], label='property',c='b')
plt.scatter(threed['Latitude'], threed['Longitude'], label='society',c='r')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

# 3. 3D district data by crime name
# Classify Dataframes by the Crime category
property = threed.copy()
person = threed.copy()
society = threed.copy()
other = threed.copy()

property = threed[threed['Crime Name1'] == 'Crime Against Property']
person = threed[threed['Crime Name1'] == 'Crime Against Person']
society = threed[threed['Crime Name1'] == 'Crime Against Society']
other = threed[threed['Crime Name1'] == 'Other']

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
plt.legend()
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()



##### bar chart
crime_name = sns.countplot(x='Crime Name1', data=df, order=df['Crime Name1'].value_counts().index)
crime_name.set_title('The number of crime name1', fontsize=15)
plt.setp(crime_name.get_xticklabels(),fontsize=6)
plt.show()

df['Crime Name2'].dropna(axis=0, inplace=True)
name2 = sns.countplot(y='Crime Name2', data=df, order=df['Crime Name2'].value_counts().index)
name2.set_title('The number of crime name2', fontsize=15)
plt.setp(name2.get_yticklabels(),fontsize=5)
plt.show()

month = sns.countplot(x='Start_Month', data=ptdd, order=ptdd['Start_Month'].value_counts().index)
month.set_title('Number of crime month', fontsize=15)
plt.show()

time = sns.countplot(x='Start_Hour', data=ptdd, order=ptdd['Start_Hour'].value_counts().index)
time.set_title('The number of crime time zone', fontsize=15)
plt.show()

df['City'].dropna(axis=0, inplace=True)
city = sns.countplot(y='City', data=df, order=df['City'].value_counts().index)
city.set_title('The number of crime city', fontsize=15)
plt.setp(city.get_yticklabels(),fontsize=5)
plt.show()

place_de = sns.countplot(y='Place_Detail', data=ptdd, order=ptdd['Place_Detail'].value_counts().index)
place_de.set_title('The number of crime place', fontsize=15)
plt.setp(place_de.get_yticklabels(),fontsize=5)
plt.show()

police = sns.countplot(x='Police District Number', data=df, order=df['Police District Number'].value_counts().index)
police.set_title('Crime police district number frequency', fontsize=15)
plt.show()



### Function definition

# Encoding function
# k = 0, Label encoding
# k = 1, Ordinal encoding
def encode(k, data, encode_list):
    if(k==0):
        encoder = preprocessing.LabelEncoder()
        for i in range(len(encode_list)):
            data[encode_list[i]] = encoder.fit_transform(data[encode_list[i]])
    elif(k==1):
        encoder = preprocessing.OrdinalEncoder()
        for i in range(len(encode_list)):
            data[encode_list[i]] = encoder.fit_transform(data[[encode_list[i]]])
    return data


# Standard, MinMax, Robust Scaling
def scaling(k, train, test):
    if(k==0):
        scaler = preprocessing.StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.fit_transform(test)
        return train, test
    elif(k==1):
        scaler = preprocessing.MinMaxScaler()
        train = scaler.fit_transform(train)
        test = scaler.fit_transform(test)
        return train, test
    elif(k==2):
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
                                  min_samples_split = 5,
                                  n_estimators = 100)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print("RandomForestClassifier test accuracy : ", rf.score(x_test,y_test))


# Run algorithms with different scaling
def algo(x_train, x_test, y_train, y_test, n):
    scale_list = ['Standard scaling','MinMax scaling','Robust scaling']
    for i in range(len(scale_list)):
        print('* ', scale_list[i], ' *')
        x_train_scale, x_test_scale = scaling(i, x_train, x_test)
        knn(x_train_scale, x_test_scale, y_train, y_test, n)
        forest(x_train_scale, x_test_scale, y_train, y_test)
        print()




### First dataset
print('***** Predict place with month, hour, police district number *****')
print()

## Label encoding
print('*** Label encoding ***')
encode_list = ['Place_Detail','Start_Hour','Police District Number']
ptdd = encode(0, ptdd, encode_list)

X = ptdd[['Start_Hour', 'Start_Month', 'Police District Number']]
y = ptdd['Place_Detail']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()
algo(X_train, X_test, y_train, y_test, 100)
print()

## Ordinal encoding
print('*** Ordinal encoding ***')
ptdd = encode(1, ptdd, encode_list)

X = ptdd[['Start_Hour', 'Start_Month', 'Police District Number']]
y = ptdd['Place_Detail']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()
algo(X_train, X_test, y_train, y_test, 100)
print()



### Second dataset
print('***** Predict Crime type with hour, state *****')
print()

## Label encoding
print('*** Label encoding ***')
encode_list = ['Start_Hour','Crime Name1','State']
css = encode(0, css, encode_list)

X = css[['Start_Hour','State']]
y = css['Crime Name1']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()
algo(X_train, X_test, y_train, y_test, 100)
print()

## Ordinal encoding
print('*** Ordinal encoding ***')
css = encode(1, css, encode_list)

X = css[['Start_Hour','State']]
y = css['Crime Name1']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()
algo(X_train, X_test, y_train, y_test, 100)
print()



### Third dataset
print('***** Predict Crime time zone with latitude, longitude *****')
print()

## Label encoding
print('*** Label encoding ***')
encode_list = ['Start_Hour']
lls = encode(0, lls, encode_list)

X = lls[['Latitude','Longitude']]
y = lls['Start_Hour']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()
algo(X_train, X_test, y_train, y_test, 100)
print()

## Ordinal encoding
print('*** Ordinal encoding ***')
lls = encode(1, lls, encode_list)

X = lls[['Latitude','Longitude']]
y = lls['Start_Hour']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)

kfold(100, 10, X, y)
print()
algo(X_train, X_test, y_train, y_test, 100)
print()



### 3D
encode_list

X = d3[['Start_Hour']]
y = d3['Crime Name1']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01, random_state=0, shuffle=True)
X_train_scale, X_test_scale = standard(X_train, X_test)

result = knn(X_train_scale, X_test_scale, y_train, y_test, 100)
df_result = pd.DataFrame(result, columns=['Crime Name'])

d3_name = sns.countplot(x='Crime Name', data=df_result, order=df_result['Crime Name'].value_counts().index)
d3_name.set_title('Number of crime name', fontsize=15)
plt.show()

