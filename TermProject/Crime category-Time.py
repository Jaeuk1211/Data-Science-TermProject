import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 10)
import warnings

# read csv file
df = pd.read_csv("/Users/jaeuk/Desktop/python/dataset/Crime.csv")

# Dispatch Date/Time의 missing value를 같은 row의 Start_Date_Time으로
# End_Date_Time은 같은 row의 Start_Date_Time으로
df['Dispatch Date / Time'] = np.where(pd.notnull(df['Dispatch Date / Time']) == True, df['Dispatch Date / Time'], df['Start_Date_Time'])
df['Crime Name1'].fillna(method='ffill', inplace=True)
df['Crime Name2'].fillna(method='ffill', inplace=True)
df['Crime Name3'].fillna(method='ffill', inplace=True)
df['End_Date_Time'] = np.where(pd.notnull(df['End_Date_Time']) == True, df['End_Date_Time'], df['Start_Date_Time'])


pt = df[['Crime Name1', 'Start_Date_Time']]


# - 를 기준으로 place를 place와 place_detail로 split
ptd = pt.copy()

# : 를 기준으로 start_date_time을 start_date, start_time, start_ampm으로 split
ptd[['Start_Date','Start_Time','Start_AMPM']] = pt['Start_Date_Time'].str.split(' ', n=2, expand=True)

# /를 기준으로 start_date를 start_day, start_month, start_year로 spilt
ptdd = ptd.copy()
ptdd[['Start_Month','Start_Day','Start_Year']] = ptd['Start_Date'].str.split('/', n=2, expand=True)
ptdd[['Start_Hour','Minute','Second']] = ptd['Start_Time'].str.split(':', n=2, expand=True)


# feature engineering
ptdd.drop(['Start_Date_Time','Start_Date','Start_Time','Start_Year','Start_Day','Minute','Second'], axis=1, inplace=True)

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

ptdd.drop(['Start_AMPM'], axis=1, inplace=True)


# Start_AMPM, Place, Place_Detail encoding
label = preprocessing.LabelEncoder()

ptdd['Start_Month'] = label.fit_transform(ptdd['Start_Month'])
ptdd['Start_Hour'] = label.fit_transform(ptdd['Start_Hour'])

ptdd = ptdd[['Start_Month', 'Start_Hour', 'Crime Name1']]

print(ptdd.head(20))
print()

ptdd.dropna(axis=0, inplace=True)

# transform the dataframe to numpy
dfn = ptdd.to_numpy()
test=[1,1,'?']

# function that calculates the euclidian distance
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += np.power((row1[i] - row2[i]),2)
	return round(np.sqrt(distance),1)

# function that gets the nearest neighbor
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# function that makes classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)

	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

#test
# Result
print('print the distance of each row')
for row in dfn:
	distance = euclidean_distance(test, row)
print()
print()
print('k=5, List the 5 nearest')
prediction = predict_classification(dfn, test, 5)
print()
print()
print('The Prediction is %s.' % (prediction))