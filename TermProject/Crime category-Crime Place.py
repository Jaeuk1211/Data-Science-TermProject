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


pt = df[['Place','Crime Name1']]


# - 를 기준으로 place를 place와 place_detail로 split
ptd = pt.copy()

ptd[['Place','Place_Detail']] = pt['Place'].str.split(' - ', n=1, expand=True)

ptdd = ptd.copy()
# place_detail의 missing value를 같은 row의 place로
ptdd['Place_Detail'] = np.where(pd.notnull(ptdd['Place_Detail']) == True, ptdd['Place_Detail'], ptdd['Place'])

# Start_AMPM, Place, Place_Detail encoding
label = preprocessing.LabelEncoder()

ptdd['Place'] = label.fit_transform(ptdd['Place'])
ptdd['Place_Detail'] = label.fit_transform(ptdd['Place_Detail'])

ptdd = ptdd[['Place', 'Place_Detail', 'Crime Name1']]

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