import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skpp
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
from sklearn import datasets
from sklearn import svm

from sklearn.preprocessing import OrdinalEncoder

path = '/Users/jaeuk/Desktop/python/dataset/Crime.csv'
dataframe = pd.read_csv(path)

dataframe = dataframe[['Latitude', 'Longitude', 'Crime Name1']]

dataframe.dropna(axis=0, inplace=True)

# transform the dataframe to numpy
df = dataframe.to_numpy()
test=[39,-77,'?']

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
print('k=5, List the 5 nearest')
prediction = predict_classification(df, test, 5)
print('The Prediction is %s.' % (prediction))