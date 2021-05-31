from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

# read the data
path = '/Users/jaeuk/Desktop/python/TermProject/Crime.csv'
df = pd.read_csv(path)

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


