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
path = 'TermProject/Crime.csv'
df = pd.read_csv(path)

# drop the 0 value
df = df[df['Latitude'] != 0]
df = df[df['Longitude'] != 0]

threed = df[['Latitude', 'Longitude', 'Police District Number','Crime Name1']]

# drop the nan value
threed.dropna(axis=0, inplace=True)

threed = threed[threed['Police District Number'] == '3D']

# Plot the Scatter
plt.scatter(df['Latitude'], df['Longitude'], label='property',c='b')
plt.scatter(threed['Latitude'], threed['Longitude'], label='society',c='r')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

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
