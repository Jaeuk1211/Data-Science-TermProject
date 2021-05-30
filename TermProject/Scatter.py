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

df = df[['Latitude', 'Longitude', 'Crime Name1']]

df = df[df['Latitude'] != 0]
df = df[df['Longitude'] != 0]

# get the median and fill the null value
df.dropna(axis=0, inplace=True)

property = df.copy()
person = df.copy()
society = df.copy()
other = df.copy()

property = df[df['Crime Name1'] == 'Crime Against Property']
person = df[df['Crime Name1'] == 'Crime Against Person']
society = df[df['Crime Name1'] == 'Crime Against Society']
other = df[df['Crime Name1'] == 'Other']


property = property.loc[:2500,:]
person = person.loc[:2500,:]
society = society.loc[:2500,:]
other = other.loc[:2500,:]


plt.scatter(property['Latitude'], property['Longitude'], label='property',c='r')
plt.scatter(person['Latitude'], person['Longitude'], label='person',c='g')
plt.scatter(society['Latitude'], society['Longitude'], label='society',c='b')
plt.scatter(other['Latitude'], other['Longitude'], label='other',c='yellow')

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()


