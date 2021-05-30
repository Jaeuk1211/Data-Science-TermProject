from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# read the data
path = '/Users/jaeuk/Desktop/python/TermProject/Crime.csv'
df = pd.read_csv(path)

pt = df[['Start_Date_Time', 'Crime Name1','Police District Number']]
# get the median and fill the null value
pt.dropna(axis=0, inplace=True)

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

# Encode using Ordinal Encoder
encoder = OrdinalEncoder()
encoder.fit(ptdd[['Start_Hour']])
ptdd[['Start_Hour']] = encoder.transform(ptdd[['Start_Hour']])

encoder.fit(ptdd[['Start_Month']])
ptdd[['Start_Month']] = encoder.transform(ptdd[['Start_Month']])

encoder.fit(df[['Police District Number']])
ptdd[['Police District Number']] = encoder.transform(ptdd[['Police District Number']])

encoder.fit(ptdd[['Crime Name1']])
ptdd[['Crime Name1']] = encoder.transform(ptdd[['Crime Name1']])


ptdd.dropna(axis=0, inplace=True)
ptdd = ptdd.astype({'Start_Hour':'int','Start_Month':'int','Police District Number':'int','Crime Name1':'int'})

# create the feature df for prediction
feature_name = ['Start_Hour','Start_Month','Police District Number']
X = ptdd[feature_name]
y = ptdd['Crime Name1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)


# Standard Scaling
standardScaler = StandardScaler()
X_train_scale = standardScaler.fit_transform(X_train)
X_test_scale = standardScaler.transform(X_test)


params ={
     'n_estimators':[30, 50, 100],
     'max_depth':[6,8,10],
     'max_features':[1,2,3],
     'min_samples_split':[5,10]}

rf_model = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_model, param_grid=params, cv=5, n_jobs=-1)
grid_cv.fit(X_train,y_train)

print('Best hyper parameter: {0}'.format(grid_cv.best_params_))

rf_tuned = RandomForestClassifier(max_depth = 8,
                                  max_features = 3,
                                  min_samples_split = 5,
                                  n_estimators = 100)

rf_tuned.fit(X_train_scale, y_train)
y_pred = rf_tuned.predict(X_test_scale)
print("RandomForestClassifier test accurarcy {}".format(rf_tuned.score(X_test_scale,y_test)))