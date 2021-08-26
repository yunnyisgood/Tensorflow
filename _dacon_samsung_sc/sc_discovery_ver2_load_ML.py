import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import math
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ss  = pd.read_csv('../_data/samsung_sc_discovery/sample_submission.csv')

X = np.load('../_data/sc_train_x.npy')
Y = np.load('../_data/sc_train_y.npy')
np_test_fps_array = np.load('../_data/sc_test_x.npy')
# test_y = np.load('../_data/sc_test_x.npy')


# def create_deep_learning_model():
#     model = Sequential()
#     model.add(Dense(10000, input_dim=2048, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     model.compile(loss='mae', optimizer='adam', metrics=['mae'])
#     return model

print('X.shape: ', X.shape)
print('Y.shape: ', Y.shape)

#validation
# estimators = []
# # estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=XGBRegressor(n_estimators=100, n_jobs=-1), epochs=20)))
# pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("%.2f (%.2f) MAE" % (results.mean(), results.std()))
parameters = [
    {'rf__n_estimators':[100, 200], 'rf__max_depth':[6, 8, 10], 'rf__min_samples_leaf':[5, 7, 10]}, 
    {'rf__max_depth':[5, 6, 7], 'rf__min_samples_leaf':[3, 6, 9, 11], 'rf__min_samples_split':[3, 4, 5]},
    {'rf__min_samples_leaf':[3, 5, 7], 'rf__min_samples_split':[2, 3, 5, 10]},
    { 'rf__min_samples_split':[6, 8, 10]}
]

#2.modeling
# model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 17 candidates, totalling 85 fits

pipe = Pipeline([("scaler", MinMaxScaler()), ("rf", RandomForestRegressor())])
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)

# model = XGBRegressor(n_estimators=100, n_jobs=-1)
# es = EarlyStopping(monitor='loss', patience=1, verbose=1, restore_best_weights=True)
model.fit(X, Y)
test_y = model.predict(np_test_fps_array)


ss['ST1_GAP(eV)'] = test_y

# ss.to_csv('../_data/samsung_sc_discovery/dacon_baseline0819.csv', index=False)

date_time = str(datetime.now())
date_time = date_time[:date_time.rfind(':')].replace(' ', '_')
date_time = date_time.replace(':','시') + '분'

folder_path = os.getcwd()
csv_file_name = 'dacon_baseline{}.csv'.format(date_time)

ss.to_csv(csv_file_name, index=False)

print('csv 저장 완료 | 경로 : {}\\{}'.format(folder_path, csv_file_name))
os.startfile(folder_path)

# from dacon_submit_api import dacon_submit_api 

# result = dacon_submit_api.post_submission_file(
#     'dacon_baseline.csv', 
#     '7845225c14d48c7885ddfb51bbeab195fbe32a16064c3e1835f5c487edcb23d3', 
#     '235789', 
#     'DACONIO', 
#     'DACON_Baseline'
# )

