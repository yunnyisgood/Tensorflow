from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import matplotlib.pyplot as plt
import time

# 1. data
dataset = load_boston()
x = dataset['data']
y = dataset['target']

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.8, shuffle=True, random_state=66)

scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. modeling
# cpu, gpu, 둘다 사용할건지 여기서 정의해준다 
# n_jobs = 1 -> core:1
model = XGBRegressor(n_estimators=10000, learning_rate = 0.01, # n_jobs=3
                        tree_method = 'gpu_hist',
                        predictor='gpu_predictor',    # cpu_predictor
                        gpu_id = 0
)

# 3. fit
start_time = time.time()

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'],
          eval_set=[(x_train, y_train), (x_test, y_test)]
)          

print('걸린 시간: ', time.time()-start_time)

'''
현재 사양:
i7-9700 / 2080ti

n_jobs = 1
걸린 시간:  10.787170886993408

n_jobs=2
걸린 시간:  8.297851085662842

n_jobs=4
걸린 시간:  7.448970317840576

n_jobs=8
걸린 시간:  7.930832147598267
'''