import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Conv1D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

'''
실습
pca를 통해 0.95 이상인 n_components가 몇개?
'''

(x_train, _), (x_test, _) = mnist.load_data()
# 

print(x_train.shape, x_test.shape)
# (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)
print(type(x))  # <class 'numpy.ndarray'>    

x = x.reshape(70000, 28*28)

pca = PCA(n_components=200)
x = pca.fit_transform(x)
print(x)
print(x.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
#  0.05365605 0.04336832]

print(sum(pca_EVR))
# 0.9661701241516084

cumsum = np.cumsum(pca_EVR)
print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196]

print(np.argmax(cumsum >= 0.95)+1)
# 154


