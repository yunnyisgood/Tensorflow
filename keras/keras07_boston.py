from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential # 교육용 예제 임포트
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13) -> 13가지의 특성 -> Input_dim = 13
print(y.shape) # (506,) -> output_dim = 1
print(datasets.DESCR)

print(datasets.feature_names)










