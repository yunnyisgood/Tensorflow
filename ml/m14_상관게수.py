'''
column간의 상관계수
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

dataset = load_iris()
print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

print(dataset.target_names)
# ['setosa' 'versicolor' 'virginica']

x = dataset.data
y = dataset.target
print(x.shape, y.shape)
# (150, 4) (150,)

df = pd.DataFrame(x, columns=dataset.feature_names) # feature_names는 x의 컬럼명
print(df)

#y컬럼 추가
df['target'] = y
print(df.head())

print('============================ 상관계수 하트맵 =================================')
print(df.corr())

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()