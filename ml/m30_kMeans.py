from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

'''
K-means clustering은 비지도 학습의 클러스터링 모델 중 하나
clustering: 비슷한 특성을 가진 데이터끼리의 묶음
'''
datasets = load_iris()

irisDF = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
print(irisDF)

kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)
# n_cluster ->3개의 라벨을 뽑겠다는 의미 
# 
kmean.fit(irisDF)

results = kmean.labels_
print(results)
print(datasets.target) # -> 원래 y값

irisDF['cluster'] = kmean.labels_ # cluster에서 생성한 값
irisDF['target'] = datasets.target # 원래 y값

print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 

iris_results = irisDF.groupby(['target', 'cluster'])['sepal length (cm)'].count()
print(iris_results)

'''
target  cluster
0       0          50
1       1          48
        2           2 -> 1일 때 2인 값 => 틀린 값
2       1          14 -> 2일 때 1로 cluster된 값 -> 틀린 값
        2          36
=> 총 16개의 데이터가 잘못 분류됨을 알 수 있다
'''
