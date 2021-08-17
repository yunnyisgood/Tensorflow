from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel

'''
feature 삭제하는 SelectFromModel
'''

'''dataset = load_boston()
x = dataset.data
y = dataset.target'''

x, y = load_boston(return_X_y=True)
print(x.shape, y.shape)
# (506, 13) (506,)

x_train,x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

# modeling
model = XGBRegressor(n_jobs=8)

#fit
model.fit(x_train, y_train)

# evaluate
score = model.score(x_test, y_test)
print("model.score: ", score)

threshhold = np.sort(model.feature_importances_)
print(threshhold)
'''[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]-> 내림차순으로 정렬'''

for thresh in threshhold:
    print(thresh)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print('selection: ', selection)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
            score*100))

'''
가장 acc가 낮은 컬럼부터 제거한 모델을 돌려서 어큐러시를 측정한다

단, # modeling
model = XGBRegressor(n_jobs=8) 여기서 일단 모델에 대한 신뢰를 갖고 
select model을 해줘야 한다 

(404, 13) (102, 13)
Thresh=0.001, n=13, R2: 92.21%
0.0036337192
(404, 12) (102, 12)
Thresh=0.004, n=12, R2: 92.16%
0.012031149
(404, 11) (102, 11)
Thresh=0.012, n=11, R2: 92.03%
0.012204577
(404, 10) (102, 10)
Thresh=0.012, n=10, R2: 92.19%
0.014479355
(404, 9) (102, 9)
Thresh=0.014, n=9, R2: 93.08%
0.014791191
(404, 8) (102, 8)
Thresh=0.015, n=8, R2: 92.37%
0.017543204
(404, 7) (102, 7)
Thresh=0.018, n=7, R2: 91.48%
0.030416546
(404, 6) (102, 6)
Thresh=0.030, n=6, R2: 92.71%
0.04246345
(404, 5) (102, 5)
Thresh=0.042, n=5, R2: 91.74%
0.051825397
(404, 4) (102, 4)
Thresh=0.052, n=4, R2: 92.11%
0.06949984
(404, 3) (102, 3)
Thresh=0.069, n=3, R2: 92.52%
0.30128643
(404, 2) (102, 2)
Thresh=0.301, n=2, R2: 69.41%
0.42848358
(404, 1) (102, 1)
Thresh=0.428, n=1, R2: 44.98%

'''
