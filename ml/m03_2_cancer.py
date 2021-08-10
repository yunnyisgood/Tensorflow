import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

datasets = load_breast_cancer() # (569, 30)

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape) # (569, 30)
print(y.shape) # (569,)

print(np.unique(y)) # y데이터는 0과 1데이터로만 구성되어 있다


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)


scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#2.modeling
# model = LinearSVC()
# model.score:  0.7543859649122807
# accuracy_score:  0.7543859649122807

# model = SVC()
# model.score:  0.9064327485380117
# accuracy_score:  0.9064327485380117

# model =  KNeighborsClassifier()
# model.score:  0.9415204678362573
# accuracy_score:  0.9415204678362573

# model = LogisticRegression()
# model.score:  0.9590643274853801
# accuracy_score:  0.9590643274853801

# model = DecisionTreeClassifier()
# model.score:  0.9415204678362573
# accuracy_score:  0.9415204678362573

model = RandomForestClassifier()
# model.score:  0.9707602339181286
# accuracy_score:  0.9707602339181286

#3. training
model.fit(x_train, y_train)

#4.predict
y_pred = model.predict(x_test)
y_pred = np.rint(y_pred) # ->  반올림을 통해 0 아니면 1의 값으로 y_pred를 변형 
print(x_train,'의 예측결과 : ', y_pred)

result = model.score(x_test, y_test)
print('model.score: ', result)

acc = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc)

# loss:  0.19166618585586548
# accuracy:  0.9298245906829834

# y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print('r2 스코어: ', r2)

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# plt.title("loss, val_loss")
# plt.xlabel("epochs")
# plt.ylabel("loss, val_loss")
# plt.legend(["train loss", "val_loss"])
# plt.show()

