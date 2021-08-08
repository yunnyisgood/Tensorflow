import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from icecream import ic



samsung = pd.read_excel('../_data/삼성 최종2021-08-06_06시21분.xlsx', header=0)
stopwords = pd.read_csv('../_data/stopwords.txt').values.tolist()

ic(samsung) 

okt = Okt()

# 다시 형태소 분석
samsung_temp_list = []
for sentence in samsung['title']:
    temp_samsung = []
    temp_samsung = okt.normalize(sentence)
    temp_samsung  = okt.morphs(sentence) #  문장에서 명사 추출 
    temp_samsung = [word for word in temp_samsung if not word in stopwords]  # 불용어 처리 
    samsung_temp_list.append(temp_samsung)
samsung['samsung_temp_list'] = samsung_temp_list

temp_list = []
for i in samsung['word_list']:
    temp = eval(''.join(i))
    print(temp)
    temp_list.append(temp)
print(temp_list)

samsung['score'] = np.nan
samsung['score'] = pd.Series(temp_list)
ic(samsung)

# 각 기사별 헤드라인의 긍정/부정 최종수치 구하기 
total_score = []
for i in range(len(samsung['samsung_temp_list'])):
    temp_len = len(samsung['samsung_temp_list'][i])
    result = sum(samsung['score'][i])
    avg = result/temp_len
    total_score.append(avg)

samsung['total_score'] = np.nan
samsung['total_score'] = pd.Series(total_score)


# 날짜별 데이터로 합산해서 평균 구하기
grouped = samsung.groupby('date')
df = pd.DataFrame(grouped.mean())

df2 =df.drop(['2021.07.03.', '2021.07.04.', '2021.07.10.', '2021.07.11.', '2021.07.17.', '2021.07.18.',
'2021.07.24.', '2021.07.25.', '2021.07.31'])

ic(df2)

samsung_score = df2['total_score'].to_numpy()
print(samsung_score) 

np.save('../_data/samsung_save.npy', arr=samsung_score)

score = np.load('../_data/samsung_save.npy')
print(score)
print(len(score))

# 삼성주가 다운로드
start_date = '2021-07-01'
end_date = '2021-08-01'
SAMSUNG = data.get_data_yahoo('005930.KS', start_date, end_date)

dic = {
    'ds': SAMSUNG.index,
    'y': SAMSUNG['Close'],
    'score':score
}

df = pd.DataFrame.from_dict(dic, orient='index')
df = df.transpose()


# x, y 분류
x = df[['y', 'score']]#[:-3]
y = df[['y']]#[3:]
print(x)

# 판다스 -> 넘파이로 변환 
x = x.to_numpy()
y = y.to_numpy()

def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset)-size+1): # range(10-4=6) -> 6번동안 반복. 10개의 데이터를 5개씩 분리하기 위한 방법 
        subset = dataset[i : (i+size)] # dataset[0:5] -> dataset 0부터 4번째 값까지 
        aaa.append(subset)
    return np.array(aaa)

size = 5

x = split_x(x, size)
y = split_x(y, size)
print(x.shape, y.shape)
# (18, 5, 2) (18, 5, 1) (2, 5, 2)

x = x.reshape(18*5, 2).astype(float)
y = y.reshape(18*5, 1).astype(float)


x_train, x_test, y_train, y_test = train_test_split(x,  y, 
                                                        train_size=0.9, random_state=9)

# scaling
scaler = MinMaxScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)


# # x와 y를 분리 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)



print(x_train.shape, x_test.shape,
      y_train.shape, y_test.shape)


# modeling
model = Sequential()
model.add(LSTM(units=100, activation='relu', input_shape=(2, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()


# compile
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_split=0.1,
shuffle=True, batch_size=1)
end_time = time.time() - start_time


# evaluate 
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss: ', loss)

y_pred = model.predict(x_test)
print('x_test를 통한 y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

# 시각화 
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1) # 2개의 플롯을 할건데, 1행 1열을 사용하겠다는 의미 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2) # 2개의 플롯을 할건데, 1행 2열을 사용하겠다는 의미 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()