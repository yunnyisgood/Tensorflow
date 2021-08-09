import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam


# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,5,4,7,9,11,13,12])

#2. 모델
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3 
# optimizer = Adam(lr=0.0001)
# 러닝메이트까지 적용할 수 있도록 따로 명시한다 
# loss:  0.9125338792800903 결과물:  [[13.821549]]

# optimizer = Adagrad(lr=0.01)
# loss:  1.0514343976974487 결과물:  [[13.550894]]

# optimizer = Adagrad(lr=0.001)
# loss:  0.9250622987747192 결과물:  [[13.798626]]

# optimizer = Adagrad(lr=0.0001)
# loss:  1.1422046422958374 결과물:  [[13.6996975]]

# optimizer = Adamax(lr=0.01)
# loss:  0.8675593137741089 결과물:  [[14.210067]]

# optimizer = Adamax(lr=0.001)
# loss:  0.8430672883987427 결과물:  [[14.285625]]

# optimizer = Adamax(lr=0.0001)
# loss:  0.9860073924064636 결과물:  [[13.860894]]

# optimizer = Adadelta(lr=0.01)
# loss:  1.0702321529388428 결과물:  [[13.823378]]

# optimizer = Adadelta(lr=0.001)
# loss:  22.932723999023438 결과물:  [[5.486394]]

# optimizer = Adadelta(lr=0.0001)
# loss:  53.697731018066406 결과물:  [[0.9587412]]

# optimizer = RMSprop(lr=0.01)
# loss:  100.63780212402344 결과물:  [[28.869345]]

# optimizer = RMSprop(lr=0.001)
# loss:  7.463292121887207 결과물:  [[18.812054]]

# optimizer = RMSprop(lr=0.0001)
# loss:  0.9022303819656372 결과물:  [[13.86165]]

# optimizer = SGD(lr=0.01)
# loss:  nan 결과물:  [[nan]]

# optimizer = SGD(lr=0.001)
# loss:  1.3261969089508057 결과물:  [[15.5102005]]

# optimizer = SGD(lr=0.0001)
# loss:  0.9646746516227722 결과물:  [[14.39785]]

# optimizer = Nadam(lr=0.01)
# loss:  2.2886006832122803 결과물:  [[11.954113]]

# optimizer = Nadam(lr=0.001)
# loss:  1.2111692428588867 결과물:  [[13.256499]]

optimizer = Nadam(lr=0.0001)
# loss:  0.9439903497695923 결과물:  [[14.808616]]

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss: ', loss, '결과물: ', y_pred)

'''
loss:  1.5179026126861572 결과물:  [[12.833406]]

 Adam(lr=0.1)
loss:  11.537897109985352 결과물:  [[15.905843]]

 Adam(lr=0.001)
 loss:  0.8941632509231567 결과물:  [[14.784194]]

 Adam(lr=0.0001)
loss:  0.9125338792800903 결과물:  [[13.821549]]
'''