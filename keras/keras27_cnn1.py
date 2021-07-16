from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential() #  시작은 (N, 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2, 2),padding='same' , input_shape=(5, 5, 1))) # (N,4, 4, 10)
#        layer 층=10 |(2,2)로 잘라서 작업 |  (가로, 세로, 1이므로 흑백)
model.add(Conv2D(20, (2,2), activation='relu'))  # (None, 3, 3, 20)   
model.add(Flatten()) # (N, 180)으로 변환 => 180은 특성이 된다
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()
