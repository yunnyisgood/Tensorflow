from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# 오후 수업 내용 부분

model = Sequential() #  시작은 (N, 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2, 2),  # (N, 10, 10, 1)
padding='same' , input_shape=(10, 10, 1))) # (N, 10, 10, 10)
#        layer 층=10 | (2,2)로 잘라서 작업 |  (가로, 세로, 1이므로 흑백)
model.add(Conv2D(20, (2,2), activation='relu'))  # (None, 9, 9, 20)   
model.add(Conv2D(30, (2,2), activation='relu', padding='valid'))  # (None, 8, 8, 20)  
model.add(MaxPooling2D()) # (N, 4, 4, 30)
model.add(Conv2D(15, (2,2), activation='relu'))  # (None, 3, 3, 15)   
model.add(Flatten()) # (N, 135)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()
