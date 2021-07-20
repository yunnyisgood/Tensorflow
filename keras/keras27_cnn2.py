from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# 오후 수업 내용 부분

model = Sequential() 
model.add(Conv2D(10, kernel_size=(2, 2),  # (N, 10, 10, 1)
padding='same' , input_shape=(10, 10, 1))) # (N, 10, 10, 10)
#        layer 층=10 | (2,2)로 잘라서 작업 |  (가로, 세로, 1이므로 흑백)
# (10, 10, 1)의 크기의 이미지를 (2, 2)로 잘라서 특징을 추출한다
# 즉, 2 X 2 형태의 필터 10개를 사용하여 특징을 추출한다 
# 그러므로 가중치는 총 40이라고 할 수 있다 
# 자르게 되면 이 층에서 출력 형태는 (None, 9, 9, 10)이 된다
# padding='same'이라고 지정했으므로 다음층에서 입력 형태의 행과 열의 크기와 동일하게 된다 


model.add(Conv2D(20, (2,2), activation='relu'))  # (None, 9, 9, 20)   
model.add(Conv2D(30, (2,2), activation='relu', padding='valid'))  # (None, 8, 8, 20)  
model.add(MaxPooling2D()) # (N, 4, 4, 30)
model.add(Conv2D(15, (2,2), activation='relu'))  # (None, 3, 3, 15)   
model.add(Flatten()) # (N, 135)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()
