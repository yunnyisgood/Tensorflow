from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 정제된 데이터 x, y 준비
x = np.array([1, 2, 3]) # 스칼라 3개짜리 벡터 1개 인것. 
y = np.array([1, 2, 3])

# 2. model
model = Sequential() # model은 순차적으로 구성된다. 
model.add(Dense(1, input_dim=1)) # 1은 output_dim.즉 y/ input_dim은 x -> x는 1차원

# 3. Compile 훈련
model.compile(loss='mse', optimizer="adam")

model.fit(x, y, epochs=1, batch_size=1)  
# 여기서 batch_size=1은 전체 데이터를 1번 훈련시켰다는 의미
# 제일 첫번째로 random한  w값으로 fit한다
# batch 작업을 한번에 한번씩 하겠다는 의미. epoch=1일 떄 batch_size=3이라면 훈련을 3번 시켰다는 의미
# epochs=?의 값이 1, 1000, 100000 등등에 따라 예측값이 변화한다. 
# fit을 통해 weight, bias 도출한다.
# 단 훈련횟수가 동일해도 도출되는 값은 달라질 수 있다. -> weight, bias 계속 달라짐. 
# 최적의 값들을 구한다면 weight, bias 모두 save 해줘야 한다. 

# 4. 평가, 예측
loss = model.evaluate(x, y) # evaluate()는 오차값을 반환해준다.
print('loss:', loss)
# loss: 0.18847961723804474

result = model.predict([4]) # 스칼라 1개짜리 벡터. 4에 대한 예측값을 반환하라는 의미
print('4의 예측값:', result)
# 4의 예측값: [[3.1961246]]