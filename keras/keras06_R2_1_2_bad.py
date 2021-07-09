# 1. R2를 음수가 아닌 0.5 이하로 만들어라.
# 2. 데이터 변경 x
# 3. layer -> Input, output 포함 6개 이상
# 4. batch_size = 1
# 5. epochs는 100 이상
# 6. hidden layer의 node는 10<= <=1000개 이하
# 7. train 70%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(100)) # 특성?
y = np.array(range(1,101))

x_train, y_train, x_test, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

