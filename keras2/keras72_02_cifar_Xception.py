from tensorflow.keras.layers import Dense, Flatten, GlobalAvgPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
import pandas as pd
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D, MaxPool2D, MaxPooling2D
import time
from tensorflow.keras.callbacks import EarlyStopping

# cifar10 실습 [FC를 모델로 하고 MAPooling 사용 ]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32 , 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


xcep = Xception(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

xcep.trainable =True   

model = Sequential()
model.add(xcep)
# model.add(GlobalAveragePooling2D())
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.trainable = True

print(len(model.weights)) # 26 -> 30  -> 4개 증가 : layer 2개 증가 했기 때문에 총 layer, bias  각각 총 4개 증가 
print(len(model.trainable_weights))  # 0 -> 4 : 위와 동일           

# model.trainable=True # 전체 모델 훈련을 동결한다 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1)
model.fit(x_train, y_train, verbose=1, epochs=100, batch_size=1024, validation_split=0.2)

# evaluate 
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])
print("걸린 시간: ", time.time()-start_time)


'''
둘다 훈련 시켰을 때가 성능이 좋은지 분석 

1. cifar 10
1) GAP
    True True 
    loss:  1.8211287260055542
    accuracy:  0.24699999392032623
    걸린 시간:  1435.0262472629547

    True False 
    loss:  2.583721399307251
    accuracy:  0.09920000284910202

    False False 
    loss:  2.5722861289978027
    accuracy:  0.10090000182390213
    걸린 시간:  180.65854239463806

    False True 
    loss:  1.825405240058899
    accuracy:  0.24140000343322754
    걸린 시간:  1438.06161236763

2)Flatten

    True True 
    loss:  2.034485101699829
    accuracy:  0.18299999833106995
    걸린 시간:  1396.2824256420135

    True False 
    loss:  2.579019069671631
    accuracy:  0.10119999945163727
    걸린 시간:  176.82081031799316

    False False 
    loss:  2.4501872062683105
    accuracy:  0.11320000141859055
    걸린 시간:  177.30922508239746

    False True 
    acc: 0.2479
    loss:  1.832729697227478
    accuracy:  0.24789999425411224
    걸린 시간:  1370.5724246501923

2. cifar 100
1) GAP

    True True 
    loss:  4.717180252075195
    accuracy:  0.010599999688565731
    걸린 시간:  1373.0469415187836

    True False 
    loss:  4.9409589767456055
    accuracy:  0.011500000022351742
    걸린 시간:  175.4619607925415

    False False 
    loss:  4.923564434051514
    accuracy:  0.009999999776482582
    걸린 시간:  174.74113011360168

    False True 
    loss:  4.196385383605957
    accuracy:  0.030300000682473183
    걸린 시간:  1378.2208552360535

2)Flatten

    True True 
    loss:  4.6027631759643555
    accuracy:  0.01140000019222498
    걸린 시간:  1374.9687767028809

    True False 
    loss:  4.9580841064453125
    accuracy:  0.004600000102072954
    걸린 시간:  176.46726512908936

    False False 
    loss:  4.88253927230835
    accuracy:  0.011599999852478504
    걸린 시간:  176.4555947780609

    False True 


'''

