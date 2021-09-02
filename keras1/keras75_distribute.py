import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time
import tensorflow as tf
from tensorflow.python.distribute.distribute_lib import Strategy

'''
모델 분산 훈련
GPU 동시에 2개 쓰기
'''

# data

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 분산처리

# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
# tf.distribute.HierarchicalCopyAllReduce()
  tf.distribute.ReductionToOneDevice()
)

#  with strategy로 모델 감싸기

with strategy.scope():
    # modeling -> CNN
    model = Sequential()
    # model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', input_shape=(28 * 28, )))
    model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (2,2), activation='relu'))  
    model.add(MaxPool2D()) 
    model.add(Flatten()) 
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # compile       -> metrics=['acc']
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.2,
shuffle=True, batch_size=256)  # 분산처리할때에는 배치사이즈가 클수록 좋다 
end_time = time.time() - start_time

# evaluate -> predict 할 필요는 없다
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss: ', loss[0])
print('accuracy: ', loss[1])


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


'''
loss:  0.46171167492866516
accuracy:  0.980400025844574

CNN FC Layer로 실행했을 떄 
걸린시간:  464.93610739707947
loss:  0.03197010979056358
accuracy:  0.9922000169754028

DNN 으로 실행했을 떄 
걸린시간:  32.44279766082764
loss:  0.14543354511260986
accuracy:  0.9768000245094299


'''