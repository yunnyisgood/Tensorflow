import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers.convolutional import Conv1D


'''
실습
cnn으로 변경
노드의 개수, activation도 추가
epochs=[1, 2, 3]
batch_size는 크게 
learning_mate 추가 

+ Dense, Conv1D와 같은 Layer도 파라미터 튠 가능하도록 만들기 
'''

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = x_train.reshape(60000, 28*28).astype('float32')/255
# x_test = x_test.reshape(10000, 28*28).astype('float32')/25

# 2. modeling
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28, 28), name='input')
    x = Dense(units=10, activation='relu', name='hidden1')(inputs)
    x = Conv1D(10, 2, activation='relu', name='hidden2')(x)
    x = Conv1D(10, 2, activation='relu', name='hidden3')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu', name='hidden4')(x)
    x = Dense(8, activation='relu', name='hidden5')(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs,outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='categorical_crossentropy')
    return model

# def build_model(drop=0.5, optimizer='adam'):
#     inputs = Input(shape=(28, 28), name='input')
#     x = Dense(units=10, activation='relu', name='hidden1')(inputs)
#     x = Dropout(drop)(x)
#     x = Conv1D(10, 2, activation='relu', name='hidden2')(x)
#     x = Dropout(drop)(x)
#     x = Flatten()(x)
#     x = Dense(32, activation='relu', name='hidden3')(x)
#     x = Dropout(drop)(x)
#     outputs = Dense(10, activation='softmax', name='output')(x)
#     model = Model(inputs,outputs)
#     model.compile(optimizer=optimizer, metrics=['acc'],
#                     loss='categorical_crossentropy')
#     return model

def create_hyper_parameter():
    batches = [1000, 2000, 3000, 4000, 5000] #  [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta'] # ['rmsprop', 'adam', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    return {"batch_size": batches, "optimizer": optimizers, 
                "drop":dropout }

hyper_parameters = create_hyper_parameter()
print(hyper_parameters)
# model2 = build_model()

model2 = KerasClassifier(build_fn=build_model, verbose=1) # epochs=2, validation_split=0.2)
# epochs, validation 모두 사용가능

model = GridSearchCV(model2, hyper_parameters, cv=2)
# model = RandomizedSearchCV(model2, hyper_parameters, cv=5)
# model = RandomizedSearchCV(model2, hyper_parameters, cv=5)
# RandomizedSearchCV에서 받아들이는 모델이 tensorflow 모델이 될 수 있을까? X
# 주로 sklearn 모델을 사용했다.
# tensorflow 모델을 sklearn 모델로 wrapping 시켜주면 사용할  수 있다
#  => KerasClassifier를 사용

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto', verbose=1, factor=0.5)
model.fit(x_train, y_train, verbose=1, epochs=2, validation_split=0.2, callbacks=reduce_lr) 
# epochs, validation 모두 사용가능
# 이 때 이미 cv를 사용했기 때문에 validation을 한번 나눈 상태에서 또 validation_data를 나누게 된다

print('model.best_params_: ', model.best_params_)
print('model.best_estimator_: ', model.best_estimator_)
print('model.best_score_: ', model.best_score_)
acc = model.score(x_test, y_test)
print('acc: ', acc)


'''
model.best_params_:  {'batch_size': 5000, 'drop': 0.3, 'optimizer': 'rmsprop'}
model.best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000017AAEB3BA30>
model.best_score_:  0.1671166643500328
acc:  0.18459999561309814

modeling 변경 이후
model.best_params_:  {'batch_size': 1000, 'drop': 0.5, 'optimizer': 'rmsprop'}
model.best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000220600E9B80>
model.best_score_:  0.1987999975681305
acc:  0.5037000179290771
'''







