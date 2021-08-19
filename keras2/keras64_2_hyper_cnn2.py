import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers.convolutional import Conv1D
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop


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

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000,  28, 28, 1).astype('float32')/25

# 2. modeling
def build_model(drop=0.2, optimizer=Adam, learning_rate=0.01): # (lr=0.01)
    inputs = Input(shape=(28, 28, 1), name='input')
    x =Conv2D(filters=64, kernel_size=(2, 2), padding='same', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(64, (2,2), activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(32, (2,2), activation='relu', name='hidden4')(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(32, activation='relu', name='hidden7')(x)
    x = Dense(8, activation='relu', name='hidden8')(x)

    outputs = Dense(10, activation='softmax', name='output')(x)
    optimizer = optimizer(learning_rate)
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
    learning_rate = [0.1, 0.001, 0.0001]
    optimizers = [Adam, RMSprop, Adadelta]
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    return {"batch_size": batches, "optimizer": optimizers, 
                "drop":dropout, "learning_rate":learning_rate }

hyper_parameters = create_hyper_parameter()

model = KerasClassifier(build_fn=build_model, verbose=1) # epochs=2, validation_split=0.2)
model = RandomizedSearchCV(model, hyper_parameters,  cv=2)

model.fit(x_train, y_train, verbose=1, epochs=1, validation_split=0.2) 
# epochs, validation 모두 사용가능
# 이 때 이미 cv를 사용했기 때문에 validation을 한번 나눈 상태에서 또 validation_data를 나누게 된다

print('model.best_params_: ', model.best_params_)
print('model.best_estimator_: ', model.best_estimator_)
print('model.best_score_: ', model.best_score_)
acc = model.score(x_test, y_test)
print('acc: ', acc)


'''
1. GridSearchCv

model.best_params_:  {'batch_size': 5000, 'drop': 0.3, 'optimizer': 'rmsprop'}
model.best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000017AAEB3BA30>
model.best_score_:  0.1671166643500328
acc:  0.18459999561309814

modeling Conv1D
model.best_params_:  {'batch_size': 1000, 'drop': 0.5, 'optimizer': 'rmsprop'}
model.best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000220600E9B80>
model.best_score_:  0.1987999975681305
acc:  0.5037000179290771

modeling Conv2D
model.best_params_:  {'batch_size': 1000, 'drop': 0.4, 'optimizer': 'rmsprop'}
model.best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001D499A61F70>
model.best_score_:  0.6403333246707916
10/10 [==============================] - 0s 11ms/step - loss: 2.5046 - acc: 0.7168
acc:  0.7167999744415283

modeling Conv2D, Adam(lr=0.001)
model.best_params_:  {'batch_size': 1000, 'drop': 0.4, 'optimizer': 'adam'}
model.best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001EA22D80700>
model.best_score_:  0.6217666566371918
10/10 [==============================] - 0s 11ms/step - loss: 5.6378 - acc: 0.6270
acc:  0.6269999742507935

1.1)lr params 설정 후
model.best_params_:  {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'learning_rate': 0.001, 'drop': 0.5, 'batch_size': 4000}
model.best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000211F6615550>
model.best_score_:  0.28333333134651184
3/3 [==============================] - 0s 30ms/step - loss: 3.6401 - acc: 0.4807
acc:  0.48069998621940613

'''







