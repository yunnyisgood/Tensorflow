import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
import warnings
warnings.filterwarnings('ignore')

# 1. data
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

y = to_categorical(y) # -> one-Hot-Encoding 하는 방법

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. modeling
def build_model(drop=0.2, optimizer=Adam, learning_rate=0.01):
    inputs = Input(shape=(13, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs,outputs)
    optimizer = optimizer(learning_rate)
    model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='categorical_crossentropy')
    return model

def create_hyper_parameter():
    batches = [1000, 2000, 3000, 4000, 5000] #  [10, 20, 30, 40, 50]
    learning_rate = [0.1, 0.001, 0.0001]
    optimizers = [Adam, RMSprop, Adadelta]
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    return {"batch_size": batches, "optimizer": optimizers, 
                "drop":dropout, "learning_rate":learning_rate }

hyper_parameters = create_hyper_parameter()
print(hyper_parameters)
# model2 = build_model()

model2 = KerasClassifier(build_fn=build_model, verbose=1) # epochs=2, validation_split=0.2)
# epochs, validation 모두 사용가능

# model = GridSearchCV(model2, hyper_parameters, cv=2)
model = RandomizedSearchCV(model2, hyper_parameters, cv=2)


model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2) 
# epochs, validation 모두 사용가능
# 이 때 이미 cv를 사용했기 때문에 validation을 한번 나눈 상태에서 또 validation_data를 나누게 된다

print('model.best_params_: ', model.best_params_)
print('model.best_estimator_: ', model.best_estimator_)
print('model.best_score_: ', model.best_score_)
acc = model.score(x_test, y_test)
print('acc: ', acc)


'''

RandomizedSearchCV
model.best_params_:  {'optimizer': 'rmsprop', 'drop': 0.5, 'batch_size': 5000}
model.best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000183F6D9BAF0>
model.best_score_:  0.5161290466785431
WARNING:tensorflow:6 out of the last 12 calls to <function Model.make_test_function.<locals>.test_function at 0x00000183F24974C0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
1/1 [==============================] - 0s 84ms/step - loss: 5.2213 - acc: 0.5185
acc:  0.5185185074806213
'''







