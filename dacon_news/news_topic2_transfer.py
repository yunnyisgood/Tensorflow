import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Flatten, MaxPooling1D, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time 
from tensorflow.keras.optimizers import Adam 
import matplotlib.pyplot as plt
import datetime
from tensorflow.python.keras.layers.core import Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


dataset = pd.read_csv('./dacon/_data/train_data.csv', sep=',', index_col='index', 
header=0)
x_pred =pd.read_csv('./dacon/_data/test_data.csv', sep=',', index_col='index', 
header=0)
submission = pd.read_csv('./dacon/_data/sample_submission.csv', sep=',', index_col='index', 
header=0)

print(dataset.columns)
print(x_pred)

def clean_text(text):# 특수 문자 제거 
    cleaned_tex = re.sub('[-=+,#/\?:^$.@*\"*~&%`!\\`|\(\)\[\]\<\>`\'...>]', '', text) 
    cleaned_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", text)
    return cleaned_text

x = dataset['title'].dropna(axis=0).apply(lambda x : clean_text(x))
y = dataset['topic_idx'].dropna(axis=0)
x_pred = x_pred['title'].dropna(axis=0).apply(lambda x : clean_text(x))

x = x.tolist()
x_pred = x_pred.tolist()
y = y.to_numpy()

print("문장최대길이: ", max(len(i) for i in x))
# 9
print("문장 평균길이: ", sum(map(len, x)) / len(x))
# 2

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=9)

vectorizer = TfidfVectorizer(analyzer='word', sublinear_tf=True,
 ngram_range=(1, 2), max_features=10000, binary=False)

print(x_train)
print(type(x_train))


x_train = vectorizer.fit_transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()
x_pred = vectorizer.transform(x_pred).toarray()

print('2: ', x_train[:10])

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_pred.shape)

print(type(x_train), print(y_train))
# (31957, 1000) (13697, 1000) (31957,) (13697,) (9131, 1000)
# (31957, 2) (13697, 2) (31957,) (13697,) (9131, 2)

model = Sequential()
model.add(Dense(100, input_dim = 10000, activation = "relu"))
model.add(Dropout(0.8))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(7, activation = "softmax"))

# compile 
optimizer = Adam(lr=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer = tf.optimizers.Adam(0.001), metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start_time = time.time()

# 계층 교차 검증
n_fold = 5  
seed = 9

cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=seed)

for i, (i_trn, i_val) in enumerate(cv.split(x_train, y_train), 1):
    print(f'training model for CV #{i}')

    hist = model.fit(x_train[i_trn], 
            to_categorical(y_train[i_trn]),
            validation_data=(x_train[i_val], to_categorical(y_train[i_val])),
            epochs=100,
            batch_size=1024)

hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, callbacks=[es, reduce_lr], validation_split=0.25)

end_time = time.time() -start_time

date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
weight_path = './dacon/_save/' 
weight_path = "".join([weight_path, "news_topic_weight_save", date_time, ".h5"]) 

model.save_weights(weight_path)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# evaluate 
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('acc: ', acc[-1])
print('val_acc: ', val_acc[-1])

csv_path = './dacon/_save/' 
csv_path = "".join([csv_path, "news_topic__submission", date_time, ".csv"])

y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis=1)
submission['topic_idx'] = y_pred
submission.to_csv(csv_path)

# 시각화 
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2) 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()


'''
max_feature = 50000
loss:  0.6577078700065613
acc:  0.9373362064361572
val_acc:  0.7999061346054077

max_feature = 55000
loss:  0.6634199023246765
acc:  0.9456287622451782
val_acc:  0.8028786182403564

loss:  0.6620551943778992
acc:  0.9416389465332031
val_acc:  0.8033479452133179

loss:  0.6238920092582703
acc:  0.995340883731842
val_acc:  0.7622027397155762

epochs=100, batch_size=10, reduce_lr=0.001
loss:  0.6961246728897095
acc:  0.803479790687561
val_acc:  0.7871088981628418

max_feature=30000
loss:  0.6803905367851257
acc:  0.8290566205978394
val_acc:  0.7948685884475708

max_feature=40000
loss:  0.6963464021682739
acc:  0.8773730397224426
val_acc:  0.788861095905304
'''