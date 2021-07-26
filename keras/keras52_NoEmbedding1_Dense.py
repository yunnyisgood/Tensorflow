from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers.recurrent import LSTM 
# Embeding이 2차원 벡터로 치환해주기때문에 별도의 인코딩 필요 없다 

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화이니다.', '한 번 더 보고 싶네요', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '청순이가 잘 생기긴 했어요'] # 13개의 문장


# 라벨링
# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index) # 라벨이 총 27개로 된다 
'''
{'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화이니다': 10, '한': 11, '번
': 12, '더': 13, '보고': 14, '싶네요': 15, '글세요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없 
어요': 22, '재미없다': 23, '재밋네요': 24, '청순이가': 25, '생기긴': 26, '했어요': 27}
'''
x = token.texts_to_sequences(docs)
print(x)
'''
[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
=> 크기가 일정하지 않다 [List는 크기에 자유롭지만, numpy array는 크기를 맞춰줘야 한다 ]
-> 의미 없는 값인 0으로 채워줘야 한다
-> 단, 뒤에 위치한 값이 가장 영향을 주게 되므로 
=> 앞에서부터 0을 넣어줘야 한다 
[11, 12, 13, 14, 15] -> 제일 길기 때문에 13, 5 형태로 만들어줘야 한다 
'''
word_size = len(token.word_index)
print(word_size) # 27

# 전처리  
# pad_sequences를 사용하여 일정한 길이로 데이터를 맞춰준다
pad_x = pad_sequences(x, padding='pre', maxlen=5) # post
print(pad_x)
print(pad_x.shape) # (13, 5)

print(np.unique(pad_x))
'''[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27]'''


# one hot encoding 하게 되면 어떻게 바뀔까?
# 라벨의 개수 만큼 (13, 5, N)에 들어가게 된다. 
# 즉 , (13, 5) => (13, 5, 27) 
# 이렇게 되면 값이 너무 커지게 된다


model = Sequential()
model.add(Dense(32, input_shape=(5, )))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(pad_x, labels, epochs=100, batch_size=8 )

# evaluate
acc = model.evaluate(pad_x, labels)
print('acc: ', acc)
