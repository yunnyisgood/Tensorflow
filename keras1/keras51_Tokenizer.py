from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}

x = token.texts_to_sequences([text])
print(x)
# [[3, 1, 4, 5, 6, 1, 2, 2, 7]]
word_size = len(token.word_index)
print(word_size) # 7

print(type(x))

x = to_categorical(x)
print(x)
print(x.shape)
'''
[[[0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1.]]]
(1, 9, 8)

'''