from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

vgg16.trainable =False  # vgg훈련을 동결한다 -> 0이 된다 

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# model.trainable=True # 전체 모델 훈련을 동결한다 

print(len(model.weights)) # 26 -> 30  -> 4개 증가 : layer 2개 증가 했기 때문에 총 layer, bias  각각 총 4개 증가 
print(len(model.trainable_weights))  # 0 -> 4 : 위와 동일           


''''
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 3, 3, 512)         14714688 => 16개의 층이 압축
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0
_________________________________________________________________
dense (Dense)                (None, 10)                46090
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 14,760,789
Trainable params: 14,760,789
Non-trainable params: 0

'''
