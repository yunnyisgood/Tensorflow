'''
전이학습
: 다른 사람이 만든 모델을 나에게 이전한다

vgg16: layer가 16으로 이루어진(실제로는 13개 사용)
'''
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
# model = VGG16()
# model = VGG19()

model.trainable=True
'''
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
-> weight의 갱신이 없다. 
-> 첫 layer의 weight를 사용하게 된다 
'''
model.summary()

print(len(model.weights)) # 26
# print(len(model.trainable_weights)) # 0 <- model.trainable=False
print(len(model.trainable_weights)) # 26 <- model.trainable=True



'''

input_2 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
........................
........................
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0


model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 100, 100, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 100, 100, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 100, 100, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 50, 50, 64)        0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 50, 50, 128)       73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 50, 50, 128)       147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 25, 25, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 25, 25, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 25, 25, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 25, 25, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 12, 12, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 12, 12, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 12, 12, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 12, 12, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 6, 6, 512)         0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 6, 6, 512)         2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 6, 6, 512)         2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 6, 6, 512)         2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 3, 3, 512)         0
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0


=> Flatten, fc1(fully connected) 등의 layer 사라짐 => FC 부분 따로 수정 가능!


CNN에는 두가지의 layer로 나뉜다
1. Convolution/Pooling 메커니즘은 이미지를 형상으로 분할하고 분석.
2. FC(Fully Connected Layer)로, 이미지를 분류/설명하는 데 가장 적합하게 예측.

FC(fully connected)?
이전 레이어의 출력을 "평탄화"하여 다음 스테이지의 입력이 될 수 있는 단일 벡터로 변환한다.
'''

