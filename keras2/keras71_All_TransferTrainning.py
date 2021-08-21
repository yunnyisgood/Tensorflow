'''
pre=trained model
'''

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

import keras

model = EfficientNetB7()

model.trainable = False

model.summary()

print(len(model.weights))
print(len(model.trainable_weights)) 

# 모델별로 parameter, weight 수 정리

'''
VGG16
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0

VGG19
Total params: 143,667,240
Trainable params: 0
Non-trainable params: 143,667,240

Xception
Total params: 22,910,480
Trainable params: 0
Non-trainable params: 22,910,480

ResNet101
Total params: 44,707,176
Trainable params: 0
Non-trainable params: 44,707,176

ResNet101V2
Total params: 44,675,560
Trainable params: 0
Non-trainable params: 44,675,560

ResNet152
Total params: 60,419,944
Trainable params: 0
Non-trainable params: 60,419,944

ResNet152V2
Total params: 60,380,648
Trainable params: 0
Non-trainable params: 60,380,648

ResNet50
Total params: 25,636,712
Trainable params: 0
Non-trainable params: 25,636,712

ResNet50V2
Total params: 25,613,800
Trainable params: 0
Non-trainable params: 25,613,800

InceptionResNetV2
otal params: 55,873,736
Trainable params: 0
Non-trainable params: 55,873,736

InceptionV3
Total params: 23,851,784
Trainable params: 0
Non-trainable params: 23,851,784

MobileNet
Total params: 4,253,864
Trainable params: 0
Non-trainable params: 4,253,864

MobileNetV2
Total params: 3,538,984
Trainable params: 0
Non-trainable params: 3,538,984

MobileNetV3Large
Total params: 5,507,432
Trainable params: 0
Non-trainable params: 5,507,432

MobileNetV3Small
Total params: 2,554,968
Trainable params: 0
Non-trainable params: 2,554,968

DenseNet121
Total params: 8,062,504
Trainable params: 0
Non-trainable params: 8,062,504

DenseNet169
Total params: 60,419,944
Trainable params: 0
Non-trainable params: 60,419,944

DenseNet201
Total params: 20,242,984
Trainable params: 0
Non-trainable params: 20,242,984

NASNetLarge
Total params: 88,949,818
Trainable params: 0
Non-trainable params: 88,949,818

NASNetMobile
Total params: 5,326,716
Trainable params: 0
Non-trainable params: 5,326,716

EfficientNetB0
Total params: 5,330,571
Trainable params: 0
Non-trainable params: 5,330,571

EfficientNetB1
Total params: 7,856,239
Trainable params: 0
Non-trainable params: 7,856,239

EfficientNetB7
Total params: 66,658,687
Trainable params: 0
Non-trainable params: 66,658,687
'''