from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
import pandas as pd

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable =True  # vgg훈련을 동결한다 -> 0이 된다 

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# model.trainable=True # 전체 모델 훈련을 동결한다 

print(len(model.weights)) # 26 -> 30  -> 4개 증가 : layer 2개 증가 했기 때문에 총 layer, bias  각각 총 4개 증가 
print(len(model.trainable_weights))  # 0 -> 4 : 위와 동일           


pd.set_option('max_colwidth', None)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer_Type', 'Layer_Name', 'Layer_Trainable'])

print(results)