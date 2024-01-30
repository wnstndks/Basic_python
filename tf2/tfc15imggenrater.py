# 이미지 보강 - 기존 이미지를 변형시켜 이미지 데이터에 양을 늘리기

'''
rotation_range : 이미지회전값
zoom_range : 이미지일부확대
shear_range : 이미지기울기
width_shift_range : 좌우이동
height_shift_range : 상하이동
horizontal_flip : 이미지가로뒤집기
vertical_filp : 이미지세로뒤집기
'''

import tensorflow as tf
import keras
import sys
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test)= keras.datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape,x_test.shape,y_test.shape) # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# print(x_train[0]) # 0번쨰 feature
# print(y_train[0]) # 0번쨰 label 9
print()

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankel boot']
print(set(y_train))

x_train=x_train.reshape(-1,28,28,1).astype('float32')/255
x_test=x_test.reshape(-1,28,28,1).astype('float32')/255
y_train=keras.utils.to_categorical(y_train)
y_test=keras.utils.to_categorical(y_test)
# print(x_train[:1])
# print(y_train[:1])

# plt.figure(figsize=(10,10))
# for c in range(100):
#     plt.subplot(10,10,c+1)
#     plt.axis('off')
#     plt.imshow(x_train[c].reshape(28,28),cmap='gray')
# plt.show()

# 이미지 보강
from keras.preprocessing.image import ImageDataGenerator
'''
img_generate=ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.5,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
)
augment_size=100
x_augment = img_generate.flow(np.tile(x_train[0].reshape(28,28),augment_size).reshape(-1,28,28,1),
                                np.zeros(augment_size),batch_size=augment_size,shuffle=False).next()[0]
                                
print(x_augment.shape)

plt.figure(figsize=(10,10))
for c in range(100):
    plt.subplot(10,10,c+1),
    plt.axis('off'),
    plt.imshow(x_augment[c].reshape(28,28),cmap='gray')
plt.show()
'''    
img_generate=ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    shear_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
augment_size=30000
randidx=np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)
x_augment=x_train[randidx].copy() #복사본 준비, 원본 이미지 복사본 준비
y_augment=y_train[randidx].copy() 

x_augment = img_generate.flow(x_augment,np.zeros(augment_size),batch_size=augment_size,shuffle=False).next()[0]                  
print(x_augment.shape)
x_train=np.concatenate((x_train, x_augment)) #6만+3만
y_train=np.concatenate((y_train, y_augment)) #6만+3만
print(x_train.shape) #(90000,28,28,1)
print(x_train.shape) #(90000,10)



model=keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),padding='same', 
                               activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Dropout(rate=0.3),
    #하나의 레이어가 됨
    
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),padding='same', 
                               activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Dropout(rate=0.3),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(rate=0.3), # 과적합 방지 , rate는 내 마음대로 줘도 된다.
   
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(rate=0.3), # 과적합 방지 , rate는 내 마음대로 줘도 된다.
   
    keras.layers.Dense(units=10, activation='softmax'), #해보고 아닌거 같으면 레이어및 파라미터를 바꾸어가면서 해주면 된다 다양하기 때문 - 설계도가 잘만들어져야 한다.
]) 

print(model.summary())

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #sparsee가 categorical임

from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
MODEL_DIR='./model2/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelPath=MODEL_DIR+'{epoch:02d}-{val_loss:.4f}.keras'
chkPoint =ModelCheckpoint(filepath=modelPath, monitor='val_loss',verbos=1,save_best_only=True)


es=EarlyStopping(monitor='val_loss', patience=5)
history=model.fit(x_train,y_train,batch_size=64, epochs=500, verbose=2, validation_split=0.2, callbacks=[es,chkPoint])

print(history.history)

#모델 평가
train_loss, train_acc=model.evaluate(x_train,y_train)
test_loss,test_acc=model.evaluate(x_test,y_test)
print('train_loss, train_acc : ',train_loss,train_acc)
print('test_loss, test_acc : ', test_loss,test_acc)
print()

history = history.history

# 시각화 함수들
def plot_acc(title=None):
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_loss(title=None):
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title(title)
    plt.legend()
    plt.show()

# accuracy 시각화
plot_acc('Accuracy')

# loss 시각화
plot_loss('Loss')






