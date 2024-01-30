# CNN은 Convolutional Neural Networks의 약자로 딥러닝에서 주로 이미지나 영상 데이터를 처리할 때 쓰이며 이름에서 알수있다시피 
# Convolution이라는 전처리 작업이 들어가는 Neural Network 모델입니다.
# CNN은 특징 추출 알고리즘 사용: 이미지나 텍스트 데이터를 conv와 pooling을 반복하여 데이터량을 줄인 후, 완전 연결층으로 전달해 분류작업을 시행한다.

import tensorflow as tf
import sys
import numpy as np
import keras

(x_train, y_train),(x_test,y_test)= keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape,x_test.shape,y_test.shape) #(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(x_train[0]) # 0번쨰 feature
print(y_train[0]) # 0번쨰 label
print()

# CNN은 채널(channel)을 사용함 - 그림의 바탕 => 삼차원을 4차원으로 변환을 해주어야 한다.
x_train= x_train.reshape((60000, 28, 28,1)) #(60000, 28, 28,1) -> 몇개인지를 정확하게 써도 되고 아니면 (-1, 28, 28,1) 이렇게 컴퓨터가 판단하도록 할수도 있다. 
x_test= x_test.reshape((10000,28,28,1)) #(-1, 28, 28,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000,28,28,1)
print(x_train[:1]) #[[[[  0]

x_train=x_train/255.0 #5값을 4차원으로 바꾼것
x_test=x_test/255.0

#모델 (CNN : 고해상도, 크기가 큰 이미지를 전처리 후 작은 이미지로 변환을 하고 변환 후 ==> Dense(완전 연결층으로 전달)로 분류를 진행함
input_shape =(28,28,1) #너비와 높이 채널을 인덱스로 넣어줌

print('방법1. Sequential API 사용')
model= keras.models.Sequential()

# Conv2D(필터수, 필터크기, 스트라이드(필터이동량), 패딩여부,...)
# padding='valid' : 원본이미지 밖에 0으로 채우기를 안함, same : 0으로 채우기를 함
# CONV-합성곱 : 필터를 이미지 일부분과 픽셀끼리 곱한 후 결과를 더한 것
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1),padding='valid', #커널과 원본 이미지가 각 셀마다 연산을 함->원본의 크기를 특징을 유지하면서 줄임
                               activation='relu', input_shape=input_shape)) #이미지는 2d, 동영상은 3d, 이 두줄이 컨볼루션임, 그래서 뒤에서 쌓아두게 된다.
# 2중은 절반으로 만들어진다. 이미지의 크기가 클때는 스트라이프로 해도 좋음
# 한칸씩만 해도 좋다. C

# Pooling : 이미지 특징을 유지한 채, 크기를 줄임. noise를 제거할 수 있음 -> noise 제거 효과도 발휘함
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))  # 이거는 선택사항 옵션임 필수는 아님
model.add(keras.layers.Dropout(rate=0.3)) # 과적합 방지, 이 두줄은 CNN의 한쌍

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),padding='valid', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))  # 이거는 선택사항 옵션임 필수는 아님
model.add(keras.layers.Dropout(rate=0.3)) #값들을 막 넣었다 뺏다 해주기에 동적인것 값들을 변형시켜야 하기에 하기는 좀 힘들다.

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='valid', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))  # 이거는 선택사항 옵션임 필수는 아님
model.add(keras.layers.Dropout(rate=0.3)) #값들을 막 넣었다 뺏다 해주기에 동적인것 값들을 변형시켜야 하기에 하기는 좀 힘들다.

model.add(keras.layers.Flatten()) #FCLayer(Fully Connected Layer) : 이미지를  , 이미지를 1차원으로 만들어줌
#reshape을 쓸필요없이 flip 하고 convert만 써주면 된다.

# Dense -> 패턴을 보고 분류
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dropout(rate=0.2)) # 과적합 방지 , rate는 내 마음대로 줘도 된다.
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dropout(rate=0.2)) # 과적합 방지 , rate는 내 마음대로 줘도 된다.
model.add(keras.layers.Dense(units=10, activation='softmax'))

print(model.summary())

#원본을 줄이고 성능이 매우 우수함

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=3)
history=model.fit(x_train,y_train,batch_size=128, epochs=100, verbose=1, validation_split=0.2, callbacks=[es])

print(history.history)

#모델 평가
train_loss, train_acc=model.evaluate(x_train,y_train)
test_loss,test_acc=model.evaluate(x_test,y_test)
print('train_loss, train_acc : ',train_loss,train_acc)
print('test_loss, test_acc : ', test_loss,test_acc)
print()
print('예측값 : ', np.argmax(model.predict(x_test[:1])))
print('예측값 : ', np.argmax(model.predict(x_test[:1])))
print('실제값 : ', y_test[0])


#-------------------
import pickle
import matplotlib.pyplot as plt

history = history.history

# history 객체를 pickle을 사용하여 저장
with open('tfc12his.pickle', 'wb') as obj:
    pickle.dump(history, obj)

# history 객체를 pickle을 사용하여 불러오기
with open('tfc12his.pickle', 'rb') as obj:
    history = pickle.load(obj)
#----------------------
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

# 모델 저장
model.save('tfc12model.h5')
#---------------
#새 이미지 분류 작업......
















