# 이미지 분류 모델 작성
# MNIST dataset( 흑백: 28*28 )을 사용

'''
Mnist 데이타셋에는 총 60,000개의 데이타가 있는데, 이 데이타는 크게 아래와 같이 세종류의 데이타 셋으로 나눠 집니다. 
모델 학습을 위한 학습용 데이타인 mnist.train 그리고, 학습된 모델을 테스트하기 위한 테스트 데이타 셋은 minst.test, 
그리고 모델을 확인하기 위한 mnist.validation 데이타셋으로 구별됩니다. 
각 데이타는 아래와 같이 학습용 데이타 55000개, 테스트용 10,000개, 그리고, 확인용 데이타 5000개로 구성되어 있습니다.
'''
#CNN은 정말 중요하다 , SVM 또한 이미지 분류를 할수 있다.

import tensorflow as tf
import sys
import numpy as np
import keras

(x_train, y_train),(x_test,y_test)= keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape,x_test.shape,y_test.shape) #(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(x_train[0]) # 0번쨰 feature
print(y_train[0]) # 0번쨰 label
print()

# for i in x_train[0]:
#     for j in i:
#         sys.stdout.write('%s  '%j)
#     sys.stdout.write('\n')
    #마치 전자시계와 비슷한 것

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0],cmap='gray')
# plt.show()

# 바탕은 검은색이고 250에 가까울수록 하얀색이된다.
# Dense는 일차원으로 만들어줌

print(x_train[0].shape)
x_train = x_train.reshape(60000,784).astype('float32') # 28 * 28 =>784
x_test = x_test.reshape(10000,784).astype('float32')
print(x_train, x_train[0].shape)
print()

#이미지는 표준화 정규화했을 떄 가장 잘나온다.

x_train /= 255.0 # 정규화하기 - 필수는 아니지만 권장은 함
x_test /=255.0
print(x_train[0]) # 데이터가 고르게 분포되게 만듦 - 학습하는데 유리
print()
# label은 one-hot 처리 해주어야한다 -> softmax를 쓰기 때문에
print(y_train[0])
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,num_classes=10)
print(y_train[0])
#label은 원핫 처리 한다.
print()

# train data의 일부(1만개)를 validation data로 사용
x_val=x_train[50000:60000]
y_val=y_train[50000:60000]
x_train=x_train[0:50000]
y_train=y_train[0:50000]
print(x_train.shape,x_val.shape)

# model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout

model=Sequential()
'''
네트워크 구성방법
model.add(Dense(units=128, input_shape=(784,))) # 병목현상에 빠질수 있다., reshape을 한 경우, 바꾸고 들어왔을떄 이걸 쓰기
# model.add(Flatten(input_shape=(28,28))) # reshape을 하지 않은 경우 안바꾸고 있을때는 이 두줄을 사용
# model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2)) - 80프로만 연산에 참여 20프로는 학습에 끼지 않음 노드가 10개 중 8개만 학습, 두개는 참여 x => 과적합 방지 - 서로 연결된 연결망에서 0부터 1사이의 확률로 뉴런을 제거하는 기법

model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=10))
model.add(Activation('softmax'))
마지막에는 dropout을 주면 안된다.

'''

model.add(Dense(units=128, input_shape=(784,), activation='relu'))
# model.add(Flatten(input_shape=(28,28))) #reshape을 안하고 있으니 빼기
model.add(Dropout(rate=0.2))
model.add(Dense(units=64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))) 
# keras.regularizers.l2(0.001) 정규화를 통해 학습시 가중치가 커지는 경우에 패널티를 준다. -> penalty를 부과하여 과적합 방지가 목적
model.add(Dropout(rate=0.2))
model.add(Dense(units=10, activation='softmax'))
#레이어마다 노드개수를 늘리면 연산량이 늘어난다. 그런데 빠름

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# 모든 파라미터가 train에 참여하고 있다.

history=model.fit(x=x_train,y=y_train, epochs=10, batch_size=128, validation_data=(x_val,y_val), verbose=2)
print('loss : ', history.history['loss'])
print('val_loss : ', history.history['val_loss'])
print('accuracy : ', history.history['accuracy'])
print('val_accuracy : ', history.history['val_accuracy'])

#시각화
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# 모델 평가
score = model.evaluate(x=x_test, y=y_test, batch_size=128, verbose=0)
print('score loss : ', score[0])
print('score accuracy : ', score[1])

model.save('tf9model.hdf5')

print('---------------')
mymodel = keras.models.load_model('tf9model.hdf5')

plt.imshow(x_test[:1].reshape(28, 28), cmap='gray')#784열짜리로 reshape을 시켰기에 여기서 다시 reshape 시켜야 함
plt.show()

print(x_test[:1])

import numpy as np
pred = mymodel.predict(x_test[:1])
print('예측값 : ', pred)
print('예측값 : ', np.argmax(pred, axis=1))
print('예측값 : ', y_test[:1])
print('예측값 : ', np.argmax(y_test[:1], axis=1))


