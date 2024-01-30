import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam

# 1. 데이터 수집 및 가공
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1]) #xor

#2. 모델구성 방법 2
model=Sequential()
# model.add(Dense(units=1,input_dim=2,activation='sigmoid')) 

# model.add(units=1,input_dim=2)
# model.Add(Activation('relu')) #relu L Sigmoid와 tanh가 갖는 Gradiant problem 문제를 해결하기 위한 함수이다.
#기울기 소실 문제는 back Propagation에서 계산 결과와 정답과의 오차를 통해 가중치를 수정하는데,
#입력층으로 갈수록 기울기가 작아져 가중치들이 업데이트 되지 않아 최적의 모델을 찾을 수 없는 문제
# model.add(units=1)
# model.Add(Activation('sigmoid'))

#활성화 함수 -sigmoid 

model.add(Dense(units=5,input_dim=2,activation='relu')) 
model.add(Dense(units=5,activation='relu')) 
model.add(Dense(units=1,activation='sigmoid')) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(x,y,epochs=100, batch_size=1,verbose=1)
loss_metrics=model.evaluate(x,y)

print(loss_metrics)

pred=(model.predict(x)>0.5).astype('int32')
print('예측 결과 :',pred.flatten())

print(model.summary())

print()
print(model.input)
print(model.output)
print(model.weights) #weight는 가중치 기울기 값 bias도 보여줌
print('***'*20)
print(history.history['loss'])
print(history.history['accuracy'])

#시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='train loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],label='train accuracy')
plt.xlabel('epochs')
#skitlearn도 한번에 하면 안됨
plt.legend()
plt.show()

import pandas as pd
pd.DataFrame(history.history)['loss'].plot(figsize=(8,5))
plt.show()
