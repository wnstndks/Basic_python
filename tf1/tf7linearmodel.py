# 단순 선형회귀 모델 작성
# keras의 내장 api를 사용 - Sequential : 다음번 예제
# GradientTape 객체를 이용해 모델을 구현 - 유연하게 복잡한 로직을 처리할 수 있음
# TensorFlow는 GradientTape을 이용하여 즉시 실행 모드 eager execution mode 에서 쉽게 오차 역전파를 수행할 수 있다.
# w를 갱신하고 미분을 통해 최적의 접점을 찾을 수 있다. w값을 찾아냄

import keras
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam

tf.random.set_seed(2)
x=tf.Variable(5.0)
w=tf.Variable(tf.random.normal((1,)))
b=tf.Variable(tf.random.normal((1,)))
print(w.numpy(),b.numpy())
print()

opti=keras.optimizers.SGD() #RMSProp, Adam

@tf.function
def trainModel(x,y):
    with tf.GradientTape() as tape:
        hypo=tf.add(tf.multiply(w,x),b) #wx+b
        loss=tf.reduce_mean(tf.square(tf.subtract(hypo,y))) #cost function
    grad=tape.gradient(loss,[w,b]) #자동 미분 계산(loss를 w와 b로 미분)
    opti.apply_gradients(zip(grad,[w,b]))    #zip: 튜플 형태로 차례로 접근할 수 잇는 반복자(iterator)를 반환
    return loss

x=[1.,2.,3.,4.,5.]
y=[1.2,2.0,3.0,3.5,5.5]
print(np.corrcoef(x,y)) #선형회귀 모델은 인과관계를 확인하기 위함


w_val=[]
cost_val=[]

for i in range(1,101):
    loss_val=trainModel(x, y)
    cost_val.append(loss_val.numpy())
    w_val.append(w.numpy())
    if i % 10 ==0:
        print(loss_val)
    
print('cost_val : ',cost_val)
print('w_val : ',w_val)

import matplotlib.pyplot as plt
plt.plot(w_val,cost_val, 'o')
plt.xlabel('w')
plt.ylabel('cost')
plt.show()

print('cost가 최소일 때 w값: ',w.numpy())
print('cost가 최소일 때 b값: ',b.numpy())

y_pred=tf.multiply(x,w)+b #선형회귀 식이 만들어짐
print('예측값 :',y_pred.numpy())

plt.plot(x,y,'ro',label='real y')
plt.plot(x, y_pred, 'b-', label='real y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#새 값으로 예측하기
new_x=[3.5,9.0]
new_pred=tf.multiply(new_x,w)+b
print('예측 결과값 : ', new_pred.numpy())
