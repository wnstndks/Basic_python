# 케라스 모델을 만드는 세가지 방법 - Sequential, Functional, Model Subclassing

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras import optimizers
import numpy as np
from keras.src.utils import generic_utils

#공부시간에 따른 성적 결과 예측
# x_data=[[1.],[2.],[3.],[4.],[5.]]
x_data=np.array([1,2,3,4,5],dtype=np.float32)
y_data=np.array([11,39,55,66,70],dtype=np.float32)
print(np.corrcoef(x_data,y_data)) #0.95469145

print(' 1) Sequential api 사용 : 가장 단순한 방법 - 레이어를 순서대로 쌓아 올리는 완전 연결층 모델을 생성')
model=Sequential() #sequential은 레이어 클래스를 쓰는것 튜플을 쓸수도 있다. input_dim 또는 shape
model.add(Dense(units=2,input_shape=(1,), activation='relu'))
model.add(Dense(units=1, activation='linear'))
# 연산한 값을 그냥 내보내는 것 linear를 썻기에 마지막에 그냥 빠져나가는 것
print(model.summary())

opti=optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opti,loss='mse',metrics=['mse'])
history=model.fit(x=x_data,y=y_data,batch_size=1,epochs=100,verbose=2)
loss_metrics=model.evaluate(x=x_data,y=y_data,batch_size=1,verbose=0)
print('loss_metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print('설명력 :', r2_score(y_data,model.predict(x_data)))
print('실제값 : ', y_data)
print('예측값 : ', model.predict(x_data).ravel())

new_data=[1.5,2.3,6.8,8.0]
print('새 예측값 : ', model.predict(new_data,verbose=0).flatten())

import matplotlib.pyplot as plt
plt.plot(x_data.flatten(),model.predict(x_data),'b',x_data.flatten(),y_data,'ko')
plt.show()

# mse의 변화량
plt.plot(history.history['mse'],label='mean squared error')
plt.xlabel('epoch')
plt.show()

print(' 2) functional api 사용 : 1번에 비해 유연한 구조를 설계한 방법')
from keras.layers import Input, Dense
from keras.models import Model

# 샘플 데이터 생성
x_data = np.array([[1.], [2.], [3.], [4.], [5.]], dtype=np.float32)
y_data = np.array([11, 39, 55, 66, 70], dtype=np.float32)

# 함수형 API를 이용한 모델 구성
inputs = Input(shape=(1,))
output1 = Dense(units=2, activation='relu')(inputs)
output2 = Dense(units=1, activation='linear')(output1)

model2 = Model(inputs, output2)

opti = optimizers.Adam(learning_rate=0.1)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])
history = model2.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=2)
loss_metrics = model2.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss_metrics:', loss_metrics)

# R-squared(결정 계수) 계산
print('설명력:', r2_score(y_data, model2.predict(x_data).ravel()))
print('실제값:', y_data)
print('예측값:', model2.predict(x_data).ravel())

# 서브클래싱(Subclassing) 방식을 이용한 모델 구성
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(2, activation='linear')
        self.d2 = Dense(1, activation='linear')

    def call(self, x):  # 처리 담당
        inputs = self.d1(x)
        return self.d2(inputs)

model3 = MyModel()

opti = optimizers.Adam(learning_rate=0.1)
model3.compile(optimizer=opti, loss='mse', metrics=['mse'])
history = model3.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=2)
loss_metrics = model3.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss_metrics:', loss_metrics)

# R-squared(결정 계수) 계산
print('설명력:', r2_score(y_data, model3.predict(x_data).ravel()))
print('실제값:', y_data)
print('예측값:', model3.predict(x_data).ravel())
print(model3.summary())


print('3) sub classing 사용2')
from keras.layers import Layer
class Linear(Layer):
    def __init__(self,units=1):
        super(Linear,self).__init__()
        self.units=units
        
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    @generic_utils.default
    def build(self, input_shape):
        #모델의 가중치 관련 작업을 기술할 수 있음
        self.w=self.add_weight(shape=(input_shape[-1],self.units), 
                               initializer='random_normal', trainable=True) #trainable= True -> 역전파 수행 여부를 선택하는 것
        self.b = self.add_weight(shape=(self.units), initializer='zeros', trainable=True) 
        #bias-절편값,(self.units) 반드시 출력개수와 같게
        Layer.build(self, input_shape)
        
    def call(self, inputs):
        #정의된 값들을 이용해 해당층의 로직을 수행
        return tf.matmul(inputs, self.w)+self.b # wx+b
        
class MlpModel(Model):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = Linear(2)
        self.linear2 = Linear(1)
        
    def call(self, inputs):
        # Layer의 build를 호출
        x=self.linear1(inputs)
        return self.linear2(x)
    
model4 = MyModel()

opti = optimizers.Adam(learning_rate=0.1)
model4.compile(optimizer=opti, loss='mse', metrics=['mse'])
history = model4.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=2)
loss_metrics = model4.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss_metrics:', loss_metrics)

# R-squared(결정 계수) 계산
print('설명력:', r2_score(y_data, model4.predict(x_data).ravel()))
print('실제값:', y_data)
print('예측값:', model4.predict(x_data).ravel())
print(model4.summary())        

        