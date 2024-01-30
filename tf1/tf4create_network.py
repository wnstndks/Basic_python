# 모델을 만드는 건 분류가 편하다.
# keras 모듈(라이브러리)을 사용하여 네트워크 구성
# 논리회로 분류 모델

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam

# 1. 데이터 수집 및 가공
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

# 2. 모델 구성(설정)
# model=Sequential([ 차례대로 층을 찾는 것
#     Dense(input_dim=2,units=1),
#     Activation('sigmoid')
# ]) 

#2. 모델구성 방법 2
model=Sequential()
# model.add(Dense(units=1,input_dim=2))
# model.add(Activation('sigmoid'))
model.add(Dense(units=1,input_dim=2,activation='sigmoid')) #위 두줄 합친 것
# 이항분류일땐 sigmoid 다항분류일땐 소프트맥스
# 노드의 개수를 늘려주면 바로 해결되는

#3. 모델 학습 과정 설정(컴파일)
# model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['acc'])
# model.compile(optimizer='rmsprob', loss='binary_crossentropy',metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy',metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01,momentum=0.9), loss='binary_crossentropy',metrics=['accuracy']) #learning rate는 너무 크게 주면 안된다.
# model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy',metrics=['accuracy']) #learning rate는 너무 크게 주면 안된다.
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy',metrics=['accuracy']) #learning rate는 너무 크게 주면 안된다.
#optimizer : 입력데이터와 손실함수를 업데이트하는 매커니즘이다. 손실함수의 최소값을 찾는 알고리즘  #metric으로 msen값을 알수 있다.

#4) 모델 학습시키기(train) 더 나은 표현을 찾는 w를 갱신 자동화 과정
model.fit(x,y,epochs=500,batch_size=1, verbose=0)
# batch_size : 훈련데이터를 여러개의 작은 묶음(batch)으로 만들어 가중치(w)를 갱신할때 사용. 1 epoch시 사용하는 데이터셋의 크기, 안주면 batch_size=32로 자동적
#verbose=1을 주면 상세히 보여줌 verbose=2를 주면 적당히

#5) 모델 평가(tset) -evaluate
loss_metrics=model.evaluate(x,y,batch_size=1,verbose=0)
print(loss_metrics)

# 6. 학습결과 확인 : 예측값 출력
# pred=model.predict(x,batch_size=1,verbose=0)
pred=(model.predict(x)>0.5).astype('int32')
print('예측 결과 :',pred.flatten())

# 7. 모델저장
model.save('tf4model.h5') #hdf5

# 8. 모델읽기
from keras.models import load_model
mymodel= load_model('tf4model.h5')

mypred=(mymodel.predict(x)>0.5).astype('int32')
print('예측결과 :', mypred.flatten())