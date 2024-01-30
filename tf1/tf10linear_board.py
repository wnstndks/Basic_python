# 다중선형회귀 분석
# tensorboard : 머신러닝 실험을 위한 시각화 툴킷(toolkit)이다
# 텐서보드를 사용하면 손실 및 정확도와 같은 측정 항목을 추적 및 시각화하는 것, 모델 그래프를 시각화하는 것, 히스토그램을 보는것
# 이미지를 출력하는 것 등이 가능하다

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping


# 5명이 치른 세 번의 시험점수로 다음 번 시험 점수 예측
x_data = np.array([[70, 85, 80], [71, 89, 78], [50, 85, 60], [55,25,50],[50,35,10]])
y_data = np.array([73,82,72,50,34])

print('1) Sequential api ---')
model = Sequential()
model.add(Dense(units=6, input_dim=3, activation='linear', name='a'))  # name을 줘야 보드를 만들 때 편함
model.add(Dense(units=3, activation='linear', name='b'))
model.add(Dense(units=1, activation='linear', name='c'))
print(model.summary())

opti = optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])
# cb_early_stopping = EarlyStopping(monitor='val_loss', patience=100)
# history = model.fit(x_data, y_data, batch_size=10, epochs=5000, verbose=0, callbacks=[cb_early_stopping])
history = model.fit(x_data, y_data, batch_size=1, epochs=50, verbose=2)

# plt.plot(history.history['loss'])
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

loss_metrics = model.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('loss_metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data)))

print('2) functional api ---')
from keras.layers import Input
from keras.models import Model

inputs =  Input(shape=(3,))
output1 = Dense(6, activation='linear', name='a')(inputs)
output2 = Dense(3, activation='linear', name='b')(output1)
output3 = Dense(1, activation='linear', name='c')(output2)
model2 = Model(inputs, output3)
print(model2.summary())

opti = optimizers.Adam(learning_rate=0.01)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])

# tensorboard
from keras.callbacks import TensorBoard
tb = TensorBoard(
    log_dir='./my',
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1
)

history = model2.fit(x_data, y_data, batch_size=1, epochs=50, verbose=1, callbacks=[tb])
print(history.history['loss'])

loss_metrics = model2.evaluate(x_data, y_data, batch_size=1, verbose=0)
print('loss_metrics : ', loss_metrics)
print('설명력 : ', r2_score(y_data, model2.predict(x_data)))

# 새로운 값 예측
x_new = np.array([[30, 35, 30], [5, 7, 88]])
print('예상점수 : ', model2.predict(x_new).flatten())




'''
# 다중선형회귀분석
# TensorBoard : 머신러닝 실험을 위한 시각화 툴킷(toolkit)입니다. 
# TensorBoard를 사용하면 손실 및 정확도와 같은 측정 항목을 추적 및 시각화하는 것, 모델 그래프를 시각화하는 것, 히스토그램을 보는 것, 이미지를 출력하는 것 등이 가능합니다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers # 모두 다 경사하강법을 쓰고 있다. 그러나 gradientdesc을 쓰진 안씀, SGD를 쓴다 그러나 SGD의 경우 모멘텀의 문제가 있다. -> 이를 개선하기 위해 RMSP가 있고 이 장점을 포함하여 Adam을 사용한다.
from cookiecutter import log

# 5명이 치른 세번의 시험점수로 다음 번 시험 점수 예측하기
x_data = np.array([[70,85,80],[71,89,78],[50,85,60],[55,25,50],[50,35,80]])
y_data = np.array([73,82,72,50,34])

print('1) Sequential api ---')
model=Sequential() #keras를 써주면 단순해진다.
model.add(Dense(units=6, input_dim=3, activation='linear', name='noname1')) #한글을 왜 안받지
model.add(Dense(units=3, activation='linear', name='noname2'))
model.add(Dense(units=1, activation='linear', name='noname3'))
#linear란 들어온값을 sigmoid를 태우지 않고 그냥 내보내는 것
print(model.summary())
print()

opti=optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])
history=model.fit(x_data,y_data, batch_size=1,epochs=50,verbose=2)
print(history.history['loss'])

# plt.plot(history.history['loss']) #loss를 볼것
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# 학습이 랜덤하게 실행되기에 학습할 때마다 계속 저장을 하고 가자 이상적인 데이터를 가지고 모델로 써야한다. - 계속 바뀌어야 한다.
# 저장된 모델을 가지고 predict 해야함, 더 이상의 학습은 낭비


loss_metrics=model.evaluate(x_data,y_data,batch_size=1,verbose=0) #evaluatesms fit이랑 batch size 똑같이
print('loss_metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data))) #설명력은 0에서부터 1사이에 있기에 모델 성능 기준을 보여준다.- but 정확하지 않기에 참고용


print('2) functional api ---')
from keras.layers import Input
from keras.models import Model

inputs= Input(shape=(3,))
output1=Dense(6,activation='linear',name='noname1')(inputs)
output2=Dense(6,activation='linear',name='noname2')(output1)
output3=Dense(6,activation='linear',name='noname3')(output2)
model2=Model(inputs,output3)
print(model2.summary())

opti=optimizers.Adam(learning_rate=0.01)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])

#TensorBoard
from keras.callbacks import TensorBoard

tb= TensorBoard(
    log_dir='./my',
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1
)

history=model2.fit(x_data,y_data, batch_size=1,epochs=50,verbose=1, callbacks=[tb])
print(history.history['loss'])

loss_metrics=model2.evaluate(x_data,y_data,batch_size=1,verbose=0) #evaluatesms fit이랑 batch size 똑같이
print('loss_metrics : ', loss_metrics)
print('설명력 : ', r2_score(y_data, model2.predict(x_data))) #설명력은 0에서부터 1사이에 있기에 모델 성능 기준을 보여준다.- but 정확하지 않기에 참고용

# 새로운 값 예측
x_new=np.array([[30,35,30],[5,7,88]])
print('예상 점수 : ', model2.predict(x_new).flatten())

'''