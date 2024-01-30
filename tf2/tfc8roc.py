# iris dataset 사용
# 3개의 분류모델 성능 출력 : ROC curve

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder, StandardScaler
#도커는 원래 리눅스에서 하는 것 윈도우에서도 리눅스 환경을 만들수 있다. 클라우드 환경을 만들때 도커를 사용한다. vmware
from sklearn.datasets import load_iris

iris=load_iris()
# print(iris.DESCR)
print(iris.keys())
x=iris.data
print(x[:2])
y=iris.target
print(y)
print(set(y)) #{0, 1, 2}
names=iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# one hot : keras to_categorical, numpy가 지원하는 np.eye(), pandas가 지원하는 get_dummies(), sklearn이 지원하는 onehotencoder를 쓸게요
onehot=OneHotEncoder(categories='auto')
print(y.shape)
y=onehot.fit_transform(y[:,np.newaxis]).toarray()
print(y.shape) #(150,3)
print(y[:3]) 

#feature에 대해서 표준화/정규화 - 안할수도 있으나 이걸 할시 일반적으로 성능이 향상됨
scaler=StandardScaler()
x_scale=scaler.fit_transform(x)
print(x_scale[:3])

#train/test split
x_train,x_test,y_train,y_test = train_test_split(x_scale,y,test_size=0.3, random_state=1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
n_features=x_train.shape[1]
n_classes = y_train.shape[1]
print(n_features,' ',n_classes) #4 3

# n의 개수만큼 모델 생성 함수 - 메세지를 보내는 것
def create_model_func(input_dim,output_dim,n, out_nodes, model_name='model'):
    # print(input_dim,output_dim, out_nodes,n, model_name)
    def create_model():
        model=Sequential(name=model_name)
        for _ in range(n): #입력층
            model.add(Dense(units=out_nodes, input_dim=input_dim,activation='relu')) #은닉층임 

        model.add(Dense(units=output_dim, activation='softmax')) #출력층
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model
    return create_model #함수에서 내부함수의 주소를 리턴하고 있음- 클로저 -> 모델스가 가지게 된다. javascript의 클로저
        
        
models=[create_model_func(n_features,n_classes, 10, n, 'model_{}'.format(n)) for n in range(1,4)]
#모델을 생성하는 것

for cre_model in models:
    print()
    cre_model().summary()
    
print()
history_dict={}

for cre_model in models:
    model=cre_model()
    print('model_name : ', model.name)
    historys=model.fit(x=x_train, y=y_train, batch_size=5,epochs=50,verbose=0,validation_split=0.3)
    score= model.evaluate(x=x_test, y=y_test, verbose=0)
    print('test loss', score[0])
    print('test accuracy : ', score[1])
    history_dict[model.name]=[historys,model] #history는 0번째 있고, model은 1번째에 있다.
    
print(history_dict)

# 시각화
'''
fig, (ax1,ax2)= plt.subplots(2,1,figsize=(8,6))

for model_name in history_dict:
    print('h_d : ', history_dict[model_name][0].history['acc'])
    val_acc=history_dict[model_name][0].history['val_acc']
    val_loss=history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('validation acc')
    ax2.set_ylabel('validation loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
plt.show()
'''

# ROC curve
plt.figure()
plt.plot([0,1],[0,1],'k--')

from sklearn.metrics import roc_curve, auc

for model_name in history_dict:
    model=history_dict[model_name][1]
    y_pred = model.predict(x_test)
    fpr,tpr,_=roc_curve(y_test.ravel(), y_pred.ravel())
    plt.plot(fpr, tpr, label='{}, auc value:{:.3f}'.format(model_name, auc(fpr,tpr)))
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend()
plt.show()








