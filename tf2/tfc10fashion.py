# fashion_mnist : mnist 대신 사용 - 구조가 같다 아무거나 편리한걸 써주면 된다.
import tensorflow as tf
import keras
import sys
import numpy as np
import matplotlib.pyplot as plt
from tfpack2.tfc10stop import MyEarlyStop

(x_train, y_train),(x_test,y_test)= keras.datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape,x_test.shape,y_test.shape) # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# print(x_train[0]) # 0번쨰 feature
# print(y_train[0]) # 0번쨰 label 9
print()

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankel boot']
print(set(y_train))

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[y_train[i]])
plt.show()
'''

x_train=x_train/255.0
x_test=x_test/255.0

#print(x_train[0])

model= keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # 차원 축소 : 784로 변환
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
#각각의 layer마다 모델의 성능에 영향을 주는 애들 성능과 아주 밀접한 관련이 있다.
])

# Dense에 밀어넣기 전에 입력자료를 바꿔줌
# label은 당연히 점정수
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

from tfpack2.tfc10stop import MyEarlyStop
my_callback = MyEarlyStop()
model.fit(x_train,y_train, batch_size=128,epochs=500,verbose=1, callbacks=[my_callback]) #별도의 클래스를 만들어두고 메소드를 오버라이딩 시키기

test_loss, test_acc = model.evaluate(x_test,y_test)
print('test_loss : ', test_loss)
print('test_acc : ', test_acc)
print()

pred= model.predict(x_test, verbose=0)
print(pred[0])
print('예측값 :' , np.argmax(pred[0]))
print('실제값 : ', y_test[0])

# 실제 레이블과 예측 이미지 비교
def plot_image(i, pred_arr,true_label, img):
    pred_arr, true_label, img=pred_arr[i], true_label[i],img[i] #순서대로 받아야 하기에
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='Greys')
    
    pred_label=np.argmax(pred_arr)
    if pred_label == true_label:
        color='blue'
    else:
        color='red'
    
    plt.xlabel('{} {:2.0f}%({})'.format(class_names[pred_label],100*np.max(pred_arr), class_names[true_label],color=color))
    
i=0
plt.figure(figsize=(6,3))
plot_image(i, pred, y_test,x_test)
plt.show()

#하나만 하였지만 for문을 돌려 여러개의 사진을 맞추는 것도 볼수 있다.






