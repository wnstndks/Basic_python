# tf.constant() : 텐서를 직접 기억
# tf.Variable() : 텐서가 저장된 주소를 참조
import tensorflow as tf
import numpy as np

node1=tf.constant(3,dtype=tf.float32)
node2=tf.constant(4.0)
print(node1)
print(node2)
imsi=tf.add(node1,node2)
print(imsi) # 그래프 내에서 실행된 것이 아닌 파이썬 가상환경에서 실행된것 - 복잡한 작업이 수행될때 적합한 내용은 아님

print()
node3=tf.Variable(3,dtype=tf.float32)
node4=tf.Variable(4.0)
print(node3)
print(node4)
node4.assign_add(node3)
print(node4)

print()
a=tf.constant(5)
b=tf.constant(10)
c=tf.multiply(a,b) #병렬 연산을 하면서 행렬곱을 쓴다.
result=tf.cond(a<b,lambda : tf.add(10,c),lambda:tf.square(a))
print('result : ', result.numpy())

print('---')
v=tf.Variable(1)

@tf.function #Graph 환경에서 처리가 됨
def find_next_func():
    v.assign(v+1)
    if tf.equal(v%2, 0):
        v.assign(v+10)

find_next_func()
print(v.numpy())
print(type(find_next_func))
# <class 'function'>
# <class 'tensorflow.python.eager.polymorphic_function.polymorphic_function.Function'>

print('func1----------')
def func1():
    imsi=tf.constant(0)
    su=1
    for _ in range(3):
        imsi=tf.add(imsi,su)
    return imsi

kbs=func1()
print(kbs.numpy(),' = ', np.array(kbs))
#얘는 파이썬 작업 내에서 수행이 되는 것

print('\nfunc2----------')
imsi=tf.constant(0)
@tf.function
def func2():
    # imsi=tf.constant(0) imsi=0
    global imsi
    su=1
    for _ in range(3):
        # imsi=tf.add(imsi,su)
        # imsi=imsi+su
        imsi+=su
    return imsi

kbs=func2()
print(kbs.numpy(),' = ', np.array(kbs))
#얘는 그래프 내에서 수행되는 것, 위와 다름

print('\nfunc3----------')
imsi=tf.Variable(0)
@tf.function #Autograp에서는 constant()는 지역변수로 써도 되지만 variable()은 지역변수로 쓰면 안된다. 따라서 Variable()은 함수 밖에서 선언해주어야 한다.
def func3():
    # imsi=tf.Variable(0) 
    su=1
    for _ in range(3):
        imsi.assign_add(su) #variable은 누적으로 가주어야하기에 add가 아닌 assign_add
        # imsi=imsi+su 안됨
        # imsi+=su 안됨
    return imsi

kbs=func3()
print(kbs.numpy(),' = ', np.array(kbs))

print('구구단 출력 ------------')
# @tf.function #텐서를 연산하는데 연산 속도를 빠르게 하기 위해서 autograph를 쓰지만 안에서 형변환을 하면 안되고 연산을 하는 곳이기에 함수 안에서 출력을 하면 안된다.
def gugu1(dan):
    su=0
    for _ in range(9):
        su=tf.add(su,1)
        # print(su) #Autograp - good - 그냥 텐서임
        # print(su.numpy()) #tensor가 numpy로 바뀜 numpy로 형변환 하는 이유 : 일반 파일로 바꾸기 위해서 이 텐서를 ndarray로 형변환이 안됨
        print('{}*{}={}'.format(dan,su,dan*su))
    
gugu1(3)

print('---------')
#내장함수 : 일반적으로 numpy 지원함수를 그대로 사용. +알파
# ...중 reduce ~ 함수
ar= [[1,2],[3,4]]
print(tf.reduce_sum(ar).numpy())
print(tf.reduce_mean(ar,axis=0).numpy())
print(tf.reduce_mean(ar,axis=1).numpy())

#one_hot encoding
print(tf.one_hot([0,1,2,0],depth=3))
#리그레션이나 classification을 하면서 레이블 값을 유니크한 개수만큼 데이터를 잡아주고 나머지는 0으로 만드는 방법 구체적으로 얼마인지는 인덱스중에서 가장 큰 값으로 잡아주는 것










