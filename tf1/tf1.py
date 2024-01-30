'''
텐서플로(TensorFlow)는 구글(Google)에서 만든, 딥러닝 프로그램을 쉽게 구현할 수 있도록 다양한 기능을 제공해주는 라이브러리다.  
텐서플로 자체는 기본적으로 C++로 구현 되어 있으며, 아래의 그림과 같이 Python, Java, Go 등 다양한 언어를 지원한다. 
하지만,  파이썬을 최우선으로 지원하며 대부분의 편한 기능들이 파이썬 라이브러리로만 구현되어 있어 Python에서 개발하는 것이 편하다.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
print(tf.__version__)
print('GPU 사용 가능' if tf.test.is_gpu_available() else 'GPU 사용불가능')
print(tf.config.list_physical_devices('GPU'))
print()

print('tensor : 수치용 컨테이너다. 임의의 차원을 가지는 행렬의 일반화된 모습이다. 계산 그래프 구조를 가진다. 병렬 연산이 기본')
print(1,type(1)) #python 1 <class 'int'>
print(tf.constant(1),type(tf.constant(1))) #tf.Tensor(1, shape=(), dtype=int32) <class 'tensorflow.python.framework.ops.EagerTensor'>  0-d tensor : scalar
print(tf.constant([1]),' ',tf.rank(tf.constant(1))) # 1-d tensor :vector
print(tf.constant([[1]])) # 2-d tensor :matrix
print()

a=tf.constant([1,2])
b=tf.constant([3,4])
c=a+b
print(c)
d=tf.add(a,b)
print(d)
print()
print(7)
print(tf.convert_to_tensor(7, dtype=tf.float32))
print(tf.cast(7, dtype=tf.float32))
print(tf.constant(7.0))
print(tf.constant(7,dtype=tf.float32))
print()

import numpy as np
arr=np.array([1,2])
print(arr,type(arr))
tfarr=tf.add(arr,5)
print(tfarr)
print(tfarr.numpy()) #자동으로 ndarray로 형변환이 이루어진다.tensorflow 형태로 바뀌어 연산가능, 텐서플로의 연산은 가급적 텐서플로우를 가지고 연산하기
print(np.add(tfarr,3))
print()
