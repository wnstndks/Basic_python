#내가 그린 숫자 이미지 모델에 분류 요청하기
from PIL import Image #이미지 확대 축소 라이브러리(모듈)
import numpy as np
import matplotlib.pyplot as plt

#내 이미지 출력
im= Image.open('number.png')
img=np.array(im.resize((28,28), Image.LANCZOS).convert('L')) #흑백 Graysccale , 'L' : 그레이스케일, '1' : 이진화 , 'RGB' ...
#LANCZOS: 높은 해상도의 사진 또는 영상을 낮은 해상도로 변환하거나 나타낼때 깨진 패턴의 형태로 나타나게 되는데 이를 최소화 시키는 방법
print(img)
print(img.shape)
print()

# plt.imshow(img, cmap='Greys')
# plt.show()

#이미지를 모델이 맞추는 지 확인
data= img.reshape([1,784])
data=data / 255.0 
print(data)

#학습이 끝난 모델로 내 이미지를 판별
import keras
mymodel = keras.models.load_model('tf9model.hdf5')
pred = mymodel.predict(data)
print('pred : ', pred)
print('pred : ', np.argmax(pred,axis=1))

# 컨볼루션안쓰고 dense만 쓰고 있음 그러나 이러면 연산량이 많아지기에 고해상도 및 칼라 이미지는 저해상도로 떨어트리고 이미지의 크기를 최대한 줄여야하기에 이거를 convolution이 해주는 것- 원래 이미지의 특징만 가져오는 것 
# 모든 이미지는 일차원으로 줄이고 연산량을 줄여야한다.
# CPU만 가지고는 딸린다.
