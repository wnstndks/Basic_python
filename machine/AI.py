import numpy as np

from sklearn.linear_model import LinearRegression

#훈련 데이터
x=np.array([[1],[2],[3],[4],[5]])
y=np.array([1,2,3,4,5])

#모델 훈련
reg=LinearRegression().fit(x,y)

#새로운 값 예측
test_x=np.array([[6]])
pred_y=reg.predict(테스트_x)

print('x=6에 대한 예측:',pred_y[0])