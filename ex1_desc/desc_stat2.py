# 표준편차, 분산의 중요성: 평균은 같으나, 분산이 다름으로 인해 전체 데이터의 분포상태가 달라진다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(1) 
print(stats.norm(loc=1,scale=2).rvs(10)) #10개의 난수 발생

print('---------'*10)
centers=[1,1.5,2]
col='rgb'

std=2 #표준편차
datas=[]

for i in range(3):
    datas.append(stats.norm(loc=centers[i],scale=std).rvs(100))
    plt.plot(np.arange(100)+i*100,datas[i],'*',color=col[i]) #산포도 그리기

plt.show()
    
    
    