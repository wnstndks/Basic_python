import numpy as np
import scipy.stats as stats
import pandas as pd

# 문제 1) 주체가 누구인지를 잘 보아야한다.
# 귀무 : 영사기에 사용되는 백열전구의 수명은 300시간이다.
# 대립 : 영사기에 사용되는 백열전구의 수명은 300시간이 아니다.

data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
print(np.array(data).mean())
print('\n정규성 확인 : ', stats.shapiro(data))

result = stats.ttest_1samp(data, popmean=300)
print(result)  
print('statistic(t값):%.5f, pvalue:%.5f'%result)  
# 해석 : pvalue:0.14361 > 0.05 이므로 귀무가설 채택, 수집된 자료는 우연히 발생된 것이라 할 수 있다.


print('\n------------')

# 문제 2)
# 귀무 : 노트북 평균 사용 시간이 5.2 시간이다.
# 대립 : 노트북 평균 사용 시간이 5.2 시간이 아니다.

data2 = pd.read_csv('../testdata/one_sample.csv')
data2['time'] = data2['time'].replace('     ', np.nan)
data2 = data2.dropna(subset=['time'])
data2['time'] = data2['time'].astype(float)
print(np.mean(data2['time']))
print(data2.head(3))

result2 = stats.ttest_1samp(data2.time, popmean=5.2)
print(result2) 
print('statistic(t값):%.5f, pvalue:%.5f'%result2) 
# 해석 : pvalue:0.00014 < 0.05 이므로 귀무가설 기각, 노트북 평균 사용 시간이 5.2 시간이 아니다.


print('\n------------')
# 문제 3)
# 귀무 : 정부에서는 전국 평균 미용 요금이 15000원이다.
# 대립 : 정부에서는 전국 평균 미용 요금이 15000원이 아니다.

data3 = pd.read_excel('beauty.xls')
data3 = data3.dropna(axis=1)
data3 = data3.drop(['번호', '품목'], axis=1)
print(data3.T)
print(np.mean(data3.T.iloc[:,0])) #18311.875

result3 = stats.ttest_1samp(data3.iloc[0], popmean=15000 )
print(result3)
print('statistic(t값) : %.5f, pvalue:%.5f'%result3)
# 해석 : pvalue: 0.00001 < 0.05이므로 귀무가설 기각. 전국 평균 미용 요금이 15000원이 아니다.
