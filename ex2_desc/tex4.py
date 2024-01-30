# 독립표본 t검정
# 어느 음식점의 매출 데이터와 날씨 데이터 두 개의 파일을 이용해 강수 여부에 따른 음식점 매출액 평균의 차이를 검정
# 귀무: 강수 여부에 따른 음식점 매출액 평균의 차이는 없다.
# 대립: 강수 여부에 따른 음식점 매출액 평균의 차이는 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats

#매출 데이터
sales_data=pd.read_csv('../testdata/tsales.csv',dtype={'YMD':'object'})
print(sales_data.head(3))
print(sales_data.info())
print()

#날씨 데이터
wtdata=pd.read_csv('../testdata/tweather.csv')
print(wtdata.head(3)) # tm : 2018-06-01 
print(wtdata.info())
wtdata.tm=wtdata.tm.map(lambda x:x.replace('-',''))
print(wtdata.head(3)) #tm:20180601
print()

# merge
frame = sales_data.merge(wtdata,how='left',left_on='YMD',right_on='tm')
print(frame.head(3))
print(frame.tail(3),frame.shape)
print(frame.columns) #'YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes'
print()

data=frame.iloc[:,[0,1,7,8]] #YMD(날짜) AMT(매출액)  maxTa(최고기온)  sumRn(강수량)
print(data.head(3))
print(data.isnull().sum())
print()

print('--two sample ttest---'*10)
# print(data['sumRn']>0)
# data['rain_yn']=(data['sumRn']>0).astype(int)
# print(data.head(3))

# print(True*1,False*1)
data['rain_yn']=(data.loc[:,('sumRn')]>0)*1
print(data.head(3))

# 매출액 시각화
import matplotlib.pyplot as plt

sp=np.array(data.iloc[:,[1,4]])
print(sp) #[[ 0 0] [ 18000 ] ]
tg1=sp[sp[:,1]==0,0] #집단1: 비 안 올떄 매출액
tg2=sp[sp[:,1]==1,0] #집단1: 비 올떄 매출액
print(tg1[:3])
print(tg2[:3])

# plt.plot(tg1)
# plt.show()
# plt.plot(tg2)
# plt.show()

plt.boxplot([tg1,tg2],notch=True,meanline=True,showmeans=True)
plt.show()

print(np.mean(tg1),' ',np.mean(tg2)) #761040.2542372881   757331.5217391305

#정규성
print(len(tg1),len(tg2))
print(stats.shapiro(tg1).pvalue)
print(stats.shapiro(tg2).pvalue)

#등분산성
print(stats.levene(tg1,tg2).pvalue)

print(stats.ttest_ind(tg1,tg2, equal_var=True)) #TtestResult(statistic=0.10109828602924716, pvalue=0.919534587722196, df=326.0)
#해석: pvalue=0.91953>0.05이므로 귀무가설 채택(귀무가설 기각 실패)

