# 추론 통계 분석 중 가설검정 - 단일표본 t 검정(one sample t-test)
# 정규분포(모집단)의 표본에 대해 기댓값을 조사(평균의 차이를 이용한다)하는 검정 방법
# ex) 새우깡 과자 무게가 진짜 120g이 맞는가?

# 실습1. 어느 남성 집단의 평균 키 검정
# 귀무: 남성의 평균키는 177.0(모집단의 평균)이다.
# 대립: 남성의 평균키는 177.0(모집단의 평균)이 아니다. 

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

one_sample=[167.0, 182.7, 169.6, 176.8, 185.0]
# plt.boxplot(one_sample)
# plt.xlabel('data')
# plt.ylabel('height')
# plt.grid()
# plt.show()

print(np.array(one_sample).mean()) #176.2199
print(np.array(one_sample).mean()- 177.0) #-0.78 -> 177.0 과의 차이가 정말 의미가 있는지 여부를 알아봐야 한다.
print('정규성 확인 : ',stats.shapiro(one_sample)) #95%신뢰구간 내에서 pvalue=0.5400515794754028 >0.05 ==>정규성 만족 => 귀무채택
result=stats.ttest_1samp(one_sample, popmean=177.0)
print(result) #(statistic=-0.22139444579394396, pvalue=0.8356282194243566, df=4)-> ttest에서는 모양이 정규분포이기에 카이제곱과는 다르게 음수가 나올 수 있다.
print()
print('statistic(t값): %.5f, pvalue: %.5f'%result)
#statistic(t값): -0.22139, pvalue: 0.83563 ->해석 : pvalue:0.83563 >0.05이므로 귀무가설을 채택한다. 수집된 자료는 우연히 발생된 것이라 할 수 있다.
print()

# 실습 예제 2)
# A 중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균검정) student.csv
# 학생들의 국어 점수 평균은 80.0이다. - 귀무가설
# 학생들의 국어 점수 평균은 80.0이 아니다 - 대립가설

data=pd.read_csv('../testdata_utf8/student.csv')
print(data.head(3))
print(data.describe())
print()
print(np.mean(data.국어)) # 72.9 : 80.0 과의 차이가 있다고 봐야할까?
#검정 시작 

result2=stats.ttest_1samp(data.국어, popmean=80.0) #popmean= 예상 평균값
print(result2) # (statistic=-1.3321801667713216, pvalue=0.19856051824785262, df=19)-> ttest에서는 모양이 정규분포이기에 카이제곱과는 다르게 음수가 나올 수 있다.
print()
print('statistic(t값): %.5f, pvalue: %.5f'%result2)
# statistic(t값): -1.33218, pvalue: 0.19856 ->해석 : pvalue:0.19856 >0.05이므로 귀무가설을 채택한다. 수집된 자료는 우연히 발생된 것이라 할 수 있다. 학생들의 국어 점수 평균은 80.0이다.
print()


# 실습 예제 2)
# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자

#귀무: 여아 신생아의 몸무게는 평균이 2800(g)이다.
#대립: 여아 신생아의 몸무게는 평균이 2800(g)보다 크다.
data2=pd.read_csv('../testdata_utf8/babyboom.csv')
print(data2.head(3),len(data2)) #44
print()
fdata=data2[data2['gender']==1]
print(fdata, len(fdata)) #18
print()
print(np.mean(fdata['weight'])) #3132.44
print()

# 정규성 확인 수치
print(stats.shapiro(fdata.iloc[:, 2])) #pvalue=0.017984 <0.05이므로 정규성 만족 못함

# 정규성 확인 - 시각화1
stats.probplot(fdata.iloc[:, 2], plot=plt) # Q-Q plot
plt.show()

# 정규성 확인 - 시각화2 : histogram
sns.displot(fdata.iloc[:,2],kde=True)
plt.show()
#왼쪽으로 꼬리가 긴 형태- 편향되어 있다.

result3=stats.ttest_1samp(fdata.weight, popmean=2800) #popmean= 예상 평균값
print(result3) # (statistic=2.233187669387536, pvalue=0.03926844173060218, df=17)
print()
print('statistic(t값): %.5f, pvalue: %.5f'%result3)
# statistic(t값): 2.23319, pvalue: 0.03927->해석 : pvalue:0.03927 < 0.05이므로 대립가설을 채택한다. =여아 신생아의 몸무게는 평균이 2800(g)보다 크다. 
# 수집된 자료는 우연히 발생된 것이라 할 수 없다. 

print()














