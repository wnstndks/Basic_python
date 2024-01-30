# 일원분산분석
# 강남구에 있는 GS 편의점 3개지역 알바생의 급여에 대한 평균에 차이가 있는가?

# 귀무: 강남구에 있는 GS 편의점 3개지역 알바생의 급여에 대한 평균에 차이가 없다.
# 대립: 강남구에 있는 GS 편의점 3개지역 알바생의 급여에 대한 평균에 차이가 있다.

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import urllib.request
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm


url='https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3.txt' 
# data=pd.read_csv(url,header=None)
# print(data.head(3),data.shape)
# data=data.values #DataFrame to ndarray

data=np.genfromtxt(urllib.request.urlopen(url),delimiter=',')
print(data,type(data)) #<class 'numpy.ndarray'>
print(data.shape) # (22,2)

gr1=data[data[:,1]==1, 0]
gr2=data[data[:,1]==2, 0]
gr3=data[data[:,1]==3, 0]
print(gr1,' ',np.mean(gr1)) # 316.6
print(gr2,' ',np.mean(gr2)) # 256.4
print(gr3,' ',np.mean(gr3)) # 278.0

# 정규성
print(stats.shapiro(gr1).pvalue)
print(stats.shapiro(gr2).pvalue)
print(stats.shapiro(gr3).pvalue)

# 등분산성
# print(stats.levene(gr1,gr2,gr3)) #0.0458
print(stats.bartlett(gr1,gr2,gr3)) #0.3508 만족    이유) 표본의 개수가 몇개 안되기에 bartlett을 사용하는게 맞다

# 데이터의 퍼짐정도 시각화 ->boxplot사용
plt.boxplot([gr1,gr2,gr3],showmeans=True)
plt.show()

print()
print('일원분산분석 처리 방법1.')
df=pd.DataFrame(data,columns=['pay','group'])
print(df)
print()
lmodel=ols('pay ~ C(group)',data=df).fit() #C(독립변수) : 변수가 범주형임을 표시
print(anova_lm(lmodel,type=1)) #0.043589 < 0.05 이므로 귀무가설 기각

print()
#일원분산분석 처리 방법2.
f_statistic,p_value=stats.f_oneway(gr1,gr2,gr3)
print('f_statistic:{}, p_value:{}'.format(f_statistic,p_value))
#f_statistic:3.7113359882669763, p_value:0.043589334959178244
#GS 편의점 3개 지역 알바생의 급여에 대한 평균에 차이가 있다.

#사후분석(post hoc) : ANOVA 검증 결과 유의미하다는 결론을 얻었을 때, 구체적으로 어떤 수준(들)에서 평균 차이가 나는지를 검증하는 방법
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tkResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tkResult)

tkResult.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show()














