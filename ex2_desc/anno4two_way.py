# 이원 분산 분석(two way annova) - 요인이 복수 -> 각 요인의 레벨(그룹)도 복수
# 두 개의 요인에 대한 집단(독립변수) 각각이 종속변수에 평균에 영향을 주는 지 검정 
# 가설이 주 효과 2개, 교호작용 1개가 나옴

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
plt.rc('font',family='malgun gothic')

data=pd.read_csv('../testdata/group3_2.txt')

print(data.head(3),data.shape) #(36, 3)
print(data['태아수'].unique()) # [1 2 3]
print(data['관측자수'].unique()) # [1 2 3 4]

# 주 효과 가설 
# 귀무: 태아수와 태아의 머리둘레 평균은 차이가 없다.
# 대립: 태아수와 태아의 머리둘레 평균은 차이가 있다.
# 귀무: 관측자수와 태아의 머리둘레 평균은 차이가 없다.
# 대립: 관측자수와 태아의 머리둘레 평균은 차이가 있다.

# 교호작용 가설
# 귀무: 교호작용이 없다.(태아수와 관측자수는 관련이 없다. - 서로간의 독립적이다.)
# 대립: 교호작용이 있다.(태아수와 관측자수는 관련이 있다. - 서로 종속이다.)

# data.boxplot(column='머리둘레',by='태아수')
# plt.show()
# data.boxplot(column='머리둘레',by='관측자수')
# plt.show()

# reg = ols('머리둘레~C("태아수") + C("관측자수"),data=data).fit()
# reg = ols('머리둘레~C("태아수") + C("관측자수")+C("태아수"):C("관측자수")',data=data).fit()
reg = ols('머리둘레~C(태아수)*C(관측자수)',data=data).fit() #교호작용 확인O
result = anova_lm(reg,type=2)
print(result)

#             df      sum_sq     mean_sq            F        PR(>F)
# C(태아수)     2.0  324.008889  162.004444  2023.182239  1.006291e-32 < 0.05 귀무가설 기각 
# C(관측자수)    3.0    1.198611    0.399537     4.989593  6.316641e-03 > 0.05 귀무가설 기각 실패
# C(태아수):C(관측자수)   6.0    0.562222    0.093704     1.222222  3.295509e-01 >0.05 귀무가설 기각 실패

print()
print('---------------poison과 treat가 독퍼짐 시간의 평균에 영향을 주는가----------------------------')
data2=pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/poison_treat.csv',index_col=0)
print(data2.head(3),data2.shape) #(48, 3)

# 주 효과 가설 
# 귀무: poison 종류와 독퍼짐 시간의 평균에 차이가 없다.
# 대립: poison 종류와 독퍼짐 시간의 평균에 차이가 있다.
# 귀무: treat(응급처치) 방법과 독퍼짐 시간의 평균은 차이가 없다.
# 대립: treat(응급처치) 방법과 독퍼짐 시간의 평균은 차이가 있다.

# 교호작용 가설
# 귀무: 교호작용이 없다.(Poison 종류와 응급처치 방법은 관련이없다. - 서로간의 독립적이다.)
# 대립: 교호작용이 있다.(Poison 종류와 응급처치 방법은 관련이있다. - 서로 종속이다.)

# 데이터의 균형설계 확인
# ->표본수가 같은 지 다른지에 따라서 사후검정 효과에 차이가 있다. 또한 검정방법도 달라질 수 있다.

print(data2.groupby('poison').agg(len)) #독립변수 중 poison 요인으로 구분된 집단 표본수는 16으로 동일하다.
print(data2.groupby('treat').agg(len)) #treat 요인으로 구분된 집단 표본수는 12로 동일
print(data2.groupby(['poison','treat']).agg(len)) # 요인별 레벨의 표본수는 4로 동일
# 모든 집단별 표본 수가 동일하므로 균형설계가 잘되어있다.라고 말할 수 있다.

result2= ols('time~C(poison)*C(treat)',data=data2).fit()
print(anova_lm(result2))

#                       df    sum_sq   mean_sq          F        PR(>F)
# C(poison)            2.0  1.033012  0.516506  23.221737  3.331440e-07 <0.05 귀무 기각
# C(treat)             3.0  0.921206  0.307069  13.805582  3.777331e-06 <0.05 귀무 기각
# C(poison):C(treat)   6.0  0.250138  0.041690   1.874333  1.122506e-01 > 0.05 교호작용이므로 상호작용효과는 발견할 수 없다.























