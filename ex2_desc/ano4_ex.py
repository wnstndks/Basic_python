# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

# 귀무: 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재한다.
# 대립: 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하지 않는다.

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import urllib.request
import pandas as pd

data = {
    'kind': [1, 2, 3, 4, 2, 1, 3, 4, 2, 1, 2, 3, 4, 1, 2, 1, 1, 3, 4, 2],
    'quantity': [64, 72, 68, 77, 56, np.nan, 95, 78, 55, 91, 63, 49, 70, 80, 90, 33, 44, 55, 66, 77]
}

df = pd.DataFrame(data)
df=df.fillna(df['quantity'].mean())
print(df,type(df)) 

# 분산분석의 전제조건 : 3가지를 충족할때 의미있음
# 독립성 : 각 집단은 서로 독립이어야 한다. (상관관계로 확인)
# 정규성 : 각 집단은 정규분포를 따라야 한다. shapiro, stats.ks_2samp (콜모고로프-스미르노프) 검정
# 불변성(등분산성) : 각 집단은 서로 분산이 일정해야 한다.

print(df['kind'].unique()) #[1 2 3 4]
result=df[['kind','quantity']]
r1 = result[result['kind'] ==1]
r2 = result[result['kind'] ==2]
r3 = result[result['kind'] ==3]
r4 = result[result['kind'] ==4]

quantity1=r1['quantity']
quantity2=r2['quantity']
quantity3=r3['quantity']
quantity4=r4['quantity']

print(np.mean(quantity1),np.mean(quantity2),np.mean(quantity3),np.mean(quantity4))

print('정규성 검정: 만족하면 anova, 만족하지 않으면 kruskal-wallis test')
print(stats.shapiro(quantity1).pvalue) 
print(stats.shapiro(quantity2).pvalue)
print(stats.shapiro(quantity3).pvalue) 
print(stats.shapiro(quantity4).pvalue) 
print(stats.ks_2samp(quantity1,quantity2).pvalue)
print(stats.ks_2samp(quantity1,quantity3).pvalue)
print(stats.ks_2samp(quantity1,quantity4).pvalue)
print(stats.ks_2samp(quantity2,quantity3).pvalue)
print(stats.ks_2samp(quantity2,quantity4).pvalue)
print(stats.ks_2samp(quantity3,quantity4).pvalue)
print()
'''
0.8680412769317627
0.5923926830291748
0.48601073026657104
0.4162167012691498
0.9307359307359307
0.9238095238095237
0.5523809523809524
0.9238095238095237
0.5523809523809524
0.7714285714285716
모두 > 0.05 이므로 정규성 만족 -> anova 사용
'''
print('등분산성 검정: 만족하면 anova, 만족하지 않으면 welch-anova')
print(stats.levene(quantity1,quantity2,quantity3,quantity4).pvalue) #모수검정 0.3268969935062273 >0.05이므로 등분산성 만족
print()

ctab=pd.crosstab(index=data['kind'],columns='count')
ctab.index=['기름1','기름2','기름3','기름4']
print(ctab)
print()

#anova_lm은 f통계량을 위해 회귀분석 결과를 사용
import statsmodels.api as sm

reg=ols('quantity~kind',data).fit()
reg=ols('df["quantity"]~df["kind"]',data=df).fit()
table=sm.stats.anova_lm(reg,typ=1)
print(table)
#pvalue= 0.428149>0.05 대립가설 기각 ->귀무: 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재한다.

#사후분석(post hoc) : ANOVA 검증 결과 유의미하다는 결론을 얻었을 때, 구체적으로 어떤 수준(들)에서 평균 차이가 나는지를 검증하는 방법
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tkResult = pairwise_tukeyhsd(endog=df.quantity, groups=df.kind)
print(tkResult)
print()

tkResult.plot_simultaneous(xlabel='mean',ylabel='group')
plt.show()


'''
[ANOVA 예제 2]
DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오. 
만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.
'''

import MySQLdb
import pickle
import sys

try:
    with open('mydb.dat',mode='rb') as obj:
        config=pickle.load(obj)
    
except Exception as e:
    print('연결 오류 :',e)
    sys.exit()
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = sql = """
        select jikwon_name ,buser_name,jikwon_pay
        from jikwon j join 
        buser b on b.buser_no=j.buser_num
    """
    cursor.execute(sql)
    df=pd.DataFrame(cursor.fetchall(),columns=['직원명','부서명','연봉'])
    jikwon=df.dropna()
    print(jikwon)
    
    j1=jikwon[jikwon['부서명']=='총무부']
    j2=jikwon[jikwon['부서명']=='영업부']
    j3=jikwon[jikwon['부서명']=='전산부']
    j4=jikwon[jikwon['부서명']=='관리부']
    print(j1,' ',np.mean(j1['연봉']))
    print(j2,' ',np.mean(j2['연봉'])) 
    print(j3,' ',np.mean(j3['연봉']))
    print(j4,' ',np.mean(j4['연봉']))

    print(stats.shapiro(j1['연봉']).pvalue)
    print(stats.shapiro(j2['연봉']).pvalue)
    print(stats.shapiro(j3['연봉']).pvalue)
    print(stats.shapiro(j4['연봉']).pvalue)

    '''
    0.02604489028453827
    0.025608452036976814
    0.4194071292877197
    0.9078023433685303
    -> 정규성을 모두 만족하지는 않는다.
    -> welch-anova 사용
    
    '''
    print()
    #등분산성    
    print(stats.levene(j1['연봉'],j2['연봉'],j3['연봉'],j4['연봉']).pvalue)
    # 0.7980753526275928>0.05
    
    from pingouin import welch_anova
    print(welch_anova(data=df, dv='연봉', between='부서명'))
    #0.797119  > 0.05 => 대립기각
    
except Exception as e:
    print('처리 오류 :',e)
finally:
    cursor.close()
    conn.close()

















