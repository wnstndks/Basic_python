'''
이원카이제곱
동질성 검정 - 두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법이다. 두 집단 이상에서 각 범주(집단) 간의 비율이 서로
동일한가를 검정하게 된다. 두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법이다.
동질성 검정실습1) 교육방법에 따른 교육생들의 만족도 분석 - 동질성 검정 survey_method.csv
'''

import pandas as pd
import scipy.stats


data=pd.read_csv('../testdata_utf8/survey_method.csv')
print(data.head(5),data.shape)
print(set(data['method']),set(data['survey'])) #{1, 2, 3} {1, 2, 3, 4, 5}

# 귀무: 교육방법에 따른 교육생들의 만족도에 차이가 없다.
# 대립: 교육방법에 따른 교육생들의 만족도에 차이가 있다.

ctab=pd.crosstab(index=data['method'],columns=data['survey'])
ctab.columns=['매우만족','만족','보통','불만족','매우불만족']
ctab.index=['방법1','방법2','방법3']
print(ctab)

chi2, pvalue, _,_ = scipy.stats.chi2_contingency(ctab)
print('chi2:{}, pvalue:{}'.format(chi2,pvalue))
# chi2:6.544667820529891, pvalue:0.5864574374550608
# 해석: pvalue:0.5864574374550608 > 0.05이므로 귀무 채택, 교육방법에 따른 교육생들의 만족도에 차이가 있다.


print('****'*20)
'''
동질성 검정 실습2) 연령대별 sns 이용률의 동질성 검정
20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용 현황을 조사한 자료를 바탕으로 연령대별로 홍보
전략을 세우고자 한다.
연령대별로 이용 현황이 서로 동일한지 검정해 보도록 하자.
'''
data=pd.read_csv('../testdata_utf8/snsbyage.csv')
print(data.head(),data.shape)
print(data['age'].unique()) # [1,2,3]
print(data['service'].unique()) # ['F' 'T' 'K' 'C' 'E']

ctab=pd.crosstab(index=data['age'],columns=data['service'])
print(ctab)
chi2, pvalue, _,_ = scipy.stats.chi2_contingency(ctab)
print('chi2:{}, pvalue:{}'.format(chi2,pvalue))
# chi2:102.75202494484225, pvalue:1.1679064204212775e-18
# 해석: pvalue: 1.1679064204212775e-18 <0.05이므로 귀무기각
# 위 수집 데ㅣ터는 표본
# 만약에 위 수집 데이터가 모집단이라면 표본 추출해서 작업
sample_data=data.sample(n=200,replace=True)
print(sample_data.head(),sample_data.shape)






























