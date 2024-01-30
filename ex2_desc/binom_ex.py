# 이항분포 : 정규분포는 연속변량인데 반해 이산변량을 사용
# 이항 검정 : 결과가 두가지 값을 가지는 확률변수의 분포를 판단하기에 효과적
# 예) 10명의 자격증 시험 합격자 주에서 여성이 6명이었다한다면 '여성이 남성보다 합격률이 높다'라고 할 수 있는가?

import pandas as pd
import scipy.stats as stats

# 직원을 대상으로 고객대응 교육을 실시하면, 고객 안내 서비스 만족율이 높아질까?
# 직원을 대상으로 고객대응 교육을 실시하면, 고객 안내 서비스 만족율이 높아질까?

data=pd.read_csv('../testdata/one_sample.csv')
print(data.head(3))
print(data['survey'].unique()) #[1,0]

ctab=pd.crosstab(index=data['survey'],columns='count')
ctab.index=['불만족','만족']
print(ctab)

print('양측검정 : 방향성이 없다. 기존 80% 만족율 기준으로 실시')
result=stats.binom_test([136, 14],p=0.8, alternative='two-sided') #stats.binorm_test(x:성공 또는 실패 횟수, N: 시도 횟수, p: 가설 확률)
print(result) #0.0006734701362867024 < 0.05
print()

result=stats.binom_test([14,136],p=0.2, alternative='two-sided')
print(result)
print()

print('단측검정 : 방향성이 있다(크다, 작다) . 기존 80% 만족율 기준으로 실시')
result=stats.binom_test([136,14],p=0.8, alternative='greater')
print(result)

result=stats.binom_test([14,136],p=0.2, alternative='less')
print(result)
