import pandas as pd
import scipy.stats as stats

data1=pd.read_csv('../testdata_utf8/cleanDescriptive.csv') #표본자료
data1=data1.dropna()
print(data1.head(3),data1.shape) # (248, 14)
print()

print(data1['level'].unique()) # [1 2 3] 독립
print(data1['pass'].unique()) # [1 2 3] 종속
print()

#교차표 작성
ctab=pd.crosstab(index=data1['level'],columns=data1['pass'])
ctab.columns=['합격','실패']
print(ctab)
print()

chi2,p,dof,_=stats.chi2_contingency(ctab) 
print('chi2:{}, p:{}, dof:{}'.format(chi2,p,dof)) # chi2:2.7669512025956684, p:0.25070568406521365, dof:2


# 판정 : p:0.25070568406521365 > 0.05이기 때문에 유의한 수준(α=0.05)에서 대립가설을 기각한다.
# 부모학력 수준와 자녀의 진학여부 간에 관련이 있다.
print('---------------------------------------------------------')

# 카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
# 그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
#  예제파일 : MariaDB의 jikwon table 
#  jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#  jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
#  조건 : 부모학력 수준이 자녀의 진학여부와 관련이 있다.

import MySQLdb
import pickle
import sys

# 데이터 불러오기 및 가공
try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('읽기 오류 : ',e)
    sys.exit()
    
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = '''
        select jikwon_jik, jikwon_pay from jikwon
    '''
    cursor.execute(sql)
    data2 = pd.DataFrame(cursor.fetchall(), columns=['jikwon_jik', 'jikwon_pay'])
    # print(data2)
    
    jik_mapping = {
    '이사': 1,
    '부장': 2,
    '과장': 3,
    '대리': 4,
    '사원': 5
}
    data2['jikwon_jik'] = data2['jikwon_jik'].replace(jik_mapping)
    
    def map_jikwon_pay(pay):
        if 1000 <= pay <= 2999:
            return 1
        elif 3000 <= pay <= 4999:
            return 2
        elif 5000 <= pay <= 6999:
            return 3
        else:
            return 4

    data2['jikwon_pay'] = data2['jikwon_pay'].apply(map_jikwon_pay)

    # print(data2)]
    
except Exception as e:
    print('처리 오류 : ',e)
finally:
    cursor.close()
    conn.close()
    
# 귀무 :  A회사의 직급과 연봉은 관련이 없다. 
# 대립 :  A회사의 직급과 연봉은 관련이 있다. 

ctab2 = pd.crosstab(index=data2['jikwon_jik'], columns=data2['jikwon_pay'])
ctab2.index=['이사','부장','과장','대리','사원']
ctab2.columns = ['적음','보통','많음','아주많음']
print(ctab2)

chi2, p, dof, _ = stats.chi2_contingency(ctab2)
print('chi2:{}, p:{}, dof:{}'.format(chi2, p, dof))

# 판정 : p-value -> 0.00019211 < 0.05이기 때문에 유의미한 수준에서 귀무가설을 기각한다.
# 따라서 A회사의 직급과 연봉은 관련이 있다















