#재색인, bool 처리, 인덱싱 지원 함수
from pandas import Series,DataFrame

# Series 재색인
data=Series([1,3,2],index=(1,4,2))
print(data)

data2 = data.reindex((1,2,4)) #행 순서를 사용해 데이터 재배치
print(data2)

print('재배치할 때 값 끼워넣기')
data3=data2.reindex([0,1,2,3,4,5]) #대응 값이 없는 인덱스는 NaN(결치값)이 됨
print(data3) 

# NaN을 임의의 값으로 채우기
data3=data2.reindex([0,1,2,3,4,5],fill_value=555)
print(data3)
print()

#NaN을 이전(앞) 행 값으로 채우기
data3=data2.reindex([0,1,2,3,4,5],method='ffill')
print(data3)
data3=data2.reindex([0,1,2,3,4,5],method='pad') #상동
print(data3)
print()

#NaN을 다음(뒤) 행 값으로 채우기

data3=data2.reindex([0,1,2,3,4,5],method='bfill')
print(data3)
data3=data2.reindex([0,1,2,3,4,5],method='backfill') #상동
print(data3)

print('bool 처리')
import numpy as np
df= DataFrame(np.arange(12).reshape(4,3),index=['1월','2월','3월','4월'],columns=['강남','강북','서대문'])
print(df)
print(df['강남'])
print(df['강남']>3)
print(df[df['강남']>3])

print('인덱싱 지원 함수 : Loc() - 라벨지원, iloc():숫자 지원')
print(df.loc['3월',:])
print(df.loc['3월',])
print(df.loc[:'2월'])
print(df.loc[:'2월',['서대문']])
print()
print(df.iloc[2])
print(df.iloc[2,:])
print(df.iloc[:3])
print(df.iloc[:3,2]) #3행미만 행 2열
print(df.iloc[:3,1:3]) #3행미만, 1,2열 출력

print('Series 연산')
s1=Series([1,2,3],index=['a','b','c'])
s2=Series([4,5,6,7],index=['a','b','d','c'])
print(s1)
print(s2)
print(s1+s2) #index명 불일치인 경우는 NaN
print(s1.add(s2)) # numpy 함수를 계승
print(s1*s2)
print(s1.mul(s2))

print('Series 연산')
df1=DataFrame(np.arange(9).reshape(3,3),columns=list('kbs'),index=['서울','대전','부산'])
df2=DataFrame(np.arange(12).reshape(4,3),columns=list('kbs'),index=['서울','대전','제주','수원'])
print(df1)
print(df2)
print(df1+df2)
print(df1.add(df2))

# -, *, / 가능 sub,mul,div

print()
print(df1)
seri=df1.iloc[0]
print(seri)
print(df1-seri) #DataFrame -Series

print()
#결측치 관련 함수(메소드)
df=DataFrame([[1.4,np.nan],[7,-4.5],[np.NaN,],[0.5,-1]],
             columns=['one','two'])
print(df)
print(df.isnull())
print(df.notnull())
# print(df.drop(1)) #1행 삭제
print(df.dropna())
print(df.dropna(how='any'))
print(df.dropna(how='all'))
print(df.dropna(subset=['one'])) #특정 열에 NaN 있는 행 삭제
print(df.dropna(axis='rows'))
print(df.dropna(axis='columns'))
print()
print(df.fillna(0))

#기술 통계 관련 함수(메소드)
print(df)
print(df.sum())
print(df.sum(axis=0)) #열의 합, 상동
print(df.sum(axis=1)) #행의 합
print()
print(df.mean(axis=1))
print(df.mean(axis=1,skipna=True)) #NaN은 연산에서 제외
print(df.mean(axis=1,skipna=False))
print(df.mean(axis=0,skipna=True)) #열 단위

print()
print(df.describe()) #요약 통계량 출력
print(df.info()) #구조 출력



