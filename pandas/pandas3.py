# DataFrame : reshape, cut, merge, pivot
import numpy as np
import pandas as pd

df=pd.DataFrame(1000+np.arange(6).reshape(2,3),index=['대전','서울'],columns=['2021','2022','2023'])
print(df)
df_row=df.stack() #재구조화 열 -> 행으로 변환 ->칼럼 쌓기

print()
print(df_row)

print()
df_col=df_row.unstack() #행 ->열로 변환
print(df_col)

print()
print('범주화 : 연속형 자료를 범주형으로 변경')
price=[10.3,5.5,7.8,3.6]
cut=[3,7,9,11] #구간 기준값

result_cut=pd.cut(price,cut)
print(result_cut) #(9, 11] : 9초과 11이하 9 < x <= 11
print()
print(pd.value_counts(result_cut))
print()
datas=pd.Series(np.arange(1,1001))
print(datas.head(3))
print(datas.tail(3))
print()

result_cut2 = pd.qcut(datas,3) #datas 값을 3개 영역으로 범주화
print(result_cut2)
print()
print(pd.value_counts(result_cut2))
print()
cut2=[1,500,1000]
result_cut3 =pd.cut(datas,cut2)
print(result_cut3)
print()
print(pd.value_counts(result_cut3))
print()
print('그룹별 함수 수행 : agg,apply')
group_col=datas.groupby(result_cut2)
print(group_col.agg(['count','mean','std','max']))
print()

#agg 대신 함수 직접 작성
def summary_fuc(gr):
    return {
        'count':gr.count(),
        'mean':gr.mean(),
        'std':gr.std(),
        'max':gr.max(),
        }
    
print(group_col.apply(summary_fuc)) #apply->함수를 실행하는 것
print(group_col.apply(summary_fuc).unstack())
print()
#웹을 만들면 브라우저 통신이 가능한 어느누구에게든 통신이 가능하다- 파이썬 데이터 분석결과를 장고로 보여주기 위해서

print('\n병합(merge)')
df1=pd.DataFrame({'data1':range(7),'key':['b','b','a','c','a','a','b']})
print(df1)
print()

df2=pd.DataFrame({'key':['a','b','d'],'data2':range(3)})
print(df2)
print()
print(pd.merge(df1,df2)) #key를 기준으로 inner join
print()
print(pd.merge(df1,df2,on='key',how='inner')) #key를 기준으로 inner join, 상동
print()
print(pd.merge(df1,df2,on='key',how='outer')) #key를 기준으로 outer join
print()
print(pd.merge(df1,df2,on='key',how='left')) #key를 기준으로 left outer join
print()
print(pd.merge(df1,df2,on='key',how='right')) #key를 기준으로 right outer join
print()

print('공통 칼럼이 없는 경우 : df1 vs df3')
print()
df3=pd.DataFrame({'key2':['a','b','d'],'data2':range(3)})
print(df3)
print()
print(pd.merge(df1,df3,left_on='key',right_on='key2'))
print()
print('\nDataFrame 자료 이어붙이기')
print(pd.concat([df1,df3],axis=0)) #행 단위- default
print()
print(pd.concat([df1,df3],axis=1)) #열 단위
print()
print('Series 병합')
s1=pd.Series([0,1],index=['a','b'])
s2=pd.Series([2,3,4],index=['c','d','e'])
s3=pd.Series([5,6],index=['f','g'])
print(pd.concat([s1,s2,s3],axis=0))

print('그룹 연산 : pivot table - 데이터 열 중에서 두개의 열(key)을 사용해 데이터를 재구성하여 새로운 집계 테이블을 작성할 수 있다.')

data={'city':['강남','강북','강남','강북'],
      'year':[2000,2001,2002,2002],
      'population':[3.3,2.5,3.0,2.0]
}

df=pd.DataFrame(data)
print(df)
print()
print(df.pivot(index='city',columns='year',values='population')) #행 열,연산데이터
print(df.pivot(index='year',columns='city',values='population')) #행 열,연산데이터
print()
hap=df.groupby(['city'])
print(hap)
print()
print(df.groupby(['city']).sum()) #위 두줄을 한줄로 표현
print()
print(df.groupby(['city']).agg('sum'))
print(df.groupby(['city']).agg(['sum','mean']))
print()
print(df.groupby(['city','year']).mean()) 
print()
print('DataFrame.pivot_table : pivot과 groupby 명령의 중간적 성격')
print(df)
print()
print(df.pivot_table(index=['city'])) # 기본 연산은 np.mean()
print()
print(df.pivot_table(index=['city'],aggfunc=np.mean)) #상동
print()
print(df.pivot_table(index=['city','year'],aggfunc=[len,np.sum]))
print()
print(df.pivot_table(values=['population'],index='city'))
print()
print(df.pivot_table(values=['population'],index='city',aggfunc=np.mean)) #상동
print()
print(df.pivot_table(values=['population'],index=['year'],columns=['city']))
print()
print(df.pivot_table(values=['population'],index=['year'],columns=['city'],margins=True)) #행과 열의 합
print()
print(df.pivot_table(values=['population'],index=['year'],columns=['city'],margins=True,fill_value=0)) #행과 열의 합, NaN을 0으로 채움
print()














