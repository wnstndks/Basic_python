# pandas로 파일 읽기

import numpy as np
import pandas as pd

# df=pd.read_csv('../testdata_utf8/ex1.csv') 
df=pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ex1.csv')  #웹에서 직접 가져올수도 있다. 그냥 그 페이지 주소를 읽으면 안되고 git같은 경우에는 raw버튼을 클릭하여 오리지날 데이터의 경로를 찍어야 한다. 
print(df,type(df)) # <class 'pandas.core.frame.DataFrame'>
print()

df=pd.read_table('../testdata_utf8/ex1.csv',sep=',') #read_table은 구분자를 주어야 한다. 
print(df.info())
print()

df=pd.read_csv('../testdata_utf8/ex2.csv',header=None,names=['col1','col2']) 
print(df)
print()

df=pd.read_csv('../testdata_utf8/ex2.csv',header=None,names=['a','b','c','d','msg'],index_col='msg') 
print(df)
print()

# df=pd.read_csv('../testdata_utf8/ex3.txt') #csv는 sep=,일 경우에만 쳐준다. 따라서 테이블로 읽어줌
df=pd.read_table('../testdata_utf8/ex3.txt',sep='\s') # sep=' ' 정규표현식에서 소문자 s는 빈칸 주기, 대문자 s는 빈칸 없애기=공백 문자가 아닌것 sep='정규표현식'
print(df)
print()
print(df.info())
print()
print(df.describe())
print()

df=pd.read_table('../testdata_utf8/ex3.txt',sep='\s+',skiprows=(1,3)) #skiprows = 특정행을 제외
print(df)
print()

df=pd.read_fwf('../testdata_utf8/data_fwt.txt',widths=(10,3,5),header=None,names=('date','name','price'),encoding='utf-8')
print(df)
print()

#대용량의 자료를 chunk(묶음) 단위로 할당해서 처리 가능
test=pd.read_csv('../testdata_utf8/data_csv2.csv',header=None,chunksize=3)
print(test) #TextFileReader object (텍스트 파서 객체)

for p in test:
    #print(p)
    print(p.sort_values(by=2,ascending=True))
    
print('\nDataFrame 저장')
items =  {'apple':{'count':10,'price':1500},'orange':{'count':5,'price':1000}}
df=pd.DataFrame(items)
print(df)
# print(df.to_html()) #html을 만들어서 넘김
# print(df.to_json())
# print(df.to_clipboard()) #출력방향이 다양 메모장에 편집-붙여넣기 하면 출력됨
# print(df.to_csv())
df.to_csv('test1.csv',sep=',')
df.to_csv('test2.csv',sep=',',index=False) # index 빠진것
df.to_csv('test3.csv',sep=',',index=False,header=False) # 헤더까지 빠진것
