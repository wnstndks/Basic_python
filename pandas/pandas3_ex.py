#Python Analysis - p10.

#문제1.

import numpy as np
from pandas import DataFrame,Series
import pandas as pd

a = np.random.randn(9, 4)
print(a)
print()

df= DataFrame(a,columns=['No1', 'No2', 'No3', 'No4'])
print(df)
print()

print(df['No1'].mean())
print(df['No2'].mean())
print(df['No3'].mean())
print(df['No4'].mean())




# 문제 2.
df = pd.DataFrame(np.array([10, 20, 30, 40]).reshape(4, 1), columns=['numbers'], index=['a', 'b', 'c', 'd'])
print(df)

print()
print(df.loc['c'])
print()
print(df.loc[['a','d']])
print()
print(df['numbers'].sum())
print()
print(df.loc['a':'d']**2)



df['floats'] = [1.5, 2.5, 3.5, 4.5]
print(df)
print()

s=Series(['길동','오정','팔계','오공'],index=['d','a','b','c'])
print(s)
print()

s2 = s.reindex(('a','b','c','d'))
print(s2)
print()

df['names'] = s2
print(df)

