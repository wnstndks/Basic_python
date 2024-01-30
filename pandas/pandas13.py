# 기상청 제공 중기 예보 웹 문서(XML) 읽기
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

url="https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"
data=urllib.request.urlopen(url).read()
# print(data)

soup=BeautifulSoup(data,'lxml')
title=soup.find('title').string
print(title)
print()
wf=soup.find('wf') #wf란 태그 읽기
# print(wf)
city=soup.find_all('city')
print(city)
cityDatas=[]
#열 단위로 데이터프레임에 넣는 방법
for c in city:
    cityDatas.append(c.string)

df=pd.DataFrame()
df['city']=cityDatas
# print(df)

# tempMins=soup.select('location>province+city+data>tmef') #+(next sibling), -(previous sibling) location의 직계자신=province, province의 형제 city data-> 그 다음 자식으로 tmEf
# print(tempMins) #tmef의 경우 소문자로 되어있기 때문에 소문자로 바꾸어주어야 한다-태그의 경우 소문자가 기분이다.
tempMins=soup.select('location>province+city+data>tmn')
tempDatas=[]
for t in tempMins:
    tempDatas.append(t.string)

df['temp_m']=tempDatas
df.columns=['지역','최저기온']
print(df)


    
