'''
교촌이나 bbq 홈페이지에서 
메뉴의 제목과 가격읽기

데이터프레임에 담고,
가격중 제일 비싼것과 싼것 
평균가격, 표준편차 구하기
'''

import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen

url = urlopen("https://www.kyochon.com/menu/chicken.asp")
chicken = BeautifulSoup(url.read(), 'html.parser')

chickens = chicken.select('#tabCont01 > ul > li') 

data = []

for i, chicken_item in enumerate(chickens):
    name = chicken_item.find('dt').text
    price = int(chicken_item.find('strong').text.replace(',', '')) 
    data.append([name, price])

df = pd.DataFrame(data, columns=['이름', '가격'])

print(df)
print()
max_price_chicken = df[df['가격'] == df['가격'].max()]
print('제일 비싼 치킨: {} 가격: {} 원'.format(max_price_chicken['이름'].values[0], max_price_chicken['가격'].values[0]))

# 제일 싼 치킨
min_price_chicken = df[df['가격'] == df['가격'].min()]
print('제일 싼 치킨: {} 가격: {} 원'.format(min_price_chicken['이름'].values[0], min_price_chicken['가격'].values[0]))

print('평균: ', df['가격'].mean())
print('표준편차: ', df['가격'].std())



