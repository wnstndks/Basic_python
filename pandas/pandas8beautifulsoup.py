#BeautifulSoup의 find, select 연습

from bs4 import BeautifulSoup
htmlData='''
<html>
    <body>
        <h1>제목 태그</h1>
        <p>문단 1</p>
        <p>문단 2</p>        
    </body>
</html>
'''

print(type(htmlData)) # <class 'str'> -> python에서 string 파일은 쓸수가 없다.

soup = BeautifulSoup(htmlData, 'html.parser') #BeautifulSoup 객체가 생성이 된다.->BeautifulSoup 모듈 지원 명령 사용 가능
print(type(soup))

h1=soup.html.body.h1
print("h1 : ",h1.text,' ',h1.string)

p1=soup.html.body.p
print('p1 : ',p1.text)

p2=p1.next_sibling.next_sibling
print('p2 : ',p2.text)

print('\n find() : 반환값 단수개 사용 ------')

htmlData2='''
<html>
    <body>
        <h1 id="title">제목 태그</h1>
        <p>문단 1</p>
        <p id="my" class="our">문단 2</p>        
    </body>
</html>
'''

soup2 = BeautifulSoup(htmlData2, 'html.parser') 
print(soup2.p,' ',soup2.p.string)
print(soup2.find('p').string)
print(soup2.find('p',id='my').string)
print(soup2.find(id='title').string)
print(soup2.find(id='my').string)
print(soup2.find(class_ ='our').string)
print(soup2.find(attrs ={'class':'our'}).string)
print(soup2.find(attrs ={'id':'my'}).string) #id를 받으면 단수로 받아온다. 위의 것들은 복수이지만 find를 썼기에 복수로 받아오지 않는다.


print('\n find_all(), findAll() : 반환값이 복수 사용 ------')
htmlData3='''
<html>
    <body>
        <h1 id="title">제목 태그</h1>
        <p>문단 1</p>
        <p id="my" class="our">문단 2</p>        
        <div>
            <a href="https://www.naver.com">네이버</a><br/> 
            <a href="https://www.daum.com">다음</a><br/> 
        </div>
    </body>
</html>
'''

soup3 = BeautifulSoup(htmlData3, 'lxml') 
print(soup3.find_all("a"))
print(soup3.find_all(["a","p"]))
print(soup3.findAll(["a"]))
print()
links=soup3.findAll(["a"])
print(links)
print()

for i in links:
    href = i.attrs['href']
    text=i.string
    print(href, ' ', text)

print('\n정규 표현식 사용')

import re
links2 = soup3.find_all(href=re.compile(r'^https'))
for j in links2:
    print(j.attrs['href'])

'''
print("\n 벅스 뮤직 상이트에서 곡 제목 읽기 - find 사용")
from urllib.request import urlopen
url= urlopen("https://music.bugs.co.kr/chart")
soup=BeautifulSoup(url.read(),'html.parser')
# print(soup)
musics= soup.find_all('td',class_="check")
# print(musics) #list형식으로 들어옴
for i,music in enumerate(musics):
    print('{}위 : {}'.format(i+1,music.input['title'])) #music은 input을 가지고 있기 떄문에
'''

print('\n select_one,select : css의 셀렉터를 사용 ------')
htmlData4='''
<html>
    <body>
        <div id="hello">
            <a href="https://www.naver.com">네이버</a><br/>
            <span> 
                <a href="https://www.daum.com">다음</a><br/> 
            </span>
            <b>
                <ul class="world">
                    <li>안녕</li>
                    <li>반갑다</li>
                </ul>
            </b>
        </div>
        <div id="hi" class="good"> 
            두번쨰 div 태그
        <div>
    </body>
</html>
'''

print('\n 단수 선택')
soup4 = BeautifulSoup(htmlData4, 'lxml')
kbs=soup4.select_one("div#hello >a") #id가 hello인 a태그
print('kbs : ',kbs,' ',kbs.string) # 태그 내의 정보 뽑아내기
kbs2=soup4.select_one("div.good") #class가 good인 a태그
print('\nkbs2 : ',kbs2,' ',kbs2.string) # 태그 내의 정보 뽑아내기

print('\n 복수 선택')
mbc=soup4.select("div#hello ul.world>li") #div 태그 중 id가 hello, 그런데 hello 내에는 ul이 포함되어있다. 그리고 '.'을 사용하면 자손 태그를 얻는다. '>' :이거는 직계로, 요소냐 속성이냐 구분
print('mbc : ',mbc)
for a in mbc:
    print(a.string,' ')

print()
msg=list()
for a in mbc:
    msg.append(a.string)
    
import pandas as pd
df=pd.DataFrame(msg,columns=['자료'])
print(df)
print(df.to_json())

