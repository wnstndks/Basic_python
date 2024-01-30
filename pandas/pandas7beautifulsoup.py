# BeautifulSoup : HTML 및 XML 파일에서 데이터를 가져오는 Python 라이브러리
# 문서 내의 element(tag, 요소) 및 attribute(속성) 찾기 함수 : find(), find_all(), select_one, select

#json은 안되고 markup만 가능

import requests
from bs4 import BeautifulSoup

def go():

    base_url = "http://www.naver.com:80/index.html"
    
    source_code = requests.get(base_url)
    print(source_code) #<Response [200]>

    plain_text = source_code.text
    print(type(plain_text)) #<class 'str'>
     
    convert_data = BeautifulSoup(plain_text, 'lxml') # string, parse-> beautiful soup 객체로 바꾸기
    print(type(convert_data)) #<class 'bs4.BeautifulSoup'>
 
    for link in convert_data.findAll('a'): # 문서 내에서 a태그를 모두 잡아서 하나씩 내보내기
        href = base_url + link.get('href') # href 속성값에 url 더하기
        print(href)                          

go()
