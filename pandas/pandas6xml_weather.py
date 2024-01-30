# https://www.kma.go.kr/XML/weather/sfc_web_map.xml #기상청이 실시간으로 데이터를 갱신해줌
# 기상청 제공 국내 주요 지역 날씨 정보 xml 문서 읽기
import urllib.request
import xml.etree.ElementTree as etree

# 웹 문서를 읽어, 파일로 저장 후 XML 문서 처리
try:
    webdata =urllib.request.urlopen("https://www.kma.go.kr/XML/weather/sfc_web_map.xml")
    print(webdata) #<http.client.HTTPResponse object at 0x0000025BF6E5B040>-> 객체로 나옴->읽어주어야 한다.
    webxml = webdata.read() #데이터 인코딩
    webxml = webxml.decode('utf-8') #데이터 디코딩
    # print(webxml)
    webdata.close() #할일 마치고 자원 반납
    with open('pandas6.xml',mode='w',encoding='utf-8') as obj:
        obj.write(webxml)
    print("success")
except Exception as e:
    print('err : ',e)


xmlfile=etree.parse('pandas6.xml')
print(xmlfile) #ElementTree object
root=xmlfile.getroot()
print(root.tag)
print(root[0].tag) #weather

children = root.findall('{current}weather')
print(children) #Element 객체가 잡힘

for it in children:
    y=it.get('year') # 속성값 읽기
    m=it.get('month')
    d=it.get('day')
    h=it.get('hour')

print()
print(y+"년 "+m+"월 "+d+"일 " +h+"시 현재 날씨 정보")


datas=[] #데이터를 담아둘 그릇 만들기

for child in root:
    # print(child.tag) # {current}weather
    for it in child:
        # print(it.tag) # {current}local
        localName=it.text # 지역명
        re_ta=it.get("ta") #속성 값 얻기
        re_desc=it.get("desc")
        # print(localName,re_ta,re_desc)
        datas += [[localName,re_ta,re_desc]] #list에는 += 을 할수 있다 
# print(datas) ['동두천', '18.8', '맑음'], ['파주', '18.2', '흐림'], ['대관령', '16.6', '맑음'], ['춘천', '19.7', '맑음'] ...
import pandas as pd
import numpy as np

df=pd.DataFrame(datas,columns=['지역','온도','상태'])
print(df.head(3))
print(df.tail(3))

imsi=np.array(df.온도)
imsi=list(map(float,imsi))
# print(imsi) #['20.7' '19.8' '18.7'
print('평균온도 : ',round(np.mean(imsi),2))
    
    















