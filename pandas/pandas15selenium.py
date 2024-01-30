# Selenium 툴을 이용해 브라우저를 통제. 웹 크롤링 가능

from selenium import webdriver
import time

'''
#다음으로 들어간것
browser = webdriver.Chrome()

browser.implicitly_wait(5)

browser.get('https://daum.net')

browser.quit()
'''

'''
#구글로 들어가 검색창에 파이썬을 친것
browser = webdriver.Chrome()  #Optional argument, if not specified will search path.

browser.get('http://www.google.com/xhtml');

search_box = browser.find_element("name", "q")

search_box.send_keys('파이썬')

time.sleep(5)          # Let the user actually see something!

browser.quit()
'''

#프로그램으로 화면 띄운거 캡쳐한 것
try:
    url = "https://www.daum.net"
    browser = webdriver.Chrome()
    browser.implicitly_wait(3)
    browser.get(url);
    browser.save_screenshot("daum_img.png")
    browser.quit()
    print('성공')
except Exception:
    print('에러')