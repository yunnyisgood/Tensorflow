import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import urllib.request

url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=삼성전자&sort=1&photo=3&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all&start=1](https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90&sort=1&photo=3&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all&start=1'
url = url.encode('UTF-8')

headers = {'User-Agent': 'Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57'}

soup = BeautifulSoup(requests.get(url).text, 'html.parser')

def main(): # 제목
    for i in soup.find_all('div',{'class' : 'news_area'}):
        title_list = [i.find('a', {'class' : 'news_tit'}).text]
        print(title_list)

    for i in soup.find_all('div',{'class' : 'info_group'}):
        date_list = [i.find('span', {'class' : 'info'}).text[1][0:]]
        print(date_list)


main()