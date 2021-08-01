import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

date = str(datetime.now())
date = date[:date.rfind(':')].replace(' ', '_')
date = date.replace(':','시') + '분'



query = input('검색 키워드를 입력하세요 : ')
news_num = int(input('총 필요한 뉴스기사 수를 입력해주세요(숫자만 입력) : '))
query = query.replace(' ', '+')


news_url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={}'

req = requests.get(news_url.format(query))
soup = BeautifulSoup(req.text, 'html.parser')


news_dict = {}
idx = 0
cur_page = 1

print()
print('크롤링 중...')

def main():
    while idx < news_num:

        table = soup.find('ul',{'class' : 'list_news'})
        li_list = table.find_all('li', {'id': re.compile('sp_nws.*')})
        date_list = [li.find('span', {'class' : 'info'}) for li in li_list]
        title_list = [i.find('a', {'class' : 'news_tit'}) for i in soup.find_all('div',{'class' : 'news_info'})]


    
    