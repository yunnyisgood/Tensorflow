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


news_url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={}&sort=1&photo=3&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all&start=1'

req = requests.get(news_url.format(query))
soup = BeautifulSoup(req.text, 'html.parser')


news_dict = {}
idx = 0
page_num = 1

print()
print('크롤링 중...')

while idx < news_num:
    
    table = soup.find('ul',{'class' : 'list_news'})
    li_list = table.find_all('li', {'id': re.compile('sp_nws.*')})
    area_list = [li.find('div', {'class' : 'news_area'}) for li in li_list]
    title_list = [area.find('a', {'class' : 'news_tit'}) for area in area_list]
    
    for n in title_list[:min(len(title_list), news_num-idx)]:
        news_dict[idx] = {'title' : n.get('title'),
                          'url' : n.get('href') }
        idx += 1

    page_num += 10

    pages = soup.find('div', {'class' : 'sc_page_inner'})
    next_page_url = [p for p in pages.find_all('a') if p.text == str(cur_page)][0].get('href')
    
    req = requests.get('https://search.naver.com/search.naver' + next_page_url)

    # pages = soup.find('div', {'class' : 'sc_page_inner'})
    # next_page_url = [p for p in pages.find_all('a') if p.text == str(cur_page)][0].get('href')
    page_num = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query=삼성전자&sort=0&photo=3&field=0&pd=2&ds=2021.07.02&de=2021.08.01&cluster_rank=17&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:1m,a:all&start={}'
    
    req = requests.get('https://search.naver.com/search.naver?where=news&sm=tab_jum&query={news_num}&sort=0&photo=3&field=0&pd=2&ds=2021.07.02&de=2021.08.01&cluster_rank=17&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:1m,a:all&start={}')
    soup = BeautifulSoup(req.text, 'html.parser')

print('크롤링 완료')

print('데이터프레임 변환')
news_df = DataFrame(news_dict).T

folder_path = os.getcwd()
xlsx_file_name = '네이버뉴스_{}_{}.xlsx'.format(query, date)

news_df.to_excel(xlsx_file_name)

print('엑셀 저장 완료 | 경로 : {}\\{}'.format(folder_path, xlsx_file_name))
os.startfile(folder_path)