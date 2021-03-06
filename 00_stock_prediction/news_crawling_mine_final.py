import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import urllib.request
import pandas as pd
import numpy as np



def main(): # 제목

    total_list_title = []
    total_list_date = []


    start = 11
    while True:

        if start != 1921 :

            # 삼성
            url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90&sort=0&photo=3&field=0&pd=3&ds=2021.02.01&de=2021.02.28&cluster_rank=24&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from20210201to20210228,a:all&start={}'

            headers = {'User-Agent': 'Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57'}

            response = requests.get(url.format(start))

            soup = BeautifulSoup(response.text, 'html.parser') 
            
            # 기사 헤드라인 추출
            title_list = [title['title'] for title in soup.find_all('a', attrs={'class' : 'news_tit'})]

            # 날짜 추출
            dates = [date.get_text() for date in soup.find_all('span', attrs={'class':'info'})]
            date_list = dates[1::2] # 다른 부분까지 같이 추출되기 때문에 

            total_list_title.append(title_list)
            total_list_date.append(date_list)

            start += 10 
            print('크롤링 중.......')

        else:
            print('크롤링 완료!')

            total_list_date = [element for array in total_list_date for element in array]
            total_list_title = [element for array in total_list_title for element in array]
            print(total_list_date)
            
            df = pd.DataFrame({'title':total_list_title, 'date':total_list_date}, columns=['title', 'date'])

            # DataFrame으로 변환 후 데이터 전처리
            def clean_text(text): # 특수 문자 제거 
                cleaned_text = re.sub('[-=+,#/\?:^$.@*\"*~&%`!\\`|\(\)\[\]\<\>`\'...>]', '', text) 
                cleaned_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", text)
                return cleaned_text

            df['title'] = df['title'].dropna(axis=0).apply(lambda x : clean_text(x))
            df['date'] = df['date'].dropna(axis=0) 

            df = df.sort_values(by=['date'], axis=0) # 날짜 순으로 정렬

            print(df)

            # 엑셀 파일로 데이터 프레임 저장
            date_time = str(datetime.now())
            date_time = date_time[:date_time.rfind(':')].replace(' ', '_')
            date_time = date_time.replace(':','시') + '분'

            folder_path = os.getcwd()
            xlsx_file_name = '네이버뉴스_2월{}.xlsx'.format(date_time)

            df.to_excel(xlsx_file_name)

            print('엑셀 저장 완료 | 경로 : {}\\{}'.format(folder_path, xlsx_file_name))
            os.startfile(folder_path)


            break

        

    


main()