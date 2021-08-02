import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import urllib.request
import pandas as pd



def main(): # 제목

    total_list_title = []
    total_list_date = []


    start = 11
    while True:

        if start != 1921:
        # if start != 1921:
            # 삼성
            # url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90&sort=1&photo=3&field=0&pd=3&ds=2021.07.01&de=2021.07.31&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:from20210701to20210731,a:all&start={}'
            
            # 카카오
            # url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EC%B9%B4%EC%B9%B4%EC%98%A4&sort=1&photo=3&field=0&pd=3&ds=2021.07.01&de=2021.07.31&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:from20210701to20210731,a:all&start={}'

            #네이버
            url ='https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EB%84%A4%EC%9D%B4%EB%B2%84&sort=1&photo=3&field=0&pd=3&ds=2021.07.01&de=2021.07.31&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:from20210701to20210731,a:all&start={}'

            headers = {'User-Agent': 'Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57'}

            response = requests.get(url.format(start))

            soup = BeautifulSoup(response.text, 'html.parser') 

            title_list = [title['title'] for title in soup.find_all('a', attrs={'class' : 'news_tit'})]

            # print(title_list)

            dates = [date.get_text() for date in soup.find_all('span', attrs={'class':'info'})]
            date_list = dates[1::2]

            total_list_title.append(title_list)
            total_list_date.append(date_list)


            start += 10 
            print('크롤링 중.......')


        else:
            print('크롤링 완료!')
            print(total_list_title)
            print(total_list_date)
            print(start)
            print(len(total_list_title))
            print(len(total_list_date))
            print(int((start-1)/10)-1)
            print(len(total_list_date))
            total_list_date = [element for array in total_list_date for element in array]
            total_list_title = [element for array in total_list_title for element in array]
            print(total_list_date)
            
            df = pd.DataFrame({'title':total_list_title, 'date':total_list_date}, columns=['title', 'date'])
            print(df)

            date = str(datetime.now())
            date = date[:date.rfind(':')].replace(' ', '_')
            date = date.replace(':','시') + '분'    

            folder_path = os.getcwd()
            xlsx_file_name = '네이버뉴스_{}.xlsx'.format(date)

            df.to_excel(xlsx_file_name)

            print('엑셀 저장 완료 | 경로 : {}\\{}'.format(folder_path, xlsx_file_name))
            os.startfile(folder_path)

            break

        

    


main()