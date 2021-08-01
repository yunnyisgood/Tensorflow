import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import urllib.request
import re

# 기간은 최근 1개월, 보도자료 옵션으로 삼성전자, 카카오, 네이버를 검색한 링크를 입력한다
# url = input('url을 입력하세요')
# # 삼성 url = 'https://search.daum.net/search?w=news&DA=STC&enc=utf8&cluster=y&cluster_page=1&q=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90&period=m&sd=20210701211349&ed=20210801211349&p=1&article_type=report'
# # 카카오  url = 'https://search.daum.net/search?nil_suggest=btn&w=news&DA=STC&q=%EC%B9%B4%EC%B9%B4%EC%98%A4&period=m&sd=20210701222847&ed=20210801222847&p=1&article_type=report'
# # 네이버 url = 'https://search.daum.net/search?nil_suggest=btn&w=news&DA=STC&q=%EB%84%A4%EC%9D%B4%EB%B2%84&spacing=3&orgq=spdlqj&period=m&sd=20210701222940&ed=20210801222940&p=1&article_type=report'

# headers = {'User-Agent': 'Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57'}

# response = requests.get(url, headers=headers)

# soup = BeautifulSoup(response.text, 'lxml')


# def get_title(): # 제목
#     for i in soup.find_all('div',{'class' : 'wrap_cont'}):
#         title_list = [i.find('a', {'class' : 'tit_main ff_dot'}).text]
#         print(title_list)

    
# def get_date():
#     dates = [date.get_text() for date in soup.find_all('span', attrs={'class':'cont_info'})]
#     date_list = []
#     match = re.search(r'\d{4}.\d{2}.\d{2}', str(dates))
#     for date in dates:
#         if match !=None:
#             date = datetime.strptime(match.group(), '%Y.%m.%d').date()
#             date = date.strftime("%Y-%m-%d")
#             date_list.append(date)
#     print(date_list)

def main():

    

# 삼성 
    url = 'https://search.daum.net/search?w=news&DA=STC&enc=utf8&cluster=y&cluster_page=1&q=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90&period=m&sd=20210701211349&ed=20210801211349&p={}&article_type=report'
# 카카오  url = 'https://search.daum.net/search?nil_suggest=btn&w=news&DA=STC&q=%EC%B9%B4%EC%B9%B4%EC%98%A4&period=m&sd=20210701222847&ed=20210801222847&p={}&article_type=report'.format(start)
# 네이버 url = 'https://search.daum.net/search?nil_suggest=btn&w=news&DA=STC&q=%EB%84%A4%EC%9D%B4%EB%B2%84&spacing=3&orgq=spdlqj&period=m&sd=20210701222940&ed=20210801222940&p={}&article_type=report'.format(start)

    headers = {'User-Agent': 'Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57'}

    start = 0
    result_df = pd.DataFrame()

    while True:
        try:
            start +=1
            response = requests.get(url.format(start), headers=headers)

            soup = BeautifulSoup(response.text, 'lxml') 
            # title
            for i in soup.find_all('div',{'class' : 'wrap_cont'}):
                title_list = [i.find('a', {'class' : 'tit_main ff_dot'}).text]


            dates = [date.get_text() for date in soup.find_all('span', attrs={'class':'cont_info'})]
            date_list = []
            match = re.search(r'\d{4}.\d{2}.\d{2}', str(dates))
            for date in dates:
                if match !=None:
                    date = datetime.strptime(match.group(), '%Y.%m.%d').date()
                    date = date.strftime("%Y-%m-%d")
                    date_list.append(date)
            
            df = pd.DataFrame({'title': title_list, 'date':date_list})
            result_df = pd.concat([result_df, df], ignore_index=True)

            start +=1
            

        except:
            print(start)
            break

        folder_path = os.getcwd()
        xlsx_file_name = '다음뉴스_{}.xlsx'.format(date)

        result_df.to_excel(xlsx_file_name)

        print('엑셀 저장 완료 | 경로 : {}\\{}'.format(folder_path, xlsx_file_name))
        os.startfile(folder_path)

main()
