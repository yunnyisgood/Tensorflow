from numpy.lib import twodim_base
import pandas as pd
import json
from konlpy.tag import Okt
import numpy as np
import konlpy
import csv
from itertools import accumulate
from datetime import datetime
import os

Feb = pd.read_excel('네이버뉴스_2월2021-08-18_22시46분.xlsx', header=0)
Mar = pd.read_excel('네이버뉴스_3월2021-08-18_22시45분.xlsx', header=0)
Apr = pd.read_excel('네이버뉴스_4월2021-08-18_22시43분.xlsx', header=0)
May = pd.read_excel('네이버뉴스_5월2021-08-18_22시41분.xlsx', header=0)
Jun = pd.read_excel('네이버뉴스_6월2021-08-18_22시37분.xlsx', header=0)
Jul = pd.read_excel('네이버뉴스_7월2021-08-18_22시26분.xlsx', header=0)

Feb = Feb[['title', 'date']]
Mar = Mar[['title', 'date']]
Apr = Apr[['title', 'date']]
May = May[['title', 'date']]
Jun = Jun[['title', 'date']]
Jul = Jul[['title', 'date']]

df = pd.concat([Feb, Mar, Apr, May, Jun, Jul])

print(df)
print(df.shape)

# 엑셀 파일로 데이터 프레임 저장
date_time = str(datetime.now())
date_time = date_time[:date_time.rfind(':')].replace(' ', '_')
date_time = date_time.replace(':','시') + '분'

folder_path = os.getcwd()
xlsx_file_name = '6개월_최종뉴스크롤링{}.xlsx'.format(date_time)

df.to_excel(xlsx_file_name)

print('엑셀 저장 완료 | 경로 : {}\\{}'.format(folder_path, xlsx_file_name))
os.startfile(folder_path)


