import pandas as pd
import json
from konlpy.tag import Okt
import numpy as np
import konlpy
import csv
import yfinance as yf
from pandas_datareader import data

samsung = pd.read_excel('../_data/네이버뉴스_2021-08-02_삼성전자.xlsx', header=0)
stopwords = pd.read_csv('../_data/stopwords.txt').values.tolist()

okt = Okt()


# 형태소 분석
samsung_temp_list = []
for sentence in samsung['title']:
    temp_samsung = []
    temp_samsung = okt.normalize(sentence)
    temp_samsung  = okt.morphs(sentence) #  문장에서 명사 추출 
    temp_samsung = [word for word in temp_samsung if not word in stopwords]  # 불용어 처리 
    samsung_temp_list.append(temp_samsung)
samsung['samsung_temp_list'] = samsung_temp_list

print(samsung)

# 삼성주가 다운로드
start_date = '2021-07-01'
end_date = '2021-07-31'
SAMSUNG = data.get_data_yahoo('005930.KS', start_date, end_date)

df = pd.DataFrame({'ds': SAMSUNG.index, 'y': SAMSUNG['Close']})
df.reset_index(inplace=True)
del df['Date']
print(df)

