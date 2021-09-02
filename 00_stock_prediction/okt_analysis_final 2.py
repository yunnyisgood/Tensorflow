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

samsung = pd.read_excel('6개월_최종뉴스크롤링2021-08-18_23시07분.xlsx', header=0)
samsung = samsung[['title', 'date']]
stopwords = pd.read_csv('stopwords.txt').values.tolist()
word_dict = pd.read_csv('SentiWord_Dict.txt', sep='\s+', header=None,
                    names=['word', 'score'])
with open('SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
    data = pd.read_json(f)
okt = Okt()

df = pd.DataFrame()
df['score'] = ""

# 형태소 분석
samsung_temp_list = []
for sentence in samsung['title']:
    temp_samsung = []
    temp_samsung = okt.normalize(sentence)
    temp_samsung  = okt.morphs(sentence) #  문장에서 명사 추출 
    temp_samsung = [word for word in temp_samsung if not word in stopwords]  # 불용어 처리 
    samsung_temp_list.append(temp_samsung)
samsung['samsung_temp_list'] = samsung_temp_list


# 각 형태소마다 점수로 수치화 시켜준다 
result_list = []
final_list = []
def data_list(wordname):   
    for i in wordname:
        print('i', i)
        i in data['word_root'].values
        if i in data['word_root'].values:
            print('i', i)
            sameword = data.loc[(data['word_root'].values == i)]
            sameword = sameword.drop_duplicates()
            sameword = sameword.values.tolist()
            score = sameword[-1][-1]
            if len(result_list)== 0:
                result_list.append(score)
            elif len(result_list) != 0:
                result_list.pop()
                result_list.append(score)
        result_list.append(0)
        subset = result_list
        print(subset)

    final_list.append(subset)
    df['score'] = pd.Series(final_list)

samsung['samsung_word list'] = np.nan
samsung['samsung_word list'] = samsung['samsung_temp_list'].apply(lambda x: data_list(x))

# 1차원의 데이터를 각 문장의 길이만큼 다시 자르기 
word_list = df['score'][0]
print(word_list)
new_list = []
sum_len = 0
samsung['word_list'] = np.nan

for i in range(len(samsung['samsung_temp_list'])):
    print('i', i)
    if i == 0:
        temp_len = len(samsung['samsung_temp_list'][i])
        temp = word_list[i:temp_len]
        new_list.append(temp)
    elif i > 0:
        temp_len = len(samsung['samsung_temp_list'][i])
        temp_len_before = len(samsung['samsung_temp_list'][i-1])
        sum_len += int(len(samsung['samsung_temp_list'][i-1]))
        temp = word_list[sum_len:temp_len+sum_len]
        new_list.append(temp)

samsung['word_list'] = pd.Series(new_list)
del samsung['samsung_word list']




# 엑셀 파일로 데이터 프레임 저장
date_time = str(datetime.now())
date_time = date_time[:date_time.rfind(':')].replace(' ', '_')
date_time = date_time.replace(':','시') + '분'

folder_path = os.getcwd()
xlsx_file_name = '삼성_6개월_최종{}.xlsx'.format(date_time)

samsung.to_excel(xlsx_file_name)

print('엑셀 저장 완료 | 경로 : {}\\{}'.format(folder_path, xlsx_file_name))
os.startfile(folder_path)

