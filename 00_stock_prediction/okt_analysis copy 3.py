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




samsung = pd.read_excel('../_data/네이버뉴스_2021-08-02_삼성전자.xlsx', header=0)
stopwords = pd.read_csv('../_data/stopwords.txt').values.tolist()
word_dict = pd.read_csv('../_data/SentiWord_Dict.txt', sep='\s+', header=None,
                    names=['word', 'score'])
with open('../_data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
    data = pd.read_json(f)
print(data)
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


# ['코스닥', '공시', '위드', '텍', '케어', '젠']
# ['대구', '미술관', '이건희', '컬렉션', '특별', '전', '개최']

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
                # print('result_list 1: ', result_list)
                result_list.append(score)
            elif len(result_list) != 0:
                result_list.pop()
                # print('result_list 2: ', result_list)
                result_list.append(score)
                # print('result_list 3: ', result_list)

            # print('sameword:',sameword)
            # print('score',sameword[-1][-1])

        print('a')
        result_list.append(0)
        subset = result_list
        print(subset)

    print('1')
    final_list.append(subset)
    df['score'] = pd.Series(final_list)
    print('final_list 1: ', final_list)
    print(df)

    # return final_list

#  ['코스닥', '공시', '위드', '텍', '케어', '젠']
# ['대구', '미술관', '이건희', '컬렉션', '특별', '전', '개최']
# [주년, 훨훨, 날아라, 코스닥, 연고, 점, 경신]



samsung['samsung_word list'] = np.nan
samsung['samsung_word list'] = samsung['samsung_temp_list'].apply(lambda x: data_list(x))
print(samsung)

word_list = df['score'][0]
print(word_list)
new_list = []
sum_len = 0
samsung['word_list'] = ""

for i in range(len(samsung['samsung_temp_list'])):
    print('i', i)
    if i == 0:
        temp_len = len(samsung['samsung_temp_list'][i])
        print('temp_len  1: ', temp_len)
        temp = word_list[i:temp_len]
        print('temp 1: ', temp)
        new_list.append(temp)
    elif i > 0:
        temp_len = len(samsung['samsung_temp_list'][i])
        print('temp_len  2: ', temp_len) # 
        temp_len_before = len(samsung['samsung_temp_list'][i-1])
        sum_len += int(len(samsung['samsung_temp_list'][i-1]))
        print('sum_len',sum_len) # 6
        temp = word_list[sum_len:temp_len+sum_len]
        print('temp 2: ', temp)
        new_list.append(temp)

samsung['word_list'] = pd.Series(new_list)
del samsung['samsung_word list']
print('new_list: ',new_list)
print(samsung)


# 엑셀 파일로 데이터 프레임 저장
date_time = str(datetime.now())
date_time = date_time[:date_time.rfind(':')].replace(' ', '_')
date_time = date_time.replace(':','시') + '분'

folder_path = os.getcwd()
xlsx_file_name = '삼성 최종{}.xlsx'.format(date_time)

samsung.to_excel(xlsx_file_name)

print('엑셀 저장 완료 | 경로 : {}\\{}'.format(folder_path, xlsx_file_name))
os.startfile(folder_path)

