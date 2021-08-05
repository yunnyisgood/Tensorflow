import pandas as pd
import json
from konlpy.tag import Okt
import numpy as np
import konlpy
import csv

samsung = pd.read_excel('../_data/네이버뉴스_2021-08-02_삼성전자.xlsx', header=0)
stopwords = pd.read_csv('../_data/stopwords.txt').values.tolist()
word_dict = pd.read_csv('../_data/SentiWord_Dict.txt', sep='\s+', header=None,
                    names=['word', 'score'])
with open('../_data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
    data = pd.read_json(f)
print(data)
okt = Okt()

df = pd.DataFrame()

# 형태소 분석
samsung_temp_list = []
for sentence in samsung['title']:
    temp_samsung = []
    temp_samsung = okt.normalize(sentence)
    temp_samsung  = okt.morphs(sentence) #  문장에서 명사 추출 
    temp_samsung = [word for word in temp_samsung if not word in stopwords]  # 불용어 처리 
    samsung_temp_list.append(temp_samsung)
samsung['samsung_temp_list'] = samsung_temp_list


for i in samsung['samsung_temp_list'][:10]:
    print(i)
    print(type(i))


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
            result_list.append(score)
            print('sameword:',sameword)
            print('score',sameword[-1][-1])
            print('result_list X: ', result_list)
        print('a')
        result_list.append(0)
        print('result_list Y: ', result_list)
    
    print('final_list: ', final_list)
    return final_list
    
# final = []
# ['코스닥', '공시', '위드', '텍', '케어', '젠']
# temp = [['가능', '아리송', '가래가', '텍', '가다듬어', '젠'], ['a', 'b', 'c', 'd', 'e']]

samsung['samsung_temp_list'] = samsung['samsung_temp_list'][:2].apply(lambda x: data_list(x))
print(samsung['samsung_temp_list'])
