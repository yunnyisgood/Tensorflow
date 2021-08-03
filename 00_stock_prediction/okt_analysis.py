import pandas as pd
import json
from konlpy.tag import Okt
import numpy as np
import konlpy
import csv


samsung = pd.read_excel('../_data/네이버뉴스_2021-08-02_삼성전자.xlsx', header=0)
stopwords = pd.read_csv('../_data/stopwords.txt').values.tolist()

okt = Okt()

# 형태소 분석

# noun_dict = pd.read_csv('../_data/polarity.csv', header=0, sep=',')
# print(noun_dict[:10])
table = dict()

with open('../_data/polarity.csv', 'r', -1, 'utf-8') as polarity:
    next(polarity)

    for line in csv.reader(polarity):
        key = str()
        for word in line[0].split(';'):
            key += word.split('/')[0]
            # print(key)          
    table[key] = {'Neg': line[3], 'Neut':line[4], 'Pos':line[6]}
print('진행중')


# 점수를 담을 빈 데이터 프레임을 만든다 
columns = ['negative', 'neutral', 'positive']
df = pd.DataFrame(columns=columns)

print('진행중')

samsung_temp_list = []


def text_processing(start, end):
    for i in range(start, end):
        for i in samsung['title']:
            temp_samsung = []
            temp_samsung = okt.normalize(i)
            temp_samsung  = okt.morphs(i, stem=True)
            temp_samsung = [word for word in temp_samsung if not word in stopwords]
            samsung_temp_list.append(temp_samsung)      

        print('진행중')
        print('samsung_temp_list: ',samsung_temp_list[:10])
        samsung_list = []
        negative_list = []
        neutral_list = []
        positive_list = []

        negative = 0
        neutral = 0
        positive = 0

        for word in samsung_temp_list:
            if word in table:
                negative += float(table[word]['Neg'])
                neutral += float(table[word]['Neut'])
                positive += float(table[word]['Pos'])
        print('진행중')

        negative_list.append(negative)
        neutral_list.append(neutral)
        positive_list.append(positive)
        print('negative:', negative_list[:10])

    df['negative'] = negative_list
    df['neutral'] = neutral_list
    df['positive'] = positive_list
    

text_processing(0, 2)

print(df)










