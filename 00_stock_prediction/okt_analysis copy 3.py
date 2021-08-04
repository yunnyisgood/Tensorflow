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



# table = dict()
# samsung_word_score = []
# negative_list = []
# neutral_list = []
# positive_list = []
# negative = 0
# neutral = 0
# positive = 0
# # for i in range(len(word_dict)):
# #     for j in range(len(samsung_temp_list)):
# #         if (i for i in samsung_temp_list[j]) in word_dict['score'][i]:
# #         # if any(i in word_dict['word'][i] for i in samsung_temp_list[j]):
# #         # (i for i in samsung_temp_list[j]) in word_dict['score'][i]:
# #             samsung_word_score.append(word_dict['score'][i])
# # print(samsung_word_score)
          
# for i in range(0, len(data)):
#     for j in range(len(samsung_temp_list)):
#         if data[i]['word'] == samsung_temp_list[j]:
#             result.pop()
#             result.pop()
#             result.append(data[i]['polarity'])    
# print(result[:10])
# # def data_list(wordname):    
# #     for i in range(0, len(data)):
# #         if data[i]['word'] == wordname:
# #             result.pop()
# #             result.pop()
# #             result.append(data[i]['word_root'])
# #             result.append(data[i]['polarity'])  
# #     r_word = result[0]
# #     s_word = result[1]
# #     print('어근 : ' + r_word)
# #     print('극성 : ' + s_word)     
# #     return r_word, s_word
# # with open('../_data/polarity.csv', 'r', -1, 'utf-8') as polarity:
# #     next(polarity)
# #     for line in csv.reader(polarity):
# #         key = str()
# #         for word in line[0].split(';'):
# #             key += word.split('/')[0]
# #             # print(key)          
# #     table[key] = {'Neg': line[3], 'Neut':line[4], 'Pos':line[6]}
# # # 점수를 담을 빈 데이터 프레임을 만든다 
# # columns = ['negative', 'neutral', 'positive']
# # df = pd.DataFrame(columns=columns)
# # # for word in samsung_temp_list:
# # #         if any(word in table):
# # #             negative += float(table[word]['Neg'])
# # #             neutral += float(table[word]['Neut'])
# # #             positive += float(table[word]['Pos'])
# # negative_list.append(negative)
# # neutral_list.append(neutral)
# # positive_list.append(positive)
# # df['negative'] = negative_list
# # df['neutral'] = neutral_list
# # df['positive'] = positive_list
# # print(df)
# # # samsung_noun = [join(okt.morphs(title[0])) for title in samsung['title'] if not title in stopwords]
# # print(samsung_temp_list[:10])
# # print(knu[:10])
# # table = dict()
# print(samsung_temp_list[0])    

# ['코스닥', '공시', '위드', '텍', '케어', '젠']
# ['대구', '미술관', '이건희', '컬렉션', '특별', '전', '개최']

result_list = []
final_list = []
new_list = []
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
                print('result_list 1: ', result_list)
                result_list.append(score)
            elif len(result_list) != 0:
                result_list.pop()
                print('result_list 2: ', result_list)
                result_list.append(score)
                print('result_list 3: ', result_list)

            print('sameword:',sameword)
            print('score',sameword[-1][-1])

        print('a')
        result_list.append(0)
        subset = result_list
        print(subset)

    print('1')
    if len(final_list) == 0 :
        print('subset', subset)
        final_list.append(subset)
        df['score'] = pd.Series(final_list)

    elif len(final_list) != 0:
        print('subset[len(final_list):]',subset[len(final_list[-1]):])
        print(len(final_list))
        final_list.append(subset[len(final_list):])
        df['score'] = pd.Series(final_list)


    print('final_list 1: ', final_list)
    print('2')
    print('final_result_list: ', result_list)
    print('final_list 2: ', final_list)
    print('3')
    print(df)
    return final_list

# ['코스닥', '공시', '위드', '텍', '케어', '젠']
# ['대구', '미술관', '이건희', '컬렉션', '특별', '전', '개최']
    

# final = []
# temp = [['가능', '아리송', '가래가', '텍', '가다듬어', '젠'], ['a', 'b', 'c', 'd', 'e']]
# temp2 = []

# samsung['samsung_word list'] = np.nan
samsung['samsung_temp_list'] = samsung['samsung_temp_list'][:2].apply(lambda x: data_list(x))
print(samsung['samsung_temp_list'])
