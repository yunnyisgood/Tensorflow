import pandas as pd
import json
from konlpy.tag import Okt
import numpy as np
import konlpy


samsung = pd.read_excel('../_data/네이버뉴스_2021-08-02_삼성전자.xlsx', header=0)
# print(samsung)

okt = Okt()

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다', '로']

# 형태소 분석
# samsung_temp_list = []
# for sentence in samsung['title']:
#     temp_samsung = []
#     temp_samsung = okt.normalize(sentence)
#     temp_samsung  = okt.morphs(sentence, stem=True)
#     temp_samsung = [word for word in temp_samsung if not word in stopwords]
#     samsung_temp_list.append(temp_samsung)


# print(samsung_temp_list[:10])





