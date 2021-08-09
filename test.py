import tensorflow as tf
import keras 
import pandas as pd

# total score 
temp = [[1,2,3,4,5,6,7],[1,2,3],[1,2,3,4,5]]
temp1 = [1,2,3,4,5,6,7,8,9]

for i in temp:
    print(i)
    print(type(i))

# sum_list = []
# def test(score):
#     list_sum = 0    
#     for i in score:
#         list_sum += int(len(i))
#         sum_list.append(list_sum)
#     print(sum_list)


# df = pd.DataFrame()
# df['a'] = ""
# df['a'] = temp

# df['a'] = df['a'].apply(lambda x: test(x))

'''
date score
df_total = pd.DataFrame()

df_total['Date_Score] = samsung['word_list'].resample('1D').mean() 
'''