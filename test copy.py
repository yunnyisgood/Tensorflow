samsung = [['코스닥', '공시', '위드', '텍', '케어', '젠'],  # 6 [0, 0, 0, 0, 0, 0]
['대구', '미술관', '이건희', '컬렉션', '특별', '전', '개최'], # 7  [0, 0, 0, 1, 0, 0, 0]
['주년', '훨훨', '날아라', '코스닥', '연고', '점', '경신']] # 7  [1, 0, 0, 2, 0, 0]
word_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0, 0, 1, 0, 0, 0, 2, 0, 0]  #20

new_list = []
sum_len = 0

for i in range(len(samsung)):
    print('i', i)
    if i == 0:
        temp_len = len(samsung[i])
        print('temp_len  1: ', temp_len) # 6
        temp = word_list[i:temp_len]
        print('temp 1: ', temp) #  [0, 0, 0, 0, 0, 0]
        new_list.append(temp)
    elif i > 0:
        temp_len = len(samsung[i])
        print('temp_len  2: ', temp_len) # 7
        temp_len_before = len(samsung[i-1])
        sum_len += int(len(samsung[i-1]))
        print('sum_len',sum_len) # 6
        temp = word_list[sum_len:temp_len+sum_len]
        print('temp 2: ', temp)
        new_list.append(temp)
    print('new_list: ',new_list)

for i in new_list:
    print(type(i))
