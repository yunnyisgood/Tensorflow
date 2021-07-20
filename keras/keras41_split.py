import numpy as np

a = np.array(range(1, 11)) # 연속적 데이터 
size = 5


def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset)-size+1): # range(10-4=6) -> 6번동안 반복. 10개의 데이터를 5개씩 분리하기 위한 방법 
        subset = dataset[i : (i+size)] # dataset[0:5] -> dataset 0부터 4번째 값까지 
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

print(dataset)

x = dataset[:, :4] #열은 없고, 처음부터 4번째 전까지의 행을 출력
y = dataset[:, 4]

print("x : ", x)
print("y : ", y)

'''
dataset:
[[ 1  2  3  4  5] 
 [ 2  3  4  5  6] 
 [ 3  4  5  6  7] 
 [ 4  5  6  7  8] 
 [ 5  6  7  8  9] 
 [ 6  7  8  9 10]]

x :  
[[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]
y :  [ 5  6  7  8  9 10]
'''