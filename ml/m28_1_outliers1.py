import numpy as np
import matplotlib.pyplot as plt

aaa = np.array([1, 2, -1000, 4, 5, 6, 7, 8, 90, 100, 500])

'''
위에서 이상치는 -10000

[1, 2, 4, 5, 6, 7, 8, 90, 100, 5000]

사분위 수 = quantile
위의 array에서 중위값은 6, 1사분위는 2, 3사분위는 90

-10000,      2,      6,      90,      5000
            1/4    중위값    3/4
'''

def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out, [25, 50, 75])
    # np.percentile(arr, 분위)
    # np.percentile(data_out, [25, 50, 75]) -> 하위 25%, 50%, 상위 25%를 찾아준다 

    print("1사분위: ", quantile_1)
    print("q2: ", q2)
    print("3사분위: ", quantile_3)
    iqr = quantile_3 - quantile_1
    lower_bound = quantile_1 - (iqr * 1.5)
    upper_bound = quantile_3 + (iqr * 1.5)

    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)

print("이상치의 위치: ", outliers_loc)

# 시각화
# 위 데이터를 boxplot 형태로 그리시오
plt.figure(figsize=(7, 6))
plt.boxplot(aaa)
plt.ylim(-100, 100)
plt.show()

# 이상치 처리
# 1. 삭제
# 2. nan 처리 후 bogan // linear 
# 3.   -----------------   -> 결측치 처리 방법과 유사 


'''
1사분위:  3.0
q2:  6.0
3사분위:  49.0
이상치의 위치:  (array([ 2, 10], dtype=int64),)
'''

'''
백분위수(Percentile)는 
크기가 있는 값들로 이뤄진 자료를 순서대로 나열했을 때 백분율로 나타낸 특정 위치의 값을 이르는 용어

'''