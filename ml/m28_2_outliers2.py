import matplotlib.pyplot as plt
import numpy as np


'''
다중컬럼일때 이상치 찾기 <<< 완료해놓기
'''

aaa = np.array([[1, 2, 10000, 3, 4, 6, 7, 8, 90, 100, 5000], 
                [1000, 2000, 3, 4000, 5000, 6000, 7000, 8, 9000, 10000, 1001]])

# (2, 10) -> (10, 2)
aaa = aaa.transpose()
print(aaa.shape)                

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


# 시각화
# 위 데이터를 boxplot 형태로 그리시오
plt.figure(figsize=(7, 6))
plt.boxplot(aaa)
# plt.ylim(-100, 100)
plt.show()
