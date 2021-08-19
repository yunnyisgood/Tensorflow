import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope


# 이상치 처리
# 1. 삭제
# 2. nan 처리 후 bogan // linear 
# 3.   -----------------   -> 결측치 처리 방법과 유사 
# 4. scaler -> Rubsorscaler, QuantileScaler ,,,, 
#    => scaler 상태에서 이상치를 잡는 경우도 있는데 수동으로 했을때랑 차이 봐야 한다 
# 5. 모델링 : tree계열, boost계열에서는 어느정도 이상치에서 자유로운 모델이라 할 수 있다
#   -> 결측치에서도 자유로운 편

aaa = np.array([[1, 2, 10000, 3, 4, 6, 7, 8, 90, 100, 5000], 
                [1000, 2000, 3, 4000, 5000, 6000, 7000, 8, 9000, 10000, 1001]])

# (2, 10) -> (10, 2)
aaa = aaa.transpose()
print(aaa.shape) 

outliers = EllipticEnvelope(contamination=.4) 
# contamination의 비율을 기준으로 비율보다 낮은 값을 검출한다
# The amount of contamination of the data set, 
# i.e. the proportion of outliers in the data set. Range is (0, 0.5)

outliers.fit(aaa)

results = outliers.predict(aaa)

print(results)

# EllipticEnvelope(contamination=.2) 
# [ 1  1 -1  1  1  1  1  1  1  1 -1]
#  위에서 -1이 이상치를 의미한다 

# EllipticEnvelope(contamination=.4)
# [ 1  1 -1  1  1  1  1  1 -1 -1 -1] 



