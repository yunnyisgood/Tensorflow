'''
[1, np.nan, np.nan, 8, 10]
-> 결측치를 보통은 삭제하거나 0으로 대체
-> 과연 옳은 방법인가? 데이터 손실이 커지게 됨

> 결측치 처리
1. 행 삭제
2. 0으로 대체(또는 특정값으로 대체) -> [1, 0, 0, 8, 10]
-앞에값                               [1, 1, 1, 8, 10]
-뒤에값                               [1, 8, 8, 8, 10]
-중위값                               [1, 4.5, 4.5, 8, 10]
보간 -> 결측치를 뺀 나머지 데이터로 훈련을 시킨 뒤 결과치를 구한다 

+ boost 계열. tree 계열은 특히 결측치에 대해 자유롭다
   => 따로 처리해줄 필요 없다 

+ 만약 [1,2, 10만, N, 5, 6, 7]
=> bogun에 따르면 N은 5만정도 
=> 만약 10만이 이상치라면?                                    
                                      
'''

from pandas import DataFrame, Series
import pandas as pd
from datetime import datetime
import numpy as np

datestrs = ['8/13/2021','8/14/2021', '8/15/2021', '8/16/2021', '8/17/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print(type(dates))

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)

'''
2021-08-13     1.0
2021-08-14     NaN
2021-08-15     NaN
2021-08-16     8.0
2021-08-17    10.0
dtype: float64

ts_intp_linear = ts.interpolate() -> series를 통해 중간값으로 결측치 처리
2021-08-13     1.000000
2021-08-14     3.333333
2021-08-15     5.666667
2021-08-16     8.000000
2021-08-17    10.000000
dtype: float64
'''