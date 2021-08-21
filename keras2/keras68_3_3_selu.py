import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x) # 0보다 큰 값은 유지, 0보다 작은 값은 0으로 처리

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()


'''
과제 

elu, selu, reaky relu ....
68_3_2, 3, 4로 만들 것

'''

