import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)

plt.pie(ratio=y, labels=y, shadow=True, startangle=90)
plt.plot(x, y)
plt.grid()
plt.show()


'''
과제 

elu, selu, reaky relu ....
68_3_2, 3, 4로 만들 것

'''

