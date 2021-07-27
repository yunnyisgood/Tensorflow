import numpy as np
from sklearn.datasets import load_iris

datasets = load_iris()

x_data_iris = datasets.data
y_data_iris = datasets.target

print(type(x_data_iris), type(y_data_iris))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris)

# boston, cancer, diabets
from sklearn.datasets import load_boston

datasets = load_boston()

x_data_boston = datasets.data
y_data_boston = datasets.target

print(type(x_data_boston), type(y_data_boston))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data_boston)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data_boston)

from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x_data_diabets = datasets.data
y_data_diabets = datasets.target

print(type(x_data_diabets), type(y_data_diabets))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_diabets.npy', arr=x_data_diabets)
np.save('./_save/_npy/k55_y_data_diabets.npy', arr=y_data_diabets)

from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

x_data_cancer = datasets.data
y_data_cancer = datasets.target

print(type(x_data_cancer), type(y_data_cancer))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_cancer.npy', arr=x_data_cancer)
np.save('./_save/_npy/k55_y_data_cancer.npy', arr=y_data_cancer)

from sklearn.datasets import load_wine

datasets = load_wine()

x_data_wine = datasets.data
y_data_wine = datasets.target

print(type(x_data_wine), type(y_data_wine))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(type(x_train), type(y_train))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_train_mnist.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_mnist.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_mnist.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_mnist.npy', arr=y_test)

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(type(x_data_wine), type(y_data_wine))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_train_fashion_mnist.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_fashion_mnist.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_fashion_mnist.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_fashion_mnist.npy', arr=y_test)

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(type(x_train), type(y_train))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_train_cifar10.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_cifar10.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_cifar10.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_cifar10.npy', arr=y_test)


from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(type(x_train), type(y_train))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=y_test)