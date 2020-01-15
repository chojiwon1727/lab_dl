import numpy as np

a = np.arange(10)
print(a)

size = 3
for i in range(0,len(a), size):
    print(a[i:(i+size)])


# 파이썬의 리스트
b = [1,2]
c = [3,4,5]
b.append(c)
print(b)

x = np.array([1,2])
y = np.array([3,4,5])
x = np.append(x, y)
print(x)
