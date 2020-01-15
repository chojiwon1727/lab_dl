"""
numpy.nditer 객체 : 반복문(for, while)을 쓰기 쉽게 도와주는 객체
"""
import numpy as np

np.random.seed(1231)
a = np.random.randint(100, size=(2,3))
print(a)

# 40 21 5 52 84 39
for row in a:
    for x in row:
        print(x, end=' ')
print()

i = 0
while i < a.shape[0]:
    j = 0
    while j < a.shape[1]:
        print(a[i, j], end=' ')
        j += 1
    i += 1
print()

# for문은 값으로 비교가 가능하지만 index를 사용할 수는 없음  -> index와 val을 함께 상요하려면 enumerate를 사용
# while문은 index를 사용할 수 있어서 특정 값, 특정 row, 특정 column만 사용 가능


# nditer 클래스 객체 생성
# for in 구문에서 iterator는 a에서 원소를 하나씩 꺼내줌
# -> 기본값: 첫번째 row꺼내고 그 다음 row로 실행(c_index) <-> f_index: column별로 실행

with np.nditer(a) as iterator:
    for val in iterator:
        print(val, end=' ')
print()

with np.nditer(a, flags=['multi_index']) as iterator:
    while not iterator.finished:            # iterator의 반복이 끝나지 않았으면
        i = iterator.multi_index            # iterator.multi_index : 2차원 배열의 인덱스 -> ex, (0, 1)
        print(f'{i}, {a[i]}', end=' ')
        iterator.iternext()
print()

with np.nditer(a, flags=['c_index']) as iterator:
    while not iterator.finished:
        i = iterator.index
        print(f'{i}, {iterator[0]}', end=' ')
        iterator.iternext()
print()

with np.nditer(a, flags=['multi_index']) as iterator:
    while not  iterator.finished:
        a[iterator.multi_index] *= 2
        iterator.iternext()

print(a)

# with np.nditer(a, flags=['c_index']) as iterator:
#     while not iterator.finished:
#         iterator[0] *= 2
#         iterator.iternext()
# print(a)
# -> ValueError: output array is read-only

with np.nditer(a, flags=['c_index'], op_flags=['readwrite']) as iterator:
    while not iterator.finished:
        iterator[0] *= 2
        iterator.iternext()
print(a)
