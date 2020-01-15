"""
pickle 데이터 타입
Serialize(직렬화 = pickling) = 객체 타입 -> 파일 저장
Deserialize(역직렬화 = unpickling) = 파일 -> 객체 타입
"""
import pickle

arr = [1, 100, 'A', 3.141592]   # list 객체 타입
with open('array.pickle', mode='wb') as f:     # W: write, b: binary   -> 2진수를 저장하겠다는 의미
    pickle.dump(arr, f)    # 객체(obj)를 파일(f)에 저장

with open('array.pickle', mode='rb') as f:
    data = pickle.load(f)
print(data)

data ={
    'name': '오쌤',
    'age': 16,
    'k1': [1, 2.0, 'AB'],
    'k2': {'tel': '010-0000-0000', 'email': 'jake@test.com'}}

# data객체를 data.pkl 파일에 저장(serialization)
with open('data.pkl', mode='wb') as f:
    pickle.dump(data, f)

# data.pkl 파일을 읽어서 dict 객체를 복원(deserialization)
with open('data.pkl', mode='rb') as f:
    data_pkl = pickle.load(f)
print(data_pkl)