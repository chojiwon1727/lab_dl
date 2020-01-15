"""
ex11에서 저장한 pickle 파일을 읽어서 파라미터들을 화면에 출력
-> 가중치, 편향 행렬 출력
"""
import pickle

with open('Weight_bias.pickle', mode='rb') as f:
    params = pickle.load(f)

for key, value in params.items():
    print(key, ':', value.shape)