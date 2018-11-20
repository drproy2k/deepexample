# Numpy를 이용하면 데이터를 한꺼번에 조작할 수 있다
import numpy as np

# 10개의 float32 자료형 데이터 생성
v = np.zeros(10, dtype=np.float32)      # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(v)

# 연속된 10개의 uint64 자료형 데이터 생성
v = np.arange(10, dtype=np.uint64)      # [0 1 2 3 4 5 6 7 8 9]
print(v)

# v 값들에 모두 3배씩 하기
v *= 3          # [ 0  3  6  9 12 15 18 21 24 27]
print(v)

# v의 평균 구하기
print(v.mean())     # 13.5

