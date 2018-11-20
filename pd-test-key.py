import pandas as pd

# 키, 몸무게, 유형 데이터프레임 생성하기
tbl = pd.DataFrame({
    "weight": [80.0, 70.4, 65.5, 49.5, 51.2],
    "height": [170, 180, 155, 143, 154],
    "type": ['f', 'n', 'n', 't', 't']
})

# 몸무게 목록 추출하기
print('몸무게 목록')
print(tbl['weight'])

# 몸무게 목록
# 0    80.0
# 1    70.4
# 2    65.5
# 3    49.5
# 4    51.2
# Name: weight, dtype: float64


# 몸무게와 키 목록 추출하기
print('몸무게와 키 목록')
print(tbl[['weight', 'height']])        # tbl안에 [[  ]] 주의할 것

# 몸무게와 키 목록
#    weight  height
# 0    80.0     170
# 1    70.4     180
# 2    65.5     155
# 3    49.5     143
# 4    51.2     154


# 원하는 위치의 값을 추출할 때는 슬라이스 사용
print("tbl[2:4]\n", tbl[2:4])
#     height type  weight
# 2     155    n    65.5
# 3     143    t    49.5

# 원하는 조건의 값을 추출할 수도 있음
print('--- height가 160 이상인 경우만 추출' )
print(tbl[tbl.height > 160])
#    height type  weight
# 0     170    f    80.0
# 1     180    n    70.4

print('--- type이 f인 경우만 추출')
print(tbl[tbl.type == 'f'])
#    height type  weight
# 0     170    f    80.0

# 정렬하기
print('--- 키로 정렬')
print(tbl.sort_values(by="height"))
#    height type  weight
# 3     143    t    49.5
# 4     154    t    51.2
# 2     155    n    65.5
# 0     170    f    80.0
# 1     180    n    70.4

print('--- 몸무게로 정렬')
print(tbl.sort_values(by='weight', ascending=False))
#    height type  weight
# 0     170    f    80.0
# 1     180    n    70.4
# 2     155    n    65.5
# 4     154    t    51.2
# 3     143    t    49.5
