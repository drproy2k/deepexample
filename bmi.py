###################################################################################
# bim.py
# Simple Classification
#
import pandas as pd
import numpy as np
import tensorflow as tf

# 키, 몸무게, 레이블이 적힌 CSV 파일을 읽어 들이기
csv = pd.read_csv("bmi.csv")

# 데이터 정규화 - numpy로 20000개 데이터 처리를 간단히 표현
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100

# 레이블을 onehot 벡터로 전환하기
bclass = {'thin': [1, 0, 0], 'normal': [0, 1, 0], 'fat': [0, 0, 1]}
csv['label_pat'] = csv['label'].apply(lambda x : np.array(bclass[x]))       # numpy로 간단히 표현
#print(csv['label_pat'])    # label 스트링이 onehot 벡터로 변환됨

# 테스트를 위한 데이터 분리
test_csv = csv[15000:20000]
test_pat = test_csv[['weight', 'height']]       # 컬럼이 weight, height인 것만 떼어라 - numpy로 간략히 표현
test_ans = list(test_csv['label_pat'])
# print(test_csv)
# input()
# print(test_ans)
# input()
# print(test_csv['label_pat'])

# 데이터 플로우 그래프 구축하기
# 플레이스 홀더 선언하기
x = tf.placeholder(tf.float32, [None, 2])   # 키와 몸무게 데이터 두개
y_ = tf.placeholder(tf.float32, [None, 3])   # 정답 레이블 벡터 크기가 3이다

# 변수 선언
W = tf.Variable(tf.zeros([2, 3]))
b = tf.Variable(tf.zeros([3]))

# 소프트맥스 회귀 정의하기
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 모델 훈련하기
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))      # loss 대신에 cross entropy 사용
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# 정답률 구하기
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# 세션 시작하기
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습하기
for step in range(3500):
    i = (step * 100) % 14000       # 14000개만 트레이닝에 사용키 위함임
    rows = csv[1 + i : 1 + i + 100]
    x_pat = rows[['weight', 'height']]
    y_ans = list(rows['label_pat'])
    sess.run(train, feed_dict={x: x_pat, y_: y_ans})
    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict={x: x_pat, y_: y_ans})
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        print('step=', step, 'cre=', cre, 'acc=', acc)

# 최종 정답률 구하기
acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
print('정답률=', acc)

# Tensorboard 사용하기
tw = tf.summary.FileWriter('log_dir', graph=sess.graph)