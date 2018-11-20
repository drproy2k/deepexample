##################################################################
# tb_mul.py
# 간단한 텐서보드 사용 예제에 곱셈을 추가한 것
#
import tensorflow as tf

# 데이터 플로우 그래프 구출하기
a = tf.constant(20, name='a')
b = tf.constant(30, name='b')
mul_op = a * b

# 세션 생성
sess = tf.Session()

# Tensorboard 사용하기
tw = tf.summary.FileWriter('log_dir', graph=sess.graph) # log_dir에 session graph 데이터를 남겨라.

# 세션 실행하기
print(sess.run(mul_op))

# 터미날에 가서, activate py35Envs하고,
# tensorboard --logdir=log_dir Starting TensorBoard b'28 on port 600