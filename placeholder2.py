#################################################################
# placeholder2.py
# 플레이스홀더 만들어서 입력과 출력 할당하기
#
import tensorflow as tf

#플레이스홀더 정의하기
a = tf.placeholder(tf.int32, [None])   # 정수 자료형 3개를 가지 배열

# 배열을 모든 값을 2배하는 연상 정의하기
b = tf.constant(2)
x2_op = a * b

# 세션 시작하기
sess = tf.Session()

# 플레이스 홀더에 값을 넣고 실행하기
r1 = sess.run(x2_op, feed_dict={a:[1, 2, 3, 4, 5]})
print(r1)
r2 = sess.run(x2_op, feed_dict={a:[10, 20]})
print(r2)
