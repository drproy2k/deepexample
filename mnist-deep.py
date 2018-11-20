####################################################################
# mnist-deep.py
# MNIST CNN 학습
#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 손글씨 이미지 데이터 읽어 들이기
mnist = input_data.read_data_sets('mnist/', one_hot=True)

pixels = 28 * 28
nums = 10

# 플레이스 홀드 정의하기
x = tf.placeholder(tf.float32, shape=[None, pixels], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, nums], name='y_')

# 가중치와 바이어스를 초기화 하는 함수를 만들자
def weight_variable(name, shape):
    W_init = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(W_init, name='W_'+name)
    return W

def bias_variable(name, size):
    b_init = tf.constant(0.1, shape=[size])
    b = tf.Variable(b_init, name='b_'+name)
    return b

# 합성곱 계측 만드는 함수
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 최대 풀링층 만든는 함수
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 레이어 만들자
# 합성곱층1
with tf.name_scope('conv1') as scope:
    W_conv1 = weight_variable('conv1', [5, 5, 1, 32])   # 5x5커널, 입력1채널, 출력32채널 = 5x5필터 32개 의미
    b_conv1 = bias_variable('conv1', 32)
    x_image = tf.reshape(x, [-1, 28, 28, 1])            # '-1' 입력 데이터 개수 오픈, 28x28이미지, '1' 그레이 1채널
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 최대풀링층1 28 x 28 --> 14 x 14로
with tf.name_scope('max_pool1') as scope:
    h_pool1 = max_pool(h_conv1)

# 합성곱층2
with tf.name_scope('conv2') as scope:
    W_conv2 = weight_variable('conv2', [5, 5, 32, 64])
    b_conv2 = bias_variable('conv2', 64)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 풀링층2 14 x 14 --> 7 x 7로
with tf.name_scope('max_pool2') as scope:
    h_pool2 = max_pool(h_conv2)

# FC층
with tf.name_scope('fc_layer') as scope:
    n = 7 * 7 * 64      # 7x7 이미지가 64장 입력으로 들어온다
    W_fc = weight_variable('fc', [n, 1024])
    b_fc = bias_variable('b_fc', 1024)
    h_pool2_flat = tf.reshape(h_pool2, [-1, n])
    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

# Drop-out for overfitting
with tf.name_scope('drop_out') as scope:
    keep_prob = tf.placeholder(tf.float32)
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

# 출력층
with tf.name_scope('readout') as scope:
    W_fc2 = weight_variable('W_fc2', [1024, 10])
    b_fc2 = bias_variable('b_fc2', 10)
    y_conv = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

# 모델 학습시키기
with tf.name_scope('loss') as scope:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))     # y_ 플레이스홀더에는 정답을 할당할꺼야
with tf.name_scope('training') as scope:
    optimizer = tf.train.AdadeltaOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entropy)

# 모델 평가하기
with tf.name_scope('predict') as scope:
    predict_step = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))     # 정답과 일치 여부 체크
    accuracy_step = tf.reduce_mean(tf.cast(predict_step, tf.float32))   # ??????????

# feed_dict 설정하기
def set_feed(images, labels, prob):
    return {x: images, y_: labels, keep_prob: prob}

# 세션 시작하기
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 텐서보드도 준비
    tw = tf.summary.FileWriter('log_dir', graph=sess.graph)
    # 테스트 전용 피드 만들자
    test_fd = set_feed(mnist.test.images, mnist.test.labels, 1)
    # 학습 시작하기
    for step in range(10000):
        batch = mnist.train.next_batch(50)
        fd = set_feed(batch[0], batch[1], 0.5)      # drop_out = 0.5
        _, loss = sess.run([train_step, cross_entropy], feed_dict=fd)
        if step % 100 == 0:
            acc = sess.run(accuracy_step, feed_dict=test_fd)    # 배치를 100번 돌때마다 acc를 test데이터로 체크해 본다
            print('step=', step, 'loss=', loss, 'acc=', acc)
    # 최종 결과 확인
    acc = sess.run(accuracy_step, feed_dict=test_fd)
    print('정답률=', acc)
