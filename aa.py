import tensorflow as tf
import cv2
import numpy as np

# 导入MNIST 数据集，不能直接下载的就先去官网下好，然后拖到工程目录下自己读取
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("input_data/", one_hot=True)

# 神经网络参数
num_input = 784   # mnist数据集里面的图片是28*28所以输入为784
n_hidden_1 = 256  # 隐藏层神经元
n_hidden_2 = 256
num_output = 10   # 输出层

# 模型类
class Model(object):
    def __init__(self, learning_rate, num_steps, batch_size, display_step):
        self.learning_rate = learning_rate  # 学习率
        self.num_steps = num_steps          # 训练次数
        self.batch_size = batch_size        # batch大小
        self.display_step = display_step    # 日志打印周期

        # 权重参数 注意此处不能讲权重全部初始化为零
        self.weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, num_output])),
        }

        # 偏置参数
        self.biases = {
            'h1': tf.Variable(tf.random_normal([n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([num_output])),
        }

    # 网络模型
    def neural_net(self, input):
        layer_1 = tf.add(tf.matmul(input, self.weights['h1']), self.biases['h1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['h2'])
        out_layer = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])
        return out_layer

    # 训练模型
    def train(self):
        # 占位符
        X = tf.placeholder(tf.float32, shape=[None, num_input])
        Y = tf.placeholder(tf.float32, shape=[None, num_output])
        # 创建模型
        logits = self.neural_net(X)
        pred = tf.nn.softmax(logits)

        # 损失函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        # 计算准确率
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 定义优化器
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            for step in range(1, self.num_steps + 1):
                batch_x, batch_y = mnist.train.next_batch(batch_size=self.batch_size)
                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

                if step % self.display_step == 0 or step == 1:
                    loss_v, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})

                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss_v) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
            print("optimization finished!")
            saver.save(sess, './model/neural_net.ckpt')
            # 用测试集计算准确率
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                                     Y: mnist.test.labels}))
    # 评估函数 用来读入自定义的图片来验证模型的准确率
    def evaluate(self, img_dir):
        with tf.Session() as sess:
            # 二值化处理
            image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            im = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
            img_gray = (im - (255 / 2.0)) / 255
            cv2.imshow('out',img_gray)
            cv2.waitKey(0)
            img = np.reshape(img_gray, [-1, 784]) # -1表示不固定当前维度大小
            # 恢复模型
            saver = tf.train.Saver()
            saver.restore(sess, save_path='./model/neural_net.ckpt')
            # 识别
            X = tf.placeholder(tf.float32, shape=[None, num_input])
            Y = tf.placeholder(tf.float32, shape=[None, num_output])
            # 创建模型
            logits = self.neural_net(X)
            pred = tf.nn.softmax(logits)
            prediction = tf.argmax(pred, 1)
            predint = prediction.eval(feed_dict={X: img}, session=sess)
            print(predint)


if __name__ == '__main__':
    model = Model(learning_rate=0.01, num_steps=5000, batch_size=128, display_step=100)
    model.train()
    model.evaluate("test.jpg")



