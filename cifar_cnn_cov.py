import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.001


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


class Model:

    def __init__(self):
        self.__build_net()

    def __build_net(self):
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

        self.x = tf.placeholder("float", [None, 32, 32, 3])
        x_img = tf.reshape(self.x, [-1, 32, 32, 3])
        self.y = tf.placeholder("float", [None, 10])

        conv1 = tf.layers.conv2d(inputs=x_img, filters=64, kernel_size=[5,5], padding="SAME", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], padding="SAME", strides=2)
        #dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5], padding="SAME", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], padding="SAME", strides=2)
        #dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3], padding="SAME",
                                 activation=tf.nn.relu)
        conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[3, 3], padding="SAME",
                                 activation=tf.nn.relu)

        flat = tf.reshape(conv5, [-1, 128 * 8 * 8])
        self.h_fc1 = tf.layers.dense(inputs=flat, units=32, activation=tf.nn.relu)
        # dropout4 = tf.layers.dropout(inputs=dense4, rate=0.3, training=self.training)

        self.logits = tf.layers.dense(inputs=self.h_fc1, units=10)

        self.y_pred = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.train_step = tf.train.RMSPropOptimizer(1e-3).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
    '''
    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})
    '''

m = Model()

# CIFAR-10 data load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# scalar 0~9 --> One-hot Encoding
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)


plot_test_acc=[]
plot_corr_mean=[]
plot_dead_relu=[]

epoch = 50000


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        batch = next_batch(128, x_train, y_train_one_hot.eval())

        if i % 100 == 0:
            train_accuracy = m.accuracy.eval(feed_dict={m.x: batch[0], m.y: batch[1], m.keep_prob: 1.0})
            loss_print = m.loss.eval(feed_dict={m.x: batch[0], m.y: batch[1], m.keep_prob: 1.0})


            test_batch = (x_test, y_test_one_hot.eval())
            test_h_fc1 = m.h_fc1.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1], m.keep_prob: 1.0})
            test_acc =m.accuracy.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1], m.keep_prob: 1.0})

            pearson_mat=np.corrcoef(test_h_fc1.T)

            pearson_flat=pearson_mat.flatten()
            original_len= len(pearson_flat)

            #delete nan
            pearson_flat_nnan = [x for x in pearson_flat if str(x) != 'nan']
            nnan_len= len(pearson_flat_nnan)

            plot_dead_relu.append( (original_len-nnan_len)/original_len )

            corr_mean = np.absolute(pearson_flat_nnan).mean()


            print("[Epoch %d]  train_acc: %f, loss: %f, test_acc: %f, corr_mean: %f" % (i, train_accuracy, loss_print, test_acc, corr_mean))
            plot_test_acc.append(test_acc)
            plot_corr_mean.append(corr_mean)

            print("len:", original_len, nnan_len)
            print(pearson_mat)
        # train with Dropout
        sess.run(m.train_step, feed_dict={m.x: batch[0], m.y: batch[1], m.keep_prob: 0.8})

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + m.accuracy.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1], m.keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("test_acc: %f" % test_accuracy)

    """
    plot_test_acc=[]
    plot_corr_mean=[]
    plot_dead_relu=[]
    """

    idx=[]
    for i in range(int(epoch/100)):
        idx.append(i)

    plt.title("Plot")
    plt.plot(idx, plot_test_acc, "r.-", idx, plot_corr_mean, "g.-", plot_dead_relu, "b.-")
    plt.show()
    
