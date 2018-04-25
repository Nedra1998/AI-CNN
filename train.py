import numpy as np
import tensorflow as tf
import data

def get_label(labels):
    new = np.zeros(labels.shape[0])
    for i, row in enumerate(labels):
        new[i] = np.where(row==1)[0]
    return new

n_inputs = 32*32*3
n_output = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
    # Single dense softmax layer
    # ==========================
    # shaped = tf.transpose(tf.reshape(X, [-1, 3, 32, 32]), (0, 2, 3, 1))
    # flat = tf.reshape(shaped, [-1, 32*32*3])
    # logits = tf.layers.dense(flat, n_output, name='outputs')
    # Convolutional Network
    # =====================
    shaped = tf.transpose(tf.reshape(X, [-1, 3, 32, 32]), (0, 2, 3, 1))
    conv_1 = tf.layers.conv2d(shaped, 32, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool_1  =tf.layers.max_pooling2d(conv_1, pool_size=2, strides=2, padding='valid')
    conv_2 = tf.layers.conv2d(pool_1, 64, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool_2  =tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2, padding='valid')
    conv_3 = tf.layers.conv2d(pool_2, 128, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool_3  =tf.layers.max_pooling2d(conv_3, pool_size=2, strides=2, padding='valid')
    flat = tf.reshape(pool_3, [-1, 4*4*128])
    logits = tf.layers.dense(flat, n_output, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 50
batch_size = 200

data.import_data([1, 2, 3, 4, 5], ['test_batch'])

test_x_batch = data.testing_data[0]
test_y_batch = get_label(data.testing_data[1])

max_val = 0

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        x_batch= None
        y_batch = None
        for iteration in range(data.data_size() // batch_size):
            x_batch, y_batch = data.next_batch(batch_size)
            y_batch = get_label(y_batch)
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: x_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: test_x_batch, y: test_y_batch})
        if acc_val > max_val:
            max_val = acc_val
            saver.save(sess, './models/best.ckpt')

        print(epoch, "Train:", acc_train, "Val:", acc_val)

    save_path = saver.save(sess, './models/model.ckpt')
