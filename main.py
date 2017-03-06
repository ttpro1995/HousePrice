import tensorflow as tf
import data_util
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # step 1: read dataset
    x, y = data_util.read_data()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)



    # step 2: create placeholder for X and Y
    X = tf.placeholder(tf.float32, [31, 2])
    Y = tf.placeholder(tf.float32, [31])

    # step 3: w and b
    w = tf.Variable(tf.random_normal(shape=[2, 1], stddev=0.01), name='a')
    b = tf.Variable(tf.random_normal(shape=[1], stddev=0.01), name='bias')

    # step 4: model to predic y
    Y_predicted = tf.tensordot(X,w, axes=1) + b

    # step 5: loss function
    loss = tf.reduce_sum(tf.square(Y - Y_predicted))  # sum of the squares

    # step 6: optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000000001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(100):
            #for i in range(len(Y_train)):
            sess.run(optimizer, feed_dict={X:X_train, Y:Y_train})
            w_value, b_value = sess.run([w, b])
            print(w_value, b_value)

