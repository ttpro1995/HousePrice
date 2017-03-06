# Thai Thien
# 1351040

import numpy as np
import tensorflow as tf
import xlrd
import matplotlib.pyplot as plt
import data_util
from sklearn.model_selection import train_test_split


x, y = data_util.read_data()
x = x[:, 0]# get only first feature
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

w = tf.Variable(0.1, name='weights')
b = tf.Variable(0.3, name='bias')

Y_predicted = X*w + b
loss= tf.square(Y-Y_predicted, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000000001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(100):
        for i in range(len(Y_train)):
            sess.run(optimizer, feed_dict={X:X_train[i], Y:Y_train[i]})
        w_value, b_value = sess.run([w, b])
        print(w_value, b_value)
    predicted = []
    total_loss = 0
    # draw
    for i in range(len(Y_test)):
        y_hat, loss_value = sess.run([Y_predicted, loss], feed_dict={X: X_test[i], Y: Y_test[i]})
        total_loss += loss_value
        predicted.append(y_hat)
    predicted = np.asarray(predicted)
    average_loss = total_loss/len(y)
    print('average_lost ',average_loss)


    plt.plot(X_test, Y_test, 'g^', X_test, predicted, 'bs')
    plt.show()

