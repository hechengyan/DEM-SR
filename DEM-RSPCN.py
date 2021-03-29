# -*- coding: utf-8 -*-
__author__ = "Ruichur"
"""
Created on March 7 10:14:36 2021


"""

#import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.contrib.slim as slim
import scipy.misc  
import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

print(__author__)

batch_size_small = 1
batch_size_super = 1
n_input_super = 1440000
n_input_small = 5625
n_classes = 1

x1 = tf.placeholder("float", [None, n_input_small])
# corrupted image
x_small = tf.reshape(x1,[-1,75,75,1])
print("x_small:",x_small.shape)
x = tf.placeholder("float", [None, n_input_super])
# corrupted image
print("x.shape",x.shape)

net = x_small

N = tf.placeholder("int", [None, n_input_small])
#rspcn

for n in range(N):

    net = slim.conv2d(net, 64, 5)
    print("net1.shape",net.shape)
    net =slim.conv2d(net, 32, 3)
    print("net2.shape",net.shape)
    net = slim.conv2d(net, 4, 3)
    print("net3.shape",net.shape)
    net = tf.depth_to_space(net,2)
    print("net4.shape",net.shape)
    m = 75*2^n
    y_pred = tf.reshape(net,[-1,m * m])
    net = tf.reshape(y_pred,[-1,m,m,1])
    

cost = tf.reduce_mean(tf.pow(x - y_pred, 2))
tf.summary.scalar('cost',cost)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

training_epochs = 10000
display_step = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/espcn',sess.graph)
    total_batch = int(mnist.test.num_examples/batch_size_super)

    for epoch in range(training_epochs):

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size_small)
            batch_xt, batch_yt = mnist.test.next_batch(batch_size_super)
            _, c = sess.run([optimizer, cost], feed_dict={x1: batch_xs, x: batch_xt, N: 4})
            
        summary_str = sess.run(merged_summary_op, feed_dict={x1: batch_xs, x: batch_xt, N: 4});
        summary_writer.add_summary(summary_str, epoch);

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("finished!")

    show_num = 6
    encode_s, y_predv= sess.run(
        [x_small,y_pred], feed_dict={x1: mnist.train.images[:show_num], x: batch_xt, N: 4})
    
    f, a = plt.subplots(3, 1, figsize=(1, 3))
 #   for i in range(0,6):
         
    ORI = np.reshape(mnist.test.images[0], (1200, 1200)) 
    PRE1 = np.reshape(encode_s[0], (75, 75)) 
    PRE2 = np.reshape(y_predv[0], (1200, 1200))
    a[0].imshow(ORI, cmap ='gray')
    a[1].imshow(PRE1, cmap ='gray')
    a[2].imshow(PRE2, cmap ='gray')
    plt.show()
