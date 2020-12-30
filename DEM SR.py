# -*- coding: utf-8 -*-
"""
Created on Wed June 12 10:57:21 2020

@author: Ruichen Z
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.contrib.slim as slim
import scipy.misc  
import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

print(__Ruichen Z__)

batch_size = 1   
n_input = 784 # img shape: 28*28
n_classes = 5  # 0-9


x = tf.placeholder("float", [None, n_input])
img = tf.reshape(x,[-1,28,28,1])
# corrupted image
x_small = tf.image.resize_bicubic(img, (14, 14))



x_bicubic = tf.image.resize_bicubic(x_small, (28, 28))
x_nearest = tf.image.resize_nearest_neighbor(x_small, (28, 28))
x_bilin = tf.image.resize_bilinear(x_small, (28, 28))


net = slim.conv2d(x_small, 64, 5)
net =slim.conv2d(net, 32, 3)
net = slim.conv2d(net, 4, 3)
net = tf.depth_to_space(net,2)
print("net.shape",net.shape)

y_pred = tf.reshape(net,[-1,784])
cost = 0.3*tf.reduce_mean(tf.pow(x - y_pred, 2)) + 0.7*tf.abs(tf.multiply(tf.reduce_mean(x-y_pred),tf.reduce_mean(tf.pow(x - y_pred, 2))))
tf.summary.scalar('cost',cost)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

training_epochs = 50000
display_step =20

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/espcn',sess.graph)
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
            
        summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs});
        summary_writer.add_summary(summary_str, epoch);
        
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Finished!")

    show_num = 5
    encode_s,encode_b,encode_n ,encode_bi,y_predv= sess.run(
        [x_small,x_bicubic,x_nearest,x_bilin,y_pred], feed_dict={x: mnist.test.images[:show_num]})
    
    f, a = plt.subplots(6, 5, figsize=(5, 6))
    for i in range(show_num):
        ORI = np.reshape(mnist.test.images[i], (28, 28))
        PRE1 = np.reshape(encode_s[i], (14, 14))
        PRE2 = np.reshape(encode_b[i], (28, 28))
        PRE3 = np.reshape(encode_n[i], (28, 28))
        PRE4 = np.reshape(encode_bi[i], (28, 28))
        PRE5 = np.reshape(y_predv[i], (28, 28))
        scipy.misc.imsave('D:/fiveresults/finalssim/ori'+str(i)+'.jpg', ORI)
        scipy.misc.imsave('D:/fiveresults/finalssim/si'+str(i)+'.jpg', PRE1)
        scipy.misc.imsave('D:/fiveresults/finalssim/bi'+str(i)+'.jpg', PRE2)
        scipy.misc.imsave('D:/fiveresults/finalssim/ni'+str(i)+'.jpg', PRE3)
        scipy.misc.imsave('D:/fiveresults/finalssim/bii'+str(i)+'.jpg', PRE4)
        scipy.misc.imsave('D:/fiveresults/finalssim/predv'+str(i)+'.jpg', PRE5)
        a[0][i].imshow(ORI, cmap ='gray')
        a[1][i].imshow(PRE1, cmap ='gray')
        a[2][i].imshow(PRE2, cmap ='gray')
        a[3][i].imshow(PRE3, cmap ='gray')
        a[4][i].imshow(PRE4, cmap ='gray')
        a[5][i].imshow(PRE5, cmap ='gray')
    plt.show()