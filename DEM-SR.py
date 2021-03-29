# -*- coding: utf-8 -*-
__author__ = "Ruichur"
"""
Created on March 3 10:14:36 2021


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
n_input_small = 360000
n_classes = 6

x1 = tf.placeholder("float", [None, n_input_small])
# corrupted image
x_small = tf.reshape(x1,[-1,600,600,1])
print("x_small:",x_small.shape)
x = tf.placeholder("float", [None, n_input_super])
# corrupted image
print("x.shape",x.shape)



x_bicubic = tf.image.resize_bicubic(x_small, (1200, 1200))
print("x_bicubic.shape:",x_bicubic.shape)
x_nearest = tf.image.resize_nearest_neighbor(x_small, (1200, 1200))
print("x_nearest.shape:",x_nearest.shape)
x_bilin = tf.image.resize_bilinear(x_small, (1200, 1200))
print("x_bilin.shape:",x_bilin.shape)


#espcn
net = slim.conv2d(x_small, 64, 5)
net =slim.conv2d(net, 32, 3)
net = slim.conv2d(net, 4, 3)
net = tf.depth_to_space(net,2)
print("net.shape",net.shape)

y_pred = tf.reshape(net,[-1,1440000])
print("y_pred.shape:",y_pred.shape)

#cost = tf.reduce_mean(tf.pow(x - y_pred, 2))
cost = 0.3*tf.reduce_mean(tf.pow(x - y_pred, 2)) + 0.7*tf.abs(tf.multiply(tf.reduce_mean(x-y_pred),tf.reduce_mean(tf.pow(x - y_pred, 2))))

tf.summary.scalar('cost',cost)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

training_epochs = 500
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/espcn',sess.graph)
    total_batch = int(mnist.test.num_examples/batch_size_super)
    for epoch in range(training_epochs):

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size_small)
            batch_xt, batch_yt = mnist.test.next_batch(batch_size_super)
            _, c = sess.run([optimizer, cost], feed_dict={x1: batch_xs, x: batch_xt})
            
        summary_str = sess.run(merged_summary_op, feed_dict={x1: batch_xs, x: batch_xt});
        summary_writer.add_summary(summary_str, epoch);

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("完成!")

    show_num = 6
    encode_s,encode_b,encode_n ,encode_bi,y_predv= sess.run(
        [x_small,x_bicubic,x_nearest,x_bilin,y_pred], feed_dict={x1: mnist.train.images[:show_num], x: batch_xt})
    
    f, a = plt.subplots(6, 6, figsize=(6, 6))
    for i in range(0,6):
        ORI = np.reshape(mnist.test.images[i], (1200, 1200)) 
        PRE1 = np.reshape(encode_s[i], (600, 600)) 
        PRE2 = np.reshape(encode_b[i], (1200, 1200))
        PRE3 = np.reshape(encode_n[i], (1200, 1200))
        PRE4 = np.reshape(encode_bi[i], (1200, 1200))
        PRE5 = np.reshape(y_predv[i], (1200, 1200))
        scipy.misc.imsave('D:/fiveresults/two-0.001-500/ori'+str(i)+'.jpg', ORI)
        scipy.misc.imsave('D:/fiveresults/two-0.001-500/si'+str(i)+'.jpg', PRE1)
        scipy.misc.imsave('D:/fiveresults/two-0.001-500/bi'+str(i)+'.jpg', PRE2)
        scipy.misc.imsave('D:/fiveresults/two-0.001-500/ni'+str(i)+'.jpg', PRE3)
        scipy.misc.imsave('D:/fiveresults/two-0.001-500/bii'+str(i)+'.jpg', PRE4)
        scipy.misc.imsave('D:/fiveresults/two-0.001-500/predv'+str(i)+'.jpg', PRE5)
        a[0][i].imshow(ORI, cmap ='gray')
        a[1][i].imshow(PRE1, cmap ='gray')
        a[2][i].imshow(PRE2, cmap ='gray')
        a[3][i].imshow(PRE3, cmap ='gray')
        a[4][i].imshow(PRE4, cmap ='gray')
        a[5][i].imshow(PRE5, cmap ='gray')
    plt.show()
