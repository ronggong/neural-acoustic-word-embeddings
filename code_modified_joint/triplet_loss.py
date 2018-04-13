import tensorflow as tf
# import numpy as np


def cos_sim(x, y):
    x_dot_y = tf.reduce_sum(tf.multiply(x, y), reduction_indices=1)
    x_dot_x = tf.reduce_sum(tf.square(x), reduction_indices=1)
    y_dot_y = tf.reduce_sum(tf.square(y), reduction_indices=1)
    return tf.divide(x_dot_y, tf.multiply(tf.sqrt(x_dot_x), tf.sqrt(y_dot_y)))


# def triplet_hinge(anchor, same, diff, margin):
#     return tf.maximum(0., margin + cos_sim(anchor, diff) - cos_sim(anchor, same))

def triplet_hinge(anchor, same, diff, margin):
    return tf.maximum(0., margin + (1.0 - cos_sim(anchor, same))/2.0 - (1.0 - cos_sim(anchor, diff))/2.0)


def triplet_loss(logits, same_partition, diff_partition, margin=0.3):
    logits = tf.segment_mean(logits, same_partition)
    batch_size = tf.reduce_max(diff_partition) + 1
    anchor, same, diff = tf.split(logits, [batch_size, batch_size, -1])
    anchor = tf.gather(anchor, diff_partition)
    same = tf.gather(same, diff_partition)
    losses = triplet_hinge(anchor, same, diff, margin)
    losses = tf.segment_max(losses, diff_partition)
    return tf.reduce_mean(losses)

#
# def test():
#     x = tf.placeholder(tf.float32, [32, 80])
#     y = tf.placeholder(tf.float32, [32, 80])
#
#     z = cos_sim(x, y)
#
#     ins_1 = np.random.rand(32,80)
#     ins_2 = np.random.rand(32,80)
#     sess = tf.Session()
#     haha = sess.run(z, feed_dict={x: ins_1, y: ins_2})
#     import pdb;pdb.set_trace()
#
# if __name__ == '__main__':
#     test()