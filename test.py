import tensorflow as tf


label = [0, 0, 1]
logits = [1.0, 1.5, 3.3]
softmax = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)

with tf.Session() as sess:
    res1 = sess.run([softmax])
    print(res1)
    res2 = sess.run([cross_entropy])
    print(res2)



