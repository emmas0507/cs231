import tensorflow as tf

labels = tf.constant([1])
logits = tf.constant([1.0])
l = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
sess = tf.Session()
sess.run(l)

