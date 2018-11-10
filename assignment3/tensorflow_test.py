# this file test tenforflow syntax
import tensorflow as tf
tf.reset_default_graph()
sess = tf.Session()
x = tf.placeholder('float')
y = tf.scalar_mul(2.0,x)

loss = tf.square(y - tf.square(x))
sess.run(loss, {x: [1.0, 2.0, 3.0, 4.0]})

grads_x = tf.gradients(loss, x)
grads_y = tf.gradients(loss, y)

[grads_x_result, grads_y_result] = sess.run([grads_x, grads_y], {x: [1.0, 2.0, 3.0, 4.0]})
