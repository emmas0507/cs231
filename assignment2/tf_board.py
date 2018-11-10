import tensorflow as tf
import numpy as np
import math
from cs231n.data_utils import load_CIFAR10


tf.reset_default_graph()

# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
def complex_model(X,y,is_training):
    # --------------- specify size of the conv layer -------------------
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1
    # --------------- relu activation layer ----------------------------
    h1 = tf.nn.relu(a1)
    # --------------- spatial batch normalization layer ----------------
    h2 = tf.layers.batch_normalization(h1, training=is_training, axis=3)
    # --------------- max pooling layer with stride = 2 ----------------
    h3 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # --------------- fully connected layer, affine + relu -------------
    h3_flat = tf.reshape(h3,[-1,13*13*32])
    h4 = tf.contrib.layers.fully_connected(h3_flat, num_outputs=1024)
    # --------------- affine from 1024 to 10 ---------------------------
    W4 = tf.get_variable("W4", shape=[1024, 10])
    b4 = tf.get_variable("b4", shape=[10])
    y_out = tf.matmul(h4, W4) + b4
    return y_out

y_out = complex_model(X,y,is_training)

total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

correct_prediction = tf.equal(tf.argmax(y_out,1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.RMSPropOptimizer(1e-3)
train_step = optimizer.minimize(mean_loss)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # initiate a writer
    writer = tf.summary.FileWriter('./reports/tmp/', graph=tf.get_default_graph())

    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ######################################################
    # create a summary for our cost and accuracy, adding those to summary
    tf.summary.scalar("mean_loss", mean_loss)
    tf.summary.scalar("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()
    #######################################################

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, accuracy, training, summary_op]
    # if training_now:
    #     variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss_, accuracy_, train_, summary_ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss_*actual_batch_size)
            correct += np.sum(accuracy_)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,loss_,np.sum(accuracy_)))
            iter_cnt += 1

            # write log
            writer.add_summary(summary_, e * batch_size + i)
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss,total_correct,e+1))

    return total_loss,total_correct

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,10,train_step,True)
        # print('Validation')
        # run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
