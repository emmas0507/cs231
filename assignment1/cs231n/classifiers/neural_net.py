from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # ----------------- expand parameter and X to incorporate b1 and b2
    X = np.c_[X, np.ones(X.shape[0])]
    b1.shape = (1, W1.shape[1])
    b2.shape = (1, W2.shape[1])
    W1 = np.r_[W1, b1]
    W2 = np.r_[W2, b2]
    # ----------------- first hidden layer ----------------------
    wx_first_layer = np.dot(X, W1)
    wx_first_layer = np.c_[wx_first_layer, np.ones(wx_first_layer.shape[0])]
    # ----------------- relu function ---------------------------
    relu_layer = wx_first_layer.copy()
    relu_layer[relu_layer < 0] = 0
    # print('relu layer output size: {}'.format(relu_layer.shape))
    # ----------------- second hidden layer ---------------------
    wx_second_layer = np.dot(relu_layer, W2)  # input feed into softmax layer
    scores = wx_second_layer
    # print('wx_second_layer size: {}'.format(wx_second_layer.shape))
    # ----------------- softmax layer ---------------------------
    exp_matrix = np.exp(wx_second_layer)
    exp_sum = exp_matrix.sum(axis=1)
    exp_sum.shape = (N, 1)
    p_matrix = exp_matrix / np.repeat(exp_sum, W2.shape[1], axis=1)  # score is probability of each class
    # print('scores size: {}'.format(scores.shape))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    y_index = [range(N), list(y)]
    loss = - np.sum(np.log(p_matrix[y_index[0], y_index[1]]))
    loss = loss / X.shape[0] + reg * (np.sum(W1[:-1,]*W1[:-1,]) + np.sum(W2[:-1,]*W2[:-1,]))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # softmax derivative
    # ------------------ add -1 to the true label class ------------
    softmax_grad = p_matrix
    softmax_grad[y_index[0], y_index[1]] = softmax_grad[y_index[0], y_index[1]] - 1
    # print('softmax_grad size {}'.format(softmax_grad.shape))
    # N * T2, T2 is number of nodes in second layer
    # ------------------ second layer gradient ---------------------
    second_layer_input = relu_layer
    w2_grad = np.dot(np.transpose(second_layer_input), softmax_grad)
    w2_grad = w2_grad / X.shape[1]
    grads['W2'] = (w2_grad[:-1, ] + reg * 2 * W2[:-1, ]).copy()
    grads['b2'] = w2_grad[-1, ].copy()
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    # todo, after adding intecept term, the size of the matrix changes
    first_layer_upstream_grad = np.dot(softmax_grad, np.transpose(W2))  # delta(L)/delta(S_1)
    first_layer_upstream_grad[wx_first_layer < 0] = 0  # add relu grad
    first_layer_upstream_grad = np.transpose(np.transpose(first_layer_upstream_grad)[:-1,])
    w1_grad = np.dot(np.transpose(X), first_layer_upstream_grad)
    w1_grad = w1_grad / X.shape[0]
    grads['W1'] = w1_grad[:-1, ] + reg * 2 * W1[:-1, ]
    grads['b1'] = w1_grad[-1, ]
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step. one step is one descend
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = int(max(num_train / batch_size, 1))

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      # print('iteration: {}'.format(it))
      if it % iterations_per_epoch == 0:
        # starting of one epoch, reshuffle X
        epoch_index = np.random.permutation(range(num_train))
      it_in_this_epoch = int(it) % int(iterations_per_epoch)
      # print('iteration in this epoch: {}'.format(it_in_this_epoch))
      X_batch = X[epoch_index[it_in_this_epoch*batch_size: (it_in_this_epoch+1)*batch_size]]
      y_batch = y[epoch_index[it_in_this_epoch*batch_size: (it_in_this_epoch+1)*batch_size]]
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] = self.params['W1'] - learning_rate * grads['W1']
      self.params['W2'] = self.params['W2'] - learning_rate * grads['W2']
      self.params['b1'] = self.params['b1'] - learning_rate * grads['b1']
      self.params['b2'] = self.params['b2'] - learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # print('learning rate: {}'.format(learning_rate))
        # Decay learning rate
        learning_rate *= learning_rate_decay


    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


