import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization snp.exp(np.dot(x_i, W[:,jj])) * x_i / np.sum(np.exp(np.dot(x_i, W)))trength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    # process i-th data
    x_i = X[i,:]
    j = y[i]  # true label of data i
    prob_i = np.exp(np.dot(x_i, W)) / np.sum(np.exp(np.dot(x_i, W)))
    loss = loss - np.log(prob_i[j])
    loss = loss + reg * np.sum(W*W)
    for jj in range(W.shape[1]):
      # one iteration update all the feature for class jj
      if jj != j:
        dW[:,jj] = dW[:,jj] + prob_i[jj] * x_i
      else :
        dW[:,jj] = dW[:,jj] + x_i * (prob_i[jj]-1)

  loss = loss / X.shape[0]
  dW = dW / X.shape[0]
  dW = dW + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  number_of_class = W.shape[1]
  number_of_training_point = X.shape[0]
  exp_matrix = np.exp(np.dot(X, W))
  exp_sum = exp_matrix.sum(axis=1)
  exp_sum.shape = (number_of_training_point, 1)

  exp_sum_matrix = np.repeat(exp_sum, number_of_class, axis=1)
  p_matrix = exp_matrix / exp_sum_matrix
  y_index = [np.array(range(len(y))), np.array([y[i] for i in range(len(y))])]
  proba_vector = p_matrix[y_index[0], y_index[1]]
  loss = -np.mean(np.log(proba_vector))
  loss = loss + reg * np.sum(W*W)

  # ----------------- get gradient -------------------------
  # gradient against w_tk, (sum_i(x_it) - sum_i(exp(s_ik) * x_it) / sum_j(exp(s_ij))
  prob_weigthed_x_feature_sum = np.dot(np.transpose(X), p_matrix)
  dW = prob_weigthed_x_feature_sum  # the case of none true class
  # construct Y_indicator matrix
  y_indicator_matrix = np.zeros((X.shape[0], W.shape[1]))  # dimension is number_of_data_point * number_of_class
  y_indicator_matrix[y_index[0], y_index[1]] = 1  # position [i, y[i]] = 1, otherwise = 0
  x_feature_agg = np.dot(np.transpose(X), y_indicator_matrix)
  dW = dW - x_feature_agg  # the case true class
  dW = dW / X.shape[0]
  dW = dW + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

