import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      if margin > 0:
        loss += margin
        # calcualte tmp_dW if margin is larger than 0
        tmp_dW = np.zeros(W.shape)
        tmp_dW[:, j] = X[i, :]
        tmp_dW[:, y[i]] = - X[i, ]
        dW = dW + tmp_dW

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW + 2 * reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  y_hat = np.dot(X, W)  # score for each class for each data
  y_index = [np.arange(len(y)), np.array([y[i] for i in range(len(y))])]  # index of true label for each training data
  y_true_label_score = y_hat[y_index[0], y_index[1]]  # one dimension array of score for true class
  y_true_label_score_array = np.array([np.repeat(y_true_label_score[i], y_hat.shape[1]) for i in range(len(y_true_label_score))])  # repeat score of true class for each class
  ones_array = np.ones(y_true_label_score_array.shape)
  # add_ones_back_true_label_array
  S_ij_matrix = (y_hat - y_true_label_score_array + ones_array)
  y_loss_array = S_ij_matrix.clip(0.0)  # hinge loss, for each class for each data, score_j - score_true_label + 1
  y_loss_sum = np.sum(y_loss_array) - y_hat.shape[0]  # total loss, aggregate for data and all classes
  penalty = reg * np.sum(np.square(W))  # regularization term
  loss = (y_loss_sum + penalty) / X.shape[0]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # derivative formula
  # j is not the true class, j' is the true class
  # delta_L_i / delta_W_tj = X_it if S_ij - S_ij' + 1 > 0 else 0
  # j is the true class:
  # delta L_i / delta_W_tj' = sum_j(indicator(S_ij - Sij' + 1) * X_it

  # derivative for j, not the true class
  S_ij_matrix_bool = np.ones(S_ij_matrix.shape)  #
  S_ij_matrix_bool[S_ij_matrix <= 0] = 0  # classes the loss is 0, which gradient is 0
  S_ij_matrix_bool[y_index[0], y_index[1]] = 0  # set the true class gradient 0
  S_ij_derivative = np.dot(np.transpose(X), S_ij_matrix_bool) / X.shape[0]  # delta_L / delta_W_tj = mean_i( delta_L_i / delta_W_tj)
  # derivative for j', the true class
  tmp_index_array = np.zeros(S_ij_matrix_bool.shape)
  # tmp_index_array[y_index[0], y_index[1]] = np.sign(S_ij_matrix).clip(0.1).sum(1)  # number of class that contribute to the gradient for i-th data
  tmp_index_array[y_index[0], y_index[1]] = S_ij_matrix_bool.sum(1)
  S_true_class_derivative = - np.dot(np.transpose(X), tmp_index_array) / X.shape[0]
  dW = S_ij_derivative + S_true_class_derivative + 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
