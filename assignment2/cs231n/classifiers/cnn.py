from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.num_layers = 3
        self.params
        # --------------------- get the first conv layer ---------------------------
        (ch, H_1, W_1) = input_dim
        pad = (filter_size - 1) // 2
        stride = 1
        W1_shape = (num_filters, ch, filter_size, filter_size)
        H_2 = int(1 + (H_1 + 2 * pad - filter_size) / stride)
        W_2 = int(1 + (W_1 + 2 * pad - filter_size) / stride)
        W2_shape = (num_filters*H_2*W_2, hidden_dim)
        W3_shape = (hidden_dim, num_classes)
        # import pdb; pdb.set_trace()
        self.params['W1'] = weight_scale * np.random.randn(np.prod(W1_shape)).reshape(W1_shape)
        self.params['W2'] = weight_scale * np.random.randn(np.prod(W2_shape)).reshape(W2_shape)
        self.params['W3'] = weight_scale * np.random.randn(np.prod(W3_shape)).reshape(W3_shape)

        self.params['b1'] = weight_scale * np.random.randn(num_filters)
        self.params['b2'] = weight_scale * np.random.randn(hidden_dim)
        self.params['b3'] = weight_scale * np.random.randn(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # ------------------------- conv layer -------------------------------------
        out_conv, cache_conv = conv_relu_forward(X, self.params['W1'], self.params['b1'], conv_param)
        # ------------------------- fully connected layer --------------------------
        out_fully, cache_fully = affine_relu_forward(out_conv, self.params['W2'], self.params['b2'])
        # ------------------------- compute score ----------------------------------
        scores, cache_affine = affine_forward(out_fully, self.params['W3'], self.params['b3'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, d_softmax = softmax_loss(scores, y)
        # ------------------------- adding regularization to loss ------------------
        W_names = [x_ for x_ in self.params.keys() if 'W' in x_]
        loss = loss + self.reg * 0.5 * np.sum(np.array([np.sum(self.params[x_] * self.params[x_]) for x_ in W_names]))

        # ------------------------- affine layer backward --------------------------
        dx_affine, grads['W3'], grads['b3'] = affine_backward(d_softmax, cache_affine)

        # ------------------------- fully connected layer backward -----------------
        dx_fully, grads['W2'], grads['b2'] = affine_relu_backward(dx_affine, cache_fully)

        # ------------------------- conv layer, backward ---------------------------
        dx_conv, grads['W1'], grads['b1'] = conv_relu_backward(dx_fully, cache_conv)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
