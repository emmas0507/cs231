from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # ------------------ initial W1, W2, b1, b2 for two layer nets -------------
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # ------------- forward pass, first layer affline node output --------------
        out_first_affine, cache_first_affine = affine_forward(X, self.params['W1'], self.params['b1'])
        # ------------- forward pass, relu layer node output -----------------------
        out_relu, cache_relu = relu_forward(out_first_affine)
        # ------------- forward pass, second layer affline node output -------------
        scores, cache_second_affine = affine_forward(out_relu, self.params['W2'], self.params['b2'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, {}

        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # ------------- forward pass, compute softmax loss -------------------------
        loss, d_softmax = softmax_loss(scores, y)
        loss = loss + self.reg * 0.5 * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']))
        # ------------- backward pass, compute grads for second affine layer -------
        dx_second_affine, grads['W2'], grads['b2'] = affine_backward(d_softmax, cache_second_affine)
        grads['W2'] = grads['W2'] + self.reg * self.params['W2']
        # ------------- backward pass, compute grads for relu layer ----------------
        dx_relu = relu_backward(dx_second_affine, out_relu)
        # ------------- backward pass, compute grads for first affine layer --------
        dx_first_affine, grads['W1'], grads['b1'] = affine_backward(dx_relu, cache_first_affine)
        grads['W1'] = grads['W1'] + self.reg * self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 1e-6
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.L = len(hidden_dims) + 1

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        input_dim_hidden_dims = [input_dim] + hidden_dims + [num_classes]
        # ---------------- initiate for params for affine layer --------------------
        for i in range(1, len(input_dim_hidden_dims)):
            W_name = 'W{}'.format(i)
            b_name = 'b{}'.format(i)
            self.params[W_name] = weight_scale * np.random.randn(input_dim_hidden_dims[i-1], input_dim_hidden_dims[i])
            self.params[b_name] = np.zeros(input_dim_hidden_dims[i])

        if self.use_batchnorm:
            # ---------------- initiate for params for normalization -------------------
            for i in range(1, len(input_dim_hidden_dims)-1):
                gamma_name = 'gamma{}'.format(i)
                beta_name = 'beta{}'.format(i)
                # print('initialize beta {}'.format(beta_name))
                self.params[gamma_name] = np.random.randn(input_dim_hidden_dims[i]) + 1.0
                self.params[beta_name] = np.random.randn(input_dim_hidden_dims[i]) + 1.0

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def _one_layer_forward(self, previous_layer_out, layer_i):
        out_dict = {}
        # ----------------- forward pass, affine layer ------------------------------
        w_ = self.params['W{}'.format(layer_i)]
        b_ = self.params['b{}'.format(layer_i)]
        out_, cache_affine = affine_forward(previous_layer_out, w_, b_)
        out_dict['cache_affine'] = cache_affine
        # ----------------- forward pass, batch normalization ------------------------
        if self.use_batchnorm:
            bn_param = self.bn_params[layer_i]
            gamma = self.params['gamma{}'.format(layer_i)]
            beta = self.params['beta{}'.format(layer_i)]
            out_, cache = batchnorm_forward(out_, gamma, beta, bn_param)
            out_dict['cache_bn'] = cache
        # ----------------- forward pass, relu layer --------------------------------
        out_, cache_relu = relu_forward(out_)
        out_dict['out_relu'] = out_.copy()
        # ----------------- forward pass, dropoff layer -----------------------------
        if self.use_dropout:
            out_dropoff, cache_dropoff = dropout_forward(out_, self.dropout_param)
            out_dict['cache_dropoff'] = cache_dropoff
            out_ = out_dropoff
        out_dict['out'] = out_.copy()
        return out_dict

    def _one_layer_backward(self, dx_upstream, layer_i_forward_dict, layer_i):
        grads = {}
        # ----------------- backward pass, dropoff layer ------------------------------
        if self.use_dropout:
            cache_dropoff = layer_i_forward_dict['cache_dropoff']
            dx_upstream_dropoff = dropout_backward(dx_upstream, cache_dropoff)
            # print('dx after dropoff layer')
            # print(dx_upstream[0])
        else:
            dx_upstream_dropoff = dx_upstream
        # ----------------- backward pass, relu layer ------------------------------
        out_relu = layer_i_forward_dict['out_relu']
        dx_upstream_relu = relu_backward(dx_upstream_dropoff, out_relu)
        # print('dx after relu layer')
        # print(dx_upstream_relu[0])
        # ----------------- backward pass, batch normalization layer ---------------
        if self.use_batchnorm:
            cache_bn = layer_i_forward_dict['cache_bn']
            dx_upstream_bn, dgamma, dbeta = batchnorm_backward(dx_upstream_relu, cache_bn)
            gamma_name = 'gamma{}'.format(layer_i)
            beta_name = 'beta{}'.format(layer_i)
            grads[gamma_name] = dgamma
            grads[beta_name] = dbeta
            # print('dx after batch normalization layer')
            # print(dx_upstream_bn[0])
            # import pdb; pdb.set_trace()
        else:
            dx_upstream_bn = dx_upstream_relu
        # ----------------- backward pass, affine layer ----------------------------
        cache_affine = layer_i_forward_dict['cache_affine']
        w_name = 'W{}'.format(layer_i)
        b_name = 'b{}'.format(layer_i)
        dx_affine, grads[w_name], grads[b_name] = affine_backward(dx_upstream_bn, cache_affine)
        grads[w_name] = grads[w_name] + self.reg * self.params[w_name]
        # print('dx after affine layer')
        # print(dx_affine[0])
        return dx_affine, grads

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # --------------------- forward iteration on L-1 layers --------------------
        # L = len(self.hidden_dims)+1
        input = X
        out_dict_all_layers = {}
        for i in range(1, self.L):
            # print('forward iteration {}'.format(i))
            out_dict_i = self._one_layer_forward(input, layer_i=i)
            out_dict_all_layers[i] = out_dict_i
            input = out_dict_i['out'].copy()
            # print('out_dict_all_layers keys: {}'.format(out_dict_all_layers.keys()))

        # --------------------- compute scores from the L-th layer -----------------
        W_L = self.params['W{}'.format(self.L)]
        b_L = self.params['b{}'.format(self.L)]
        b_L.shape = (1, np.prod(b_L.shape))
        scores = np.dot(input, W_L) + np.repeat(b_L, input.shape[0], axis=0)
        cache_layer_L = (input, W_L, b_L)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                    TY                                      #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # ------------------------- softmax loss -----------------------------------
        loss, d_softmax = softmax_loss(scores, y)
        W_names = [x_ for x_ in self.params.keys() if 'W' in x_]
        loss = loss + self.reg * 0.5 * np.sum(np.array([np.sum(self.params[x_] * self.params[x_]) for x_ in W_names]))

        # ------------------------- backward, softmax grads ------------------------
        dx_upstream, grads['W{}'.format(self.L)], grads['b{}'.format(self.L)] = affine_backward(d_softmax, cache_layer_L)
        grads['W{}'.format(self.L)] = grads['W{}'.format(self.L)] + self.reg * self.params['W{}'.format(self.L)]
        for i in range(self.L-1, 0, -1):
            # print('backward iteration {}'.format(i))
            layer_i_forward_dict = out_dict_all_layers[i]
            dout_i, grads_i = self._one_layer_backward(dx_upstream, layer_i_forward_dict, i)
            dx_upstream = dout_i
            grads.update(grads_i)
            # print('grads keys: {}'.format(grads.keys()))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
