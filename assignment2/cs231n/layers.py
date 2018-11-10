from builtins import range
import numpy as np
from scipy.stats import bernoulli


def affine_forward(x, w_, b_):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    xc = x.copy()
    N = x.shape[0]
    D = int(np.product(x.shape) / x.shape[0])
    M = np.prod(b_.shape)
    xc.shape = (N, D)
    b_.shape = (1, M)
    out = np.dot(xc, w_) + np.repeat(b_, N, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w_, b_)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x_, w_, b_ = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x_.shape[0]
    D = int(np.product(x_.shape) / N)
    # ------------- get gradient of x ----------------------
    dx = np.dot(dout, np.transpose(w_))
    dx.shape = x_.shape
    # ------------- get gradient of w ----------------------
    xc = x_.copy()
    xc.shape = (N, D)
    dw = np.dot(np.transpose(xc), dout)
    # ------------- get gradient of b ----------------------
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x.copy()
    out[out <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout.copy()
    dx[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.setdefault('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.setdefault('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        xscaled = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + eps)
        # print('gamma shape: {}'.format(gamma.shape))
        # print('xscaled shape: {}'.format(xscaled.shape))
        # print('beta shape: {}'.format(beta.shape))
        out = gamma * xscaled + beta
        running_mean = momentum * bn_param['running_mean'] + (1-momentum) * sample_mean
        running_var = momentum * bn_param['running_var'] + (1-momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # running_mean = bn_param['running_mean']
        # running_var = bn_param['running_var']
        xscaled = (x - running_mean) / (np.sqrt(running_var) + eps)
        out = gamma * xscaled + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    cache = (x, gamma, beta, bn_param)
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    (x, gamma, beta, bn_param) = cache

    # ----------------- get component for calculate dx ------------------------
    eps = 1e-7
    x_std = np.std(x, axis=0) + eps
    x_mean = np.mean(x, axis=0)
    x_diff = x - x_mean
    x_dout = (x_diff * dout).mean(axis=0)
    dx = - dout.mean(axis=0) * np.power(x_std, -1) - x_diff * x_dout * np.power(x_std, -3) + dout * np.power(x_std, -1)
    # print('gamma')
    # print(gamma)
    dx = dx * gamma
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    xscaled = (x - x_mean) / (x_std + eps)
    dgamma = np.sum(dout * xscaled, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        d_ = np.prod(x.shape)
        mask = np.array(bernoulli.rvs(1-p,size=d_))
        mask.shape = x.shape
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x * (1-p)
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param['stride']
    pad = conv_param['pad']
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    # --------------------------- padded x dimension --------------------------
    x_pad = np.zeros((N, C, H+2*pad, W+2*pad))
    x_pad[:,:,pad:-pad, pad:-pad] = x
    # --------------------------- out dimension -------------------------------
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))
    w_ = w
    # ---------------------------
    for f in range(0, F):
        for i in range(0, H_out):
            for j in range(0, W_out):
                i_start = i*stride
                j_start = j*stride
                # print('i_start, j_start: {}, {}'.format(i_start, j_start))
                x_ = x_pad[:,:,i_start:i_start+HH, j_start:j_start+WW] * w_[f]
                x_sum = np.apply_over_axes(np.sum, x_, [1,2,3]) + b[f]
                out[:, f:f+1, i:i+1, j:j+1] = x_sum

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w_, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    (x, w_, b_, conv_param) = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    # ----------------------- padding -----------------------------------------
    (N, C, H, W) = x.shape
    x_pad = np.zeros((N, C, H+2*pad, W+2*pad))
    x_pad[:,:,pad:-pad, pad:-pad] = x
    (_, _, HH, WW) = w_.shape
    (N, F, H_out, W_out) = dout.shape

    # -----------------------
    dw = np.zeros(w_.shape)
    dx = np.zeros(x_pad.shape)
    db = np.zeros(b_.shape)
    for f in range(0, F):
        # print('f is {}'.format(f))
        for i in range(0, H_out):
            # print('i is {}'.format(i))
            for j in range(0, W_out):
                i_start = i*stride
                j_start = j*stride
                # print('j is {}'.format(j))
                # print('i_start is {}'.format(i_start))
                # print('j_start is {}'.format(j_start))
                # print('i_start, j_start: {}, {}'.format(i_start, j_start))
                x_window_ij = x_pad[:,:,i_start:i_start+HH, j_start:j_start+WW]
                dout_f_ij = dout[:, f:f+1, i:i+1, j:j+1]
                dw[f] = (x_window_ij * dout_f_ij).sum(axis=0) + dw[f]
                w_high_dimension = w_[f].copy()
                w_high_dimension.shape = tuple([1] + list(dw[f].shape))
                dx_ij = dout_f_ij * w_high_dimension
                dx[:,:,i_start:i_start+HH, j_start:j_start+WW] = dx[:,:,i_start:i_start+HH, j_start:j_start+WW] + dx_ij
        db[f] = dout[:,f,:,:].sum()
    # ------------------ get dx of original x dimension -----------------------
    dx = dx[:, :, 1:-1, 1:-1]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    (N, C, H, W) = x.shape
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_out, W_out))
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    i_start = i*stride
                    j_start = j*stride
                    x_window = x[n,c,i_start:i_start+stride, j_start:j_start+stride]
                    out[n, c, i, j] = x_window.max()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

def _convert_index(flat_index, dim_tuple):
    dim_array = np.array(dim_tuple)
    matrix_index = [0] * len(dim_array)
    for d in range(len(dim_array)):
        d_dimension = np.prod(dim_array[d+1:])
        index_d = flat_index // d_dimension
        matrix_index[d] = index_d
        flat_index = flat_index - index_d * d_dimension
    return matrix_index

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    (x, pool_param) = cache
    dx = np.zeros(x.shape)
    (N, C, H_out, W_out) = dout.shape
    stride = pool_param['stride']
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    dout_tmp = dout[n, c, i, j]
                    i_start = i*stride
                    j_start = j*stride
                    x_window = x[n,c,i_start:i_start+stride, j_start:j_start+stride]
                    max_index = x_window.argmax()
                    [max_i, max_j] = _convert_index(max_index, (stride, stride))
                    dx[n, c, i_start+max_i, j_start+max_j] = dout_tmp

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    ch = x.shape[1]
    d_ = np.prod(x.shape) // x.shape[1]
    x_flat = np.zeros((d_, ch))
    for i in range(x.shape[1]):
        tmp = x[:,i,:,:]
        x_flat[:,i] = tmp.flatten()
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)

    out = np.zeros(x.shape)
    for i in range(x.shape[1]):
        tmp = out_flat[:,i]
        tmp.shape = (x.shape[0], x.shape[2], x.shape[3])
        out[:,i,:,:] = tmp.copy()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    ch = dout.shape[1]
    d_ = np.prod(dout.shape) // dout.shape[1]
    dout_flat = np.zeros((d_, ch))
    for i in range(dout.shape[1]):
        tmp = dout[:,i,:,:]
        dout_flat[:,i] = tmp.flatten()
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    dx = np.zeros(dout.shape)
    for i in range(dout.shape[1]):
        tmp = dx_flat[:,i]
        tmp.shape = (dout.shape[0], dout.shape[2], dout.shape[3])
        dx[:,i,:,:] = tmp.copy()
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
