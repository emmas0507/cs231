from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    dot_prod = np.dot(prev_h, Wh) + np.dot(x, Wx) + b
    next_h = np.tanh(dot_prod)
    cache = (x, prev_h, Wx, Wh, b, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    (x, prev_h, Wx, Wh, b, next_h) = cache
    # ------------------ tanh derivative -------------------
    d_tanh = (1 - np.power(next_h, 2)) * dnext_h
    dWx = np.dot(np.transpose(x), d_tanh)
    dWh = np.dot(np.transpose(prev_h), d_tanh)
    dx = np.dot(d_tanh, np.transpose(Wx))
    dprev_h = np.dot(d_tanh, np.transpose(Wh))
    db = d_tanh.sum(axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    (N, T, D) = x.shape
    (_, H) = h0.shape
    h = np.zeros((N, T, H))
    h_ = h0
    cache = []
    for t in range(T):
        h_, cache_ = rnn_step_forward(x[:,t,:], h_, Wx, Wh, b)
        h[:,t,:] = h_
        cache = cache + [cache_]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    (N, T, H) = dh.shape
    x_ = cache[0][0]
    D = x_.shape[-1]
    dx_T = np.zeros((N, T, D))
    dWx_T = np.zeros((D,H,T))
    dWh_T = np.zeros((H,H,T))
    db_T = np.zeros((H,T))

    # ------------------ derivative relative to h_T is zeros(N, H) ---------------
    dprev_h_ = np.zeros((N, H))

    for t in range(T-1, -1, -1):
        # (x, prev_h, Wx, Wh, b, next_h) = cache[t]
        dh_t = dh[:,t,:] + dprev_h_
        (dx_, dprev_h_, dWx_, dWh_, db_) = rnn_step_backward(dh_t, cache[t])
        dx_T[:,t,:] = dx_.copy()
        dWx_T[:,:,t] = dWx_.copy()
        dWh_T[:,:,t] = dWh_.copy()
        db_T[:,t] = db_.copy()

    dx = dx_T
    dWx = dWx_T.sum(axis=2)
    dWh = dWh_T.sum(axis=2)
    db = db_T.sum(axis=1)
    dh0 = dprev_h_.copy()
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    (N, T) = x.shape
    (V, D) = W.shape
    x_onehot = np.zeros((N, T, V))
    out = np.zeros((N, T, D))
    for n in range(N):
        x_onehot[n,range(T), x[n]] = 1
        out[n,:,:] = np.dot(x_onehot[n,:,:], W)
    cache = (x_onehot, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    (x_onehot, W) = cache
    (N, T, D) = x_onehot.shape
    (V, D) = W.shape
    dW = np.zeros(W.shape)
    for v_ in range(V):
        for d_ in range(D):
            dW[v_, d_] = np.sum(dout[:,:,d_] * x_onehot[:,:,v_])
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid_f(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    (N, D) = x.shape
    (N, H) = prev_h.shape
    x_h_combine = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    comp_i = x_h_combine[:, 0:H].copy()
    comp_f = x_h_combine[:, H:(2*H)]
    comp_o = x_h_combine[:, (2*H):(3*H)]
    comp_g = x_h_combine[:, (3*H):(4*H)]
    # gate_i = 1 / np.exp(-comp_i)
    # gate_f = 1 / np.exp(-comp_f)
    # gate_o = 1 / np.exp(-comp_o)
    gate_i = sigmoid_f(comp_i)
    gate_f = sigmoid_f(comp_f)
    gate_o = sigmoid_f(comp_o)
    gate_g = np.tanh(comp_g)
    next_c = gate_f * prev_c + gate_i * gate_g
    next_h = gate_o * np.tanh(next_c)
    cache = (x, prev_h, prev_c, Wx, Wh, b, gate_i, gate_f, gate_o, gate_g, next_h, next_c)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    (x, prev_h, prev_c, Wx, Wh, b, gate_i, gate_f, gate_o, gate_g, next_h, next_c) = cache

    (N, D) = x.shape
    (_, H_4) = Wx.shape
    # ----------------- derivative of date_o and -------------------------------
    dnext_c = dnext_c + dnext_h * gate_o * (1 - np.power(np.tanh(next_c),2))
    dgate_o = dnext_h * np.tanh(next_c)
    dprev_c = dnext_c * gate_f
    dgate_f = dnext_c * prev_c
    dgate_i = dnext_c * gate_g
    dgate_g = dnext_c * gate_i

    dcomp_o = dgate_o * gate_o * (1-gate_o)
    dcomp_f = dgate_f * gate_f * (1-gate_f)
    dcomp_i = dgate_i * gate_i * (1-gate_i)
    dcomp_g = dgate_g * (1 - np.power(gate_g, 2))

    dscores = np.column_stack((dcomp_i, dcomp_f, dcomp_o, dcomp_g))
    W = np.row_stack((Wx, Wh))
    x_h = np.column_stack((x, prev_h))
    dW = np.dot(np.transpose(x_h), dscores)
    dWx = dW[0:D,:]
    dWh = dW[D:,:]
    db = dscores.sum(axis=0)
    dx_h = np.dot(dscores, np.transpose(W))
    dx = dx_h[:, 0:D]
    dprev_h = dx_h[:,D:]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    (N, T, D) = x.shape
    (N, H) = h0.shape
    cache = []
    h = np.zeros((N, T, H))
    next_c = np.zeros(h0.shape)
    next_h = h0
    for t in range(T):
        x_ = x[:,t,:]
        c_ = next_c.copy()
        h_ = next_h.copy()
        next_h, next_c, cache_ = lstm_step_forward(x_, h_, c_, Wx, Wh, b)
        cache = cache + [cache_]
        h[:,t,:] = next_h.copy()

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    (N, T, H) = dh.shape
    (_,D) = cache[0][0].shape
    dx = np.zeros((N, T, D))
    (N, D) = cache[0][0].shape
    dWx = np.zeros((T, D, 4*H))
    dWh = np.zeros((T, H, 4*H))
    db = np.zeros((T, 4*H))
    dprev_c = np.zeros((N, H))
    dprev_h = np.zeros((N, H))
    for t in range(T-1, -1, -1):
        cache_ = cache[t]
        dh_ = dh[:,t,:] + dprev_h.copy()
        dc_ = dprev_c.copy()
        dx_, dprev_h, dprev_c, dWx_, dWh_, db_ = lstm_step_backward(dh_, dc_, cache_)
        dWx[t,:,:] = dWx_
        dWh[t,:,:] = dWh_
        db[t,:] = db_
        dx[:,t,:] = dx_

    dh0 = dprev_h.copy()
    dWx = dWx.sum(axis=0)
    dWh = dWh.sum(axis=0)
    db = db.sum(axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
