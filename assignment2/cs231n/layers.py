import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)

  """
  N, D = x.shape[0], np.prod(x.shape[1:])

  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_rows = x.reshape(N, D)
  out = np.dot(x_rows, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  cache = (x, w, b)

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
  x, w, b = cache

  N, D = x.shape[0], np.prod(x.shape[1:])
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  x_rows = x.reshape(N, D)

  dx = np.dot(dout, w.T)
  dw = np.dot(x_rows.T, dout)
  db = dout.sum(axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx.reshape(*x.shape), dw, db

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
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  mask = np.ones_like(x)
  mask[x < 0] = 0

  out = mask * x
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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

  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  mask = np.ones_like(dout)
  mask[x < 0] = 0

  dx = mask * dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

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
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride, pad = conv_param['stride'], conv_param['pad']

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  # Padding
  H += 2*pad
  W += 2*pad

  H_, W_ = (H-HH)/stride + 1, (W-WW)/stride + 1

  out = np.zeros((N, F, H_, W_))
  for k, img in enumerate(x):
    # Pad with zeros
    x_padded = np.pad(img, ([0], [pad], [pad]), mode='constant', constant_values=0)

    # Activations for single image
    a = np.zeros((F, H_, W_))
    for i, ii in enumerate(range(0, H-HH+1, stride)):
      for j, jj in enumerate(range(0, W-WW+1, stride)):
        x_ = x_padded[:, ii:ii+HH, jj:jj+WW]
        
        convolved = x_ * w # x_ broadcasted to multiply all filters
        filter_sums = convolved.sum(axis=(1, 2, 3)) + b # sum up convolutions from all filters
        a[:, i:i+1, j:j+1] = filter_sums.reshape(F, 1, 1) # give sums depth

    out[k] = a # fill in activations for this image
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)

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
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache

  S, pad = conv_param['stride'], conv_param['pad']

  N, C, H, W = x.shape
  N, F, H_, W_ = dout.shape
  F, C, HH, WW = w.shape

  # Padding
  H += 2*pad
  W += 2*pad

  dx, dw, db = np.zeros((N, C, H, W)), np.zeros((F, C, HH, WW)), np.zeros(F)
  #
  # Loop over pairs of (image, activation) gradient pairs
  #
  for k, (img, da) in enumerate(zip(x, dout)):
    #
    # Compute gradients for this pair
    #
    x_padded = np.pad(img, ([0], [1], [1]), mode='constant', constant_values=0)
    for i in range(H_):
      for j in range(W_):
        da_ = da[:, i:i+1, j:j+1] # activations by all the filters for this little segment
        idx, jdx = S*i, S*j # retrive coordinates back in the image
        x_ = x_padded[:, idx:idx+HH, jdx:jdx+WW] # slice of original image

        db += da_.flatten()
        full_da = np.ones((F, C, HH, WW)) * da_.reshape(F, 1, 1, 1) # broadcast to achieve dim of scores
        dx[k, :, idx:idx+HH, jdx:jdx+WW] += np.sum(w*full_da, axis=0)
        dw += x_ * full_da # x_padded broadcasted to multiply all filters
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx[:, :, pad:H-pad, pad:W-pad], dw, db # remove padding

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
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  N, C, H, W = x.shape
  pooled_height, pooled_width = (H-pool_height)/stride + 1, (W-pool_width)/stride + 1

  out = np.zeros((N, C, pooled_height, pooled_width))
  for k, img in enumerate(x):
    #
    # Max pools for single activation volume
    #
    a = np.zeros((C, pooled_height, pooled_width))
    for i, ii in enumerate(range(0, H-pool_height+1, stride)):
      for j, jj in enumerate(range(0, W-pool_width+1, stride)):
        x_ = img[:, ii:ii+pool_height, jj:jj+pool_width] # extract little volume piece

        maximum = x_.max(axis=(1, 2), keepdims=True) # maximum along the slices
        a[:, i:i+1, j:j+1] = maximum

    out[k] = a # fill in activations for this image
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)

  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x

  """
  x, pool_param = cache

  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  S = pool_param['stride']

  N, C, H, W = x.shape
  N, F, pooled_height, pooled_width = dout.shape

  dx = np.zeros_like(x)
  #
  # Loop over pairs of (image, activation-gradient) pairs
  #
  for k, (img, da) in enumerate(zip(x, dout)):
    #
    # Compute gradients for this pair
    #
    dimg, dcube = np.zeros_like(img), np.zeros((F, pool_height, pool_width))
    for i in range(pooled_height):
      for j in range(pooled_width):
        idx, jdx = S*i, S*j # coordinates in image-space
        x_ = img[:, idx:idx+pool_height, jdx:jdx+pool_width] # slice of original image
        dcube = np.zeros((F, pool_height, pool_width))

        maximums = x_.max(axis=(1, 2), keepdims=True) # maximums in each of the slices
        dcube[x_ == maximums] = da[:, i, j] # only let the gradient through these maximums
        
        dimg[:, idx:idx+pool_height, jdx:jdx+pool_width] += dcube

    dx[k] = dimg

  return dx

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
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
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

