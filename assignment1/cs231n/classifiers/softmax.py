import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  (num_class, D), (D, num_train) = (W.shape, X.shape)
  class_scores = np.dot(W, X)
  
  # Subtract maximum unnormalized score from each set of class scores
  for i in range(num_train):
    max_class_score = np.max(class_scores[:, i])
    for j in range(num_class):
      class_scores[j, i] -= max_class_score
    
  # Compute softmax and update gradient
  for i in range(num_train):
    normalization_term = sum(np.exp(class_score) for class_score in class_scores[:, i])
    for j in range(num_class):
      class_scores[j, i] = np.exp(class_scores[j, i]) / normalization_term
      # Thanks again to MyHumbleSelf for making me examine this further and discover a bug in my derivation of the softmax gradient!
      dW[j] += (class_scores[j, i] - (j==y[i])) * X[:, i]
    
  # Compute cross-entropy errors and total loss from that
  losses = [np.log(class_scores[y[i], i]) for i in range(num_train)]
  loss = -sum(losses) / num_train

  # Add regularization to loss and normalize dW
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
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

  # Compute class scores
  (num_class, D), (D, num_train) = W.shape, X.shape
  class_scores = np.dot(W, X)

  # Softmax them
  e_x = np.exp(class_scores - class_scores.max(axis=0))
  class_scores = e_x / e_x.sum(axis=0)
  
  # Create mask of ys
  gold_class_matrix = np.zeros((num_class, num_train))
  gold_class_matrix[y, range(num_train)] = 1
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Cross entropy loss
  loss = -(gold_class_matrix * np.log(class_scores)).sum()
    
  # Add regularization and normalize
  loss += 0.5 * reg * np.sum(W * W)
  loss /= num_train
    
  # Gradients
  augmented_scores = class_scores - gold_class_matrix
  (num_class, num_train), (num_train, D) = augmented_scores.shape, X.T.shape
  dW = np.dot(augmented_scores, X.T)
    
  # Add regularization and normalize
  dW += reg * W
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
