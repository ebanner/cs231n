import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    
    num_contrib_classes = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
        
      margin = scores[j] - correct_class_score + 1 # note delta = 1
    
      if margin > 0:
        loss += margin
        # Update dW_incorrect_class weights
        dW[j] += X[:, i]
        num_contrib_classes += 1
        
    # Update dW_correct_class weights
    dW[y[i]] -= num_contrib_classes * X[:, i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  dW /= num_train

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # Get dimensions (great way to foreshadow impending matrix multiplication!)
  (num_class, D), (D, num_train) = (W.shape, X.shape)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Compute class scores
  scores = np.dot(W, X)

  # Get the correct class scores and compute losses from it
  correct_class_scores = scores[y, range(num_train)]
  losses = scores - correct_class_scores + 1

  # Hinge
  losses[losses < 0] = 0

  # Zero out the losses in which y was the correct class and add up the losses
  losses[y, range(num_train)] = 0
  loss = losses.sum() / num_train
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
  # Make `losses` into a bit matrix so we know which classes in which examples contribute to gradient
  losses[losses > 0] = 1
  
  # Put in the number of bad class scores for each example and this becomes the term of each correct class' gradient
  losses[y, range(num_train)] = -losses.sum(axis=0)

  # Take the sum of outer products of the bit matrix and X itself
  #
  # There's no way I would have found this out by myself. Credit to https://github.com/MyHumbleSelf/cs231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py. Thank you for putting this up so I could study from it and learn an awesome trick!
  dW = np.dot(losses, X.T)
  dW /= num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
