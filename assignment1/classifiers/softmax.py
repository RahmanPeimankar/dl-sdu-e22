from builtins import range
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
    - reg: (float) regularization strength

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]
  
    """
    --note--
    for samme måde som for svm udregnes scores for softmax
    classifieren som prikproduktet af X (input) billeded og W (vægtmatricen)
    
    X har en shape på (500,6913) 
    500 træningsbilleder med størrelse 48 x 48 x 3 + 1 for bias = 6913 
    
    W en shape på (6913,3) 6913 parametre for hver "classifier"
    én classier for hver klasser som giver en score for den respektive klasse
    for hvert eksempel. 

    Modsat svm, så normaliseres scoren for de enkelte klasser for
    softmax classifieren til en værdi mellem 0 og 1. Loss udregnes som den
    negative logaritme til den normaliserede score for den korrekte klasse.
    så længe denne ikke er 1, så vil loss akkumuleres for Softmax loss 
    funktionen.

    gradienten for softmax funktionen kan findes ved at gange softmax scoren 
    for de classifiers som giver scores for de "ukorrekte" klasser 
    med input billedet X[i], og for den classifier der giver scores
    for den korrekte klasse trækkes 1 fra scoren og ganges derefter med input 
    (X[i]). Intuitivt giver det mening, hvis scoren for de ukorrekte/forkerte 
    klasser er høj, så er gradienten stor, om omvendt, hvis scoren for den 
    korrekte klasse er høj, så er gradienten lille. Husk, når gradient descent 
    implmenteres, så trækkes gradient matricen fra den tidligere vægt matrice.
    i tilfældet her med softmax gradienten, så giver gradienten for parametrene 
    i vægtmatricen som giver scoren for de forkerte klasser en positiv værdier,
    altså, bliver disse værdier mindre. Omvendt, 
    så giver gradienten for de parametre i vægtmatricen som giver scoren 
    for den korrekte klasse negative værdier, altså bliver disse værdier højere

    De normaliserede softmax værdier er mellem 0 og 1, og hvis softmax værdien
    for den korrekte klasse stiger, så falder den samlede værdi for de resterende
    klasser tilsvarende. Derfor vil der være loss, og dermed også en gradient 
    i softmax funktionen, så længe softmax scoren for den korrekte klasse < 1
    og alle de andre klasser > 0. 
    """


    # Softmax Loss
    for i in range(num_train):
      s = scores[i] - np.max(scores[i])
      # ^trick til at sikre numerisk stabilitet således der ikke anvendes 
      # for store værdier i softmaxfunktionen nedenfor
      # https://cs231n.github.io/linear-classify/

      softmax = np.exp(s)/np.sum(np.exp(s))

      loss += -np.log(softmax[y[i]])
      # Weight Gradients

      for j in range(num_classes):
        if j == y[i]:
          continue
        dW[:,j] += X[i] * softmax[j]
      
      dW[:,y[i]] += (softmax[y[i]]-1) * X[i]
 
    # Average
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """
    prikproduktet af X og W giver scores
    på samme måde som ovenfor trækkes max scoren fra alle værdier
    for at sikre numerisk stabilitet
    """

    num_train = X.shape[0]
    scores = X.dot(W)
    scores = scores - np.max(scores, axis=1, keepdims=True)

    """
    softmax_matrix holder alle softmax værdier og loss
    udregnes ved at summere og alle værdier for den negative logarimte for den 
    korrekte klasse ved at indexere vha. a label vektoren "y"
    """
    # Softmax Loss
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    softmax_matrix = np.exp(scores)/sum_exp_scores
    loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]) )


    """
    gradienten udregnes udfra samme princip som ovenfor, men ved at udregne 
    prikproduktet mellem X og softmax matricen. For af få gradienten for de 
    parametre i vægmatricen som giver scoren for den korrekte klasse,
    så trækkes 1 først fra softmax scoren for de korrekte klasser ved at
    indexere vha. af labelvektoren "y".
    """


    # Weight Gradient
    softmax_matrix[np.arange(num_train),y] -= 1
    dW = X.T.dot(softmax_matrix)

    # Average
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
