from builtins import range
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    #print(dW.shape)

    # compute the loss and the gradient

    """
    -note-
    Den linære SVM classifier forsøger at scorer den korrekte klasse højere end
    alle andre klasser med en sikkerhedsmargin (eks. 1). Hvis dette kriterie
    opfyldes vil loss for denne klassifier være == 0.
    I den naive implementering herunder udregnes klasse scores for de tre klasser
    for hvert eksemple X[i] i træningssættet hvorefter scores[y[i]] 
    for den korrekte klasse sammenlignes med scoren for de to andre klasser. 

    (y[i] indikerer index (label) i scores som indeholder scores for den 
    korrekte klasse)
    (scores[y[j]] indeholder scores for de to andre klasser y[j] != y[i])

    loss akummuleres hvis den korrekte klasse scorer ikke er højere med en
    værdi svarende til sikkerhedsmarginen for hver sammenligning. I.e. for
    værdier størrer end 0 for hver sammenligning lægges denne værdi til loss

    Gradienten for parametrene i W i forhold til resultatet af funktionen
    F(x,W) = W*x er lig x, da dette er en liniær funktion. 
    I.e hvis værdien af x ændres en smule, f.eks med 0.01, så stiger output af
    funktionen med en faktor x*0.01. 

    SVM loss funktionen kan opstilles som en 
    "computational graph" med 2 noder hvor den første node i grafen er 
    den liniære funktion W*x og den anden node er selve loss funtionen
    max(0,f(w,x)-1). Gradient descent i en graf udføres vha. af 
    backprobagation hvor gradienten "flyder" baglæns igennem grafen fra 
    output til input. For hver node i grafen opdateres parametrene i nodens
    funktion ved at trække den lokale gradient * gradienten som flyder ind i
    noden fra output * en lille værdi (eks. 0.01) fra parametrene i W 
    (da gradienten giver "retningen" som øger vædienten af output).
    Hvis den gradient som kommer ind i noden = 0, vil denne værdi 
    derfor være = 0, og der vil ikke ske nogen opdatering af parametrene. 
    
    SVM Funktionen er ikke liniær og kan strengt taget ikke differentiers
    men gradienten for SVM loss funktion er defineret som 0 
    hvis kriteriet opfyldes og ellers er gradienten = -1 for parametrene 
    som giver scoren for den korrekte klasse akummuleret for alle klasser
    hvor kriteriet ikke er opfyldt 1 for scorene for de resterende klasser. 
    Hvilket giver intutiv mening, da vi gerne vil øge værdierne 
    som giver scoren for den korrekte klasse, 
    og sænke værdierne som giver scoren for de andre
    klasser. 
    vægt-matricen er intialiseret med små random postive værdier
    og alle værdier i x er positive. 
    """ 


    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                dW[:,y[i]] += -1*X[i]
                dW[:,j] += 1*X[i]
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    scores = X.dot(W)

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    """
    ---note---
    scores for udregnes som prikproduktet af X og W
    X har en shape på (500,6913) 
    500 træningsbilleder med størrelse 48 x 48 x 3 + 1 for bias = 6913 
    
    W en shape på (6913,3) 6913 parametre for hver "classifier"
    én classier for hver klasser som giver en score for den respektive klasse
    for hvert eksempel. 

    resultatet er en matrice med shape (500,3) hvor der er en scorer for hver
    af de klasser for hvert træningseksempel

    y_scores = scores[np.arange(X.shape[0]), y].reshape(X.shape[0],1)

    scores[np.arange(X.shape[0]), y] giver en 1-d vektor med shape (500,)
    som indeholder værdierne i y (labels for hvert eksemple)

    .reshape(X.shape[0],1) konvertere denne til en 2-d vektor med shape (500,1)
    på samme måde som np.newaxis 
    kan ses ved at printe nedenstående  

    print(scores[np.arange(X.shape[0]), y].shape)
    print(scores[np.arange(X.shape[0]), y].reshape(X.shape[0],1).shape)

    hvilket gør det muligt for numpy at trække
    y_scores fra scores rækkevis hvilket giver en 500,3 matrice med forskellen
    mellem den korrekte klasse scorer og alle klassescores +1.


    margin[np.arange(X.shape[0]), y]=0 sætter de værdier på index svarende
    til label på den korrekte klasse == 0 for alle rækker i score som eller 
    ville have værdien 1. (grundet der lægges 1 til alle værdier i scores)

    loss akkumuleres for alle værdier i margin > 0 hvilket som indikerer
    kritieret for sikkerhedsmarginen ikke er opfyldt. 

    v_marg = np.maximum(0,margin) sætter alle værdier under 0 hvilket 
    indikerer at kriteriet er opfyldt til 0.

    v_marg[v_marg>0] = 1 sætter alle værdier over 0 til 1 jf. definitionen
    for gradienten er SVM loss funktionen ovenfor. 

    v_marg[np.arange(X.shape[0]), y] = -np.sum(v_marg, axis=1) akkumulere
    gradientententen for alle de tilfælde hvor kriteriet ikke er opfyldt.

    gradienten for parametrene i W med hensyn til loss udregnes ved at gange
    den globale gradient v_marg med den lokale gradient for W som == X

    """

    y_scores = scores[np.arange(X.shape[0]), y].reshape(X.shape[0],1)
    
    margin = scores+1-y_scores


    margin[np.arange(X.shape[0]), y]=0

    loss = np.sum(np.maximum(0,margin))

    loss /= X.shape[0]
    loss += reg * np.sum(W * W)

    v_marg = np.maximum(0,margin)  
    v_marg[v_marg>0] = 1
    v_marg[np.arange(X.shape[0]), y] = -np.sum(v_marg, axis=1)


    dW = X.T.dot(v_marg)

    dW /= X.shape[0]
    dW += reg * 2*W



    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
