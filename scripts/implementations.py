import numpy as np
from helpers import *
from costs import *


# Function asked for the assignment (MSE always used as lost function)
# All these functions have the required signature, as detailed on the project description sheet

def least_squares(y, tx):
    """Least square algorithm."""

    weight = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_loss(y,tx,weight)

    return mse, weight




def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm. Only return the last value of the loop"""

    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w);
        w = w  - gamma*gradient;

    return compute_loss(y,tx,w), w




def least_squares_SGD(y, tx,initial_w, max_iters, gamma):
    """Stochastic Gradient descent algorithm. Only return the last value of the loop"""

    batch_size = 1
    batch = batch_iter(y, tx, batch_size, shuffle=True);
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch:
            gradient = compute_gradient(y_batch,tx_batch,w);
            w = w  - gamma*gradient;

    return compute_loss(y_batch,tx_batch,w), w




def ridge_regression(y, tx, lambda_):
    """Ridge regression algorithm"""

    m = np.shape(tx)
    M = 2 * m[0] * lambda_ * np.eye(m[1])
    
    M[0,0] = 0; #depends on the matrix tx we give

    weight = np.linalg.solve(tx.T @ tx + M, tx.T @ y)  
    mse = compute_loss(y, tx, weight)  
 
    return mse, weight




def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression algorithm"""
    
    # Equivalent to the regularized version with lambda set to 0
    return reg_logisitc_regression(y, tx, 0, initial_w, max_iters, gamma)




def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized Logistic regression algorithm, return only last iteration values"""
    
    w = initial_w
    for n_iter in range(max_iters):
        
        gradient = tx.T @ (sigmoid(np.dot(tx,w)) - y)
             
        #Penalized version since we have lambda_
        pen_gradient = (gradient + lambda_*np.sum(w))/(2*y.shape[0]) 

        w = w - gamma*pen_gradient

        
    return compute_loss(y, tx, w), w








# ****************** Helper function for the ones above ******************

def compute_gradient(y, tx, w):
    """ Compute the gradient (for mse)"""
    return -tx.T @ (y - tx @ w) /len(y);



def split_data(y, x, ratio):
    """split the dataset based on the split ratio."""
    arr = np.arange(len(x))
    np.random.shuffle(arr)
    x_shuffle = x[arr]
    y_shuffle = y[arr]

    x_new = np.split(x_shuffle,[len(x)*ratio])
    x_train = x_new[0]
    x_test = x_new[1]
    
    y_new = np.split(y_shuffle,[len(x)*ratio])
    y_train = y_new[0]
    y_test = y_new[1]

    return x_train,x_test,y_train,y_test



def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x_poly = np.ones(np.shape(x))
    for i in range(1,degree+1):
        x_poly = np.c_[x_poly,np.power(x,i)]

    return x_poly;

def build_poly2(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x_poly = np.ones(np.shape(x))
    for i in range(1,degree+1):
        vect = []
        for j in range(0,i+1):
            c = []
            for n in range(0,np.shape(x)[0]):
                a = np.power(x[n],i-j)
                b = np.power(x[np.shape(x)[0]-n-1],j)
                c.append(a*b)
            x_poly = np.c_[x_poly,c]

    return x_poly;



def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.power(np.e,-t))



def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx @ w)) - y*(tx @ w))



def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx @ w) - y)