import numpy as np
from helpers import *
from costs import *

def compute_gradient(y, tx, w):
    e = y - tx @ w;
    return -(tx.T @ e)/len(y);

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

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    # ***************************************************
    S = np.zeros([tx.shape[0],tx.shape[0]])
    S_vect = sigmoid(tx @ w)*(1-sigmoid(tx @ w))
    for i in range(tx.shape[0]):
        for j in range(tx.shape[0]): 
            if(i == j):
                S[i,j] = S_vect[i]
    return tx.T @ S @ tx

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx @ w)) - y*(tx @ w))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx @ w) - y)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    pen = lambda_*np.sum(np.power(w,2))
    loss = calculate_loss(y,tx,w) + pen
    gradient = calculate_gradient(y,tx,w) + lambda_ * 2 * w
    H = calculate_hessian(y,tx,w)
    return loss,gradient,H

def learning_by_penalized_gradient(y, tx, w, alpha, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient, H = penalized_logistic_regression(y,tx,w,lambda_)
    new_w = w - alpha*np.linalg.inv(H) @ gradient
    w = new_w
    return loss, w

def least_squares(y, tx):
    """Least square algorithm."""
    weight = np.linalg.solve(tx.T @ tx,tx.T @ y)
    mse = compute_loss(y,tx,weight)
    return mse, weight

def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w);
        loss = compute_loss(y,tx,w);
        w = w  - gamma*gradient;
        ws.append(w)
        losses.append(loss)
    return losses[max_iters-1], ws[max_iters-1]

def least_squares_SGD(y,tx,initial_w,batch_size,max_iters,gamma):
    
    batch = batch_iter(y, tx, batch_size, shuffle=True);
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch:
            gradient = compute_gradient(y_batch,tx_batch,w);
            loss = compute_loss(y_batch,tx_batch,w);
            w = w  - gamma*gradient;
            ws.append(w);
            losses.append(loss);

    return losses[max_iters -1], ws[max_iters -1]

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    m = np.shape(tx)[0]
    M = lamb*2*m*np.eye(np.shape(tx)[1])
    M[0,0] = 0; #depends on the matrix tx we give
    weight = np.linalg.solve(tx.T @ tx + M,tx.T @ y)  
    mse = compute_loss(y, tx, weight)   
    return mse,weight

    
def reg_logistic_regression(y, tx, lambda_,gamma,max_iters):
    
    initial_w = np.random.uniform(low=-0.05, high=0.05, size=tx.shape[1])
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        c = np.apply_along_axis(sigmoid,0,tx @ b)
        d = np.subtract(c,y)
        gradient = tx.T @ d
             
        grad=gradient/y.shape[0] + lambda_*np.sum(w)
        
        loss=compute_loss(y, tx, w)
        w = w - gamma*grad
        
        ws.append(np.copy(w))
        losses.append(loss)
        y_new=np.sign(np.dot(tx,w))
        
        if n_iter%1 == 0:
            print('iteration: ', n_iter, ' Test accuracy: ', float(np.sum(y==y_new))/y.shape[0], ' Loss : ', loss)
        
    return losses, ws

def logistic_regression(y, tx, gamma, max_iters):
    return reg_logisitc_regression(y, tx,0,gamma, max_iters)
    