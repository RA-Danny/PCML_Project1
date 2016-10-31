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



def split_data(y, x, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed (set an 'order' in the random variable (not so random so))
    np.random.seed(seed)
    
    # The split data will be randomnized
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    
    size_data = int(len(x)*ratio) # cast to 'int' in order to avoid ipython warning
    
    # Randomnize the order in x and y matrices
    x_shuffle = x[indexes]
    y_shuffle = y[indexes]
    
    # Train
    x_train = x_shuffle[:size_data]
    y_train = y_shuffle[:size_data]
    
    # Test
    x_test = x_shuffle[size_data:]
    y_test = y_shuffle[size_data:]
    
    return x_train, x_test, y_train, y_test



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



def split_over_column(y, tX, ids, columnNbr):
    """
       Split the data for all unique values in a specified column.
        
       Return a list containing all matrices obtained by the split operation
    """
    
    y_sort = []
    tX_sort = []
    ids_sort = []

    uniques = np.unique(tX[:,columnNbr])
    for i in uniques:
        idx = np.where(tX[:,columnNbr]==i)[0]
        
        y_sort.append(y[idx])  
        tX_sort.append(tX[idx,:])
        ids_sort.append(ids[idx])

    return y_sort, tX_sort, ids_sort



def clean_unique_values_columns(tx_arr):
    """Remove all columns in a matrix containing an unique value for each row"""
    
    tX_sort_clean = []
    
    for x in tx_arr:
        col_to_keep = []
        
        for i, col in enumerate(x.T):
            nbr_of_unique = np.unique(col).shape[0]
            if nbr_of_unique != 1:
                col_to_keep.append(i)
        
        tX_sort_clean.append(x[:,col_to_keep])

    return tX_sort_clean;



def remove_and_standardize(tx):
    """Change all values containg -999 by the mean of its column and standardize the column after"""
    
    tX_stand = []
    
    for x in tx:
        #remove -999 to the mean of the colum
        for column in x.T:
            clean = column[np.where(column != -999)]
            if len(clean) != 0:
                mean = np.mean(clean)
                column[np.where(column == -999)] = mean

        standard_data = standardize(x)
        tX_stand.append(standard_data[0])
    
    return tX_stand



def split_data_for_train_test(y, x, ratio):
    """Split all the matrices into train and test ones"""
    
    x_train = []
    x_test  = []
    y_train = []
    y_test  = []
    
    for i in range(0, len(y)):
        x_tr, x_te, y_tr, y_te = split_data(y[i],x[i],ratio)

        x_train.append(x_tr)
        x_test.append(x_te)
        y_train.append(y_tr)
        y_test.append(y_te)
        
    return x_train, x_test, y_train, y_test




def k_fold_means_ridge(y, x, degree, lambda_, k_fold, idx):
    """Cross-validation with ridge regression algorithm and k-fold """
    train_mse = []
    cv_mse = []

    #k fold
    for k in range(k_fold):
        x_cv = x[idx[k]]
        y_cv = y[idx[k]]

        #remove
        rem_indice = idx[~(np.arange(idx.shape[0])==k)]

        #set them in a vector
        rem_indice = rem_indice.reshape(-1)
        x_train_fold = x[rem_indice]
        y_train_fold = y[rem_indice]

        #get the ones in the matrix
        x_train_poly = build_poly(x_train_fold,degree)
        x_cv_poly = build_poly(x_cv,degree)

        mse_train,w = ridge_regression(y_train_fold, x_train_poly, lambda_)
        mse_cv = compute_loss(y_cv,x_cv_poly,w)

        train_mse.append(mse_train)
        cv_mse.append(mse_cv)

    return np.mean(train_mse), np.mean(cv_mse)



def k_fold_means_logistic(y, x, lambda_, k_fold, idx, max_iters, gamma):
    """Cross-validation with logistic ridge regression algorithm and k-fold """
    
    train_mse = []
    cv_mse = []

    #k fold
    for k in range(k_fold):
        x_cv = x[idx[k]]
        y_cv = y[idx[k]]

        #remove
        rem_indice = idx[~(np.arange(idx.shape[0])==k)]

        #set them in a vector
        rem_indice = rem_indice.reshape(-1)
        x_train_fold = x[rem_indice]
        y_train_fold = y[rem_indice]

        
        w_initial = np.random.uniform(low = -0.05,high = 0.05, size = np.shape(x_train_fold)[1])
           
        mse_train,w = reg_logistic_regression(y_train_fold, x_train_fold, lambda_, w_initial, max_iters, gamma)
        mse_cv = compute_loss(y_cv,x_cv,w)
        
        
        train_mse.append(mse_train)
        cv_mse.append(mse_cv)

    return np.mean(train_mse), np.mean(cv_mse)