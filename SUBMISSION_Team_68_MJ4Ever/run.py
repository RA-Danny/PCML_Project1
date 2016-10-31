# coding: utf-8

# # 0. Imports
import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from implementations import *
from proj1_helpers import *


print("TEAM: 68_MJ4Ever")



# # 1. Load the training data into feature matrix, class labels, and event ids
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
print("Loading data done")




# # 2. Cleaning of data and visualisation

# From the dataset, we can see that the 22th column is name **PRI_jet_num**. We consider that this is a type of test. Therefore, we split the dataset according to this value.
# This step will result in **4 matrices** of data, one for each value of the 22th column.

y_sort, tX_sort, ids_sort = split_over_column(y, tX, ids, 22)


# With this new data, we perform two cleaning steps:
# <ul>
#     <li>
#         The rank of these matrices may be smaller than its size. We noticed that some columns have the same value for each row. Therefor we delete these columns, as they bring no value.
#     </li>
#     
#     <li>
#         In the data given, some cells have a value set to -999, meaning that a problem occured. In order to avoid the noise cause by this value, we decided to replace all -999 cells by the mean of the others values in the same column.
#     </li>
#     
#     <li>
#         After having remove the noise cause by -999 values, we standardize all columns.
#     </li>
# </ul>


# Cleaning 
tX_sort_clean = clean_unique_values_columns(tX_sort)
tX_stand = remove_and_standardize(tX_sort_clean)



# We achieve better results by deleting some rows in each matrices.
column_to_remove = [[10],[14],[6],[11]]
tX_clean = remove_column(tX_stand,column_to_remove)

print("Cleaning data done")





# # 3. Compute the weight

# The degrees and lambdas used for each data matrix
best_degrees = [11,11,9,9]
best_lambdas = [0.00024,0.000043,0.0012, 0.0005]

weights = []
for i in range(len(tX_clean)):
    x_tr = tX_clean[i]
    x_train_poly = build_poly(x_tr,best_degrees[i])
    w, mse_train = ridge_regression(y_sort[i],x_train_poly, best_lambdas[i])
    weights.append(w)
    

print("Compute weights	 done")





# # 4. Generate predictions and save ouput in csv format for submission:
print("Apply our prediction to the test data...")
# Retrieve test data
DATA_TEST_PATH = '../data/test.csv' 
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


# Compute prediction
y_sort, tX_sort, ids_sort = split_over_column(y_test, tX_test, ids_test, 22)
tX_sort_clean = clean_unique_values_columns(tX_sort)
tX_stand = remove_and_standardize(tX_sort_clean)
tX_clean = remove_column(tX_stand,column_to_remove)

y_pred_all = []
for i in range(len(weights)):
    w = weights[i]
    tX_test_clean_poly = build_poly(tX_clean[i], best_degrees[i])
    y_pred = predict_labels(w, tX_test_clean_poly)
    y_pred_all.append(y_pred)
 
   
# Since we have divided the data into 4 matrices, the indexes aren't the same as initially
# Need to re-align the prediction with the corresponding index
ids_test = np.concatenate(ids_sort).ravel()
y_pred = np.concatenate(y_pred_all).ravel()


# Create submission csv file
OUTPUT_PATH = 'submission.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print("Submission csv file created")

