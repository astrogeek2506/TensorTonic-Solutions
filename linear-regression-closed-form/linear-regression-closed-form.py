import numpy as np

def linear_regression_closed_form(X, y):
    # Convert inputs to numpy arrays to enable matrix operations
    X = np.array(X)
    y = np.array(y)
    
    # Now .T and np.dot will work correctly
    xtx = np.dot(X.T, X)
    xtx_inv = np.linalg.inv(xtx)
    w = np.dot(np.dot(xtx_inv, X.T), y)
    
    return w