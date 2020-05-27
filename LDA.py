import numpy as np


def train_lda(X, y):
    ''' Train an LDA
    Input: X data matrix with shape NxD
           y label vector with shape Nx1
           
    Output: weight vector with shape Nx1 
            bias term - real-valued '''

    # initialisations
    mu_c1 = np.mean(X[y==0.], 0)
    mu_c2 = np.mean(X[y==1.], 0)

    cov_c1 = np.cov(X[y==0.], rowvar=False)
    cov_c2 = np.cov(X[y==1.], rowvar=False)
    cov_w = 0.5*(cov_c1+cov_c2)

    cov_I = np.linalg.pinv(cov_w, hermitian=True)


    "Using Fisher's Criterion #2"
    weights = cov_I@(mu_c2 - mu_c1)
    bias = -0.5*weights@(mu_c1+mu_c2) 

    return weights, bias

def apply_lda(X_test, weights, bias):
    '''Predict the class label per sample 
    Input: X_test - data matrix with shape NxD
           weight vector and bias term from train_LDA
    Output: vector with entries 1 or 2 depending on the class'''

    y_hat = weights@X_test.T + bias #  (1,M)(M,N) + 1 = (1xN)
    temp = []
    for _ in y_hat:

        if _ >0:
            temp.append(1)
        else:
            temp.append(0)

    return np.array(temp)

def accu(y_test, y_hat_D):
	""" Test the accuracy of LDA """
    print('The Accuracy on the test set is %.2f %%' %(sum(y_test==y_hat_D)/len(y_test)*100))

