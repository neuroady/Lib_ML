import matplotlib.pyplot as plt
import numpy as np

def plotD(X_train, y_train, tiit):
    CovXtrain = np.cov(X_train, rowvar=False)
    EvecXtrain = np.linalg.eigh(CovXtrain)[1]
    
    # plt.scatter(X_train[:,0], X_train[:,1], marker="o", s=5, alpha=0.4)
    plt.scatter(X_train[y_train==0.][:,0], X_train[y_train==0.][:,1],
                color="r", marker="o", s=5, alpha=0.7)
    plt.scatter(X_train[y_train==1.][:,0], X_train[y_train==1.][:,1],
                color="b", marker="o", s=5, alpha=0.7)
    
    plt.quiver(np.mean(X_train, 0)[0], np.mean(X_train, 0)[1],
               EvecXtrain[:,0], EvecXtrain[:,1],
               color=["r", "g"], scale=6)
    
    plt.legend([r"$C_1$", r"$C_2$"])
    plt.title(tiit)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid(True, lw=0.2)
    plt.plot()


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
    print('The Accuracy on the test set is %.2f %%' %(sum(y_test==y_hat_D)/len(y_test)*100))

def tune_x1(x):
    #     return  x**3.2 # accuracy = 89.83
    return np.exp(1 / 13 * x)**3.2  # accuracy = 91.33


def tune_x2(x):
    return np.log10(x)


def pProcess(X_train):
    return np.array([tune_x1(X_train[:, 0]), tune_x2(X_train[:, 1])]).T