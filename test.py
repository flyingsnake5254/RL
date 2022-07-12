import numpy as np

theta0 = np.array([
    [np.nan,1,1,np.nan],
    [np.nan,1,np.nan,1],
    [np.nan,np.nan,1,1],
    [1,1,1,np.nan],
    [np.nan,np.nan,1,1],
    [1,np.nan,np.nan,np.nan],
    [1,np.nan,np.nan,np.nan],
    [1,1,np.nan,np.nan],
])

def softmax(theta):
    beta = 1.0
    [m,n] = theta.shape
    pi = np.zeros((m,n))
    exp_theta = np.exp(beta * theta)
    print(exp_theta)
    for i in range(0,m):
        pi[i,:] = exp_theta[i,:]/np.nansum(exp_theta[i,:])
    # pi = np.nan_to_num(pi)
    return pi

p = softmax(theta0)
print(p)