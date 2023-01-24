import numpy as np
from decimal import *

def gaussian_dec(x, m, var):
    '''
    Computes the Gaussian pdf value at x for decimal inputs.
    x: value at which the pdf is computed (Decimal type)
    m: mean (Decimal type)
    var: variance
    '''    
    p = np.exp(-(x-m)**Decimal(2)/(Decimal(2)*Decimal(var)))/(np.sqrt(Decimal(2)*Decimal(np.pi)*Decimal(var)))
    return p


def gaussian(x, m, var):
    '''
    Computes the Gaussian pdf value at x.
    x: value at which the pdf is computed (float)
    m: mean (float)
    var: variance
    '''
    p = np.exp(-(x-m)**2/(2*var))/(np.sqrt(2*np.pi*var))
    return p


def bayesian_update(L, mu):
    '''
    Computes the Bayesian update.
    L: likelihoods matrix
    mu: beliefs matrix
    '''
    aux = L*mu
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu


def DKL(m, n, dx):
    '''
    Computes the KL divergence between m and n.
    m: true distribution in vector form
    n: second distribution in vector form
    dx : sample size
    '''
    mn = m / n
    mnlog = np.log(mn)
    return np.sum(m * dx * mnlog)


def DKL_dec(m, n, dx):
    '''
    Computes the KL divergence between m and n for Decimal inputs.
    m: true distribution in vector form (array of Decimal)
    n: second distribution in vector form (array of Decimal)
    dx : sample size (Decimal type)
    '''
    mn = m / n
    mnlog= [x.ln() for x in mn]
    return np.sum(m * dx * mnlog)


def decimal_array(arr):
    '''
    Converts an array to an array of Decimal objects.
    arr: array to be converted
    '''
    if len(arr.shape)==1:
        return np.array([Decimal(y) for y in arr])
    else:
        return np.array([[Decimal(x) for x in y] for y in arr])

    
def partial_info_d(mu_0, csi, A_dec, L, N_ITER, tx=0, self_aware = False, partial=True, memo=False, id_matrix=None):
    '''
    Executes the social learning algorithm for a discrete family of likelihoods.
    mu_0: initial beliefs
    csi: observations
    A_dec: Combination matrix (Decimal type)
    L: likelihoods matrix
    N_ITER: number of iterations
    tx: transmitted hypothesis (can be a numerical value or 'max')
    self_aware: self-awareness flag
    partial: partial information flag
    id_matrix: identifiability matrix indicating undistinguishable hypotheses
    '''
    (N, M) = mu_0.shape

    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = L[:, csi[:,i]].T
        if id_matrix is not None:
            L_i = transf_identif(L_i, id_matrix)

        psi = bayesian_update(L_i, mu)
        decpsi = np.array([[x.ln() for x in y] for y in psi])

        if partial:
            if tx=='max': # share theta max
                aux = np.array([[Decimal(1) for x in y] for y in mu])*(1-np.max(psi,axis=1))[:,None]/(M-1)
                aux[np.arange(N), np.argmax(psi,axis=1)]=np.max(psi,axis=1)
            else: # share tx
                if memo:
                    aux = (1-psi[:, tx])[:, None] * mu / (1-mu[:, tx])[:, None]
                else:
                    aux = np.array([[Decimal(1) for x in y] for y in mu])*(1-psi[:,tx])[:, None]/(M-1)
                aux[np.arange(N), np.ones(N, dtype=int)*tx]=psi[:,tx]
            decaux = np.array([[x.ln() for x in y] for y in aux])


        if partial:
            if not self_aware: # Without non-cooperative term:
                mu = np.exp((A_dec.T).dot(decaux))/np.sum(np.exp((A_dec.T).dot(decaux)),axis =1)[:,None]
            else: # With non-cooperative term:
                mu = np.exp(np.diag(np.diag(A_dec)).dot(decpsi)+(A_dec.T-np.diag(np.diag(A_dec))).dot(decaux))/np.sum(np.exp(np.diag(np.diag(A_dec)).dot(decpsi)+(A_dec.T-np.diag(np.diag(A_dec))).dot(decaux)),axis =1)[:,None]
        else:
            mu = np.exp((A_dec.T).dot(decpsi))/np.sum(np.exp((A_dec.T).dot(decpsi)),axis =1)[:,None]

        MU.append(mu)
    return MU


def partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=0, self_aware=False, partial=True, id_matrix=None):
    '''
    Executes the social learning algorithm.
    mu_0: initial beliefs
    csi: observations
    thetadec: vector of means for the Gaussian likelihoods (array of Decimals)
    vardec: variance of Gaussian likelihoods (Decimal type)
    A_dec: Combination matrix (Decimal type)
    N_ITER: number of iterations
    tx: transmitted hypothesis (can be a numerical value or 'max')
    self_aware: self-awareness flag
    partial: partial information flag
    id_matrix: identifiability matrix indicating indistinguishable hypotheses
    '''
    (N, M) = mu_0.shape

    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([gaussian_dec(csi[:,i], t, vardec) for t in thetadec]).T
        if id_matrix is not None:
            L_i = transf_identif(L_i, id_matrix)

        psi = bayesian_update(L_i, mu)
        decpsi = np.array([[x.ln() for x in y] for y in psi])
        if partial:
            if tx=='max': # share theta max
                aux = np.array([[Decimal(1) for x in y] for y in mu])*(1-np.max(psi,axis=1))[:,None]/(M-1)
                aux[np.arange(N), np.argmax(psi,axis=1)]=np.max(psi,axis=1)
            else: # share tx
                aux = np.array([[Decimal(1) for x in y] for y in mu])*(1-psi[:,tx])[:, None]/(M-1)
                aux[np.arange(N), np.ones(N, dtype=int)*tx]=psi[:,tx]
            decaux = np.array([[x.ln() for x in y] for y in aux])

        if partial:
            if not self_aware: # Without non-cooperative term:
                mu = np.exp((A_dec.T).dot(decaux))/np.sum(np.exp((A_dec.T).dot(decaux)),axis =1)[:,None]

            else: # With non-cooperative term:
                mu = np.exp(np.diag(np.diag(A_dec)).dot(decpsi)+(A_dec.T-np.diag(np.diag(A_dec))).dot(decaux))/\
                     np.sum(np.exp(np.diag(np.diag(A_dec)).dot(decpsi)+(A_dec.T-np.diag(np.diag(A_dec))).dot(decaux)),axis =1)[:,None]
        else:
            mu = np.exp((A_dec.T).dot(decpsi))/np.sum(np.exp((A_dec.T).dot(decpsi)),axis =1)[:,None]


        MU.append(mu)
    return MU


def transf_identif(L, id_matrix):
    '''
    Extract the likelihoods given an identifiability matrix.
    id_matrix: identifiability matrix indicating indistinguishable hypotheses
    '''
    X = L.copy()
    for k in range(len(L)):
        X[k] = L[k][id_matrix[k]]
    return X