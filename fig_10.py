import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from decimal import *
from functions import *
import random
import os
import networkx as nx


# Latex settings
getcontext().prec = 300
mpl.style.use('seaborn-deep')
mpl.rcParams['text.usetex']= 'True'
mpl.rcParams['text.latex.preamble']= r'\usepackage{amsmath} \usepackage{amssymb}'
mpl.rcParams['font.family'] = 'serif'

# Figure path
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)
    
# Setup
N = 10
M = 10
N_ITER = 100

# Identifiability matrix
id_matrix = np.array([
    [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 0, 0, 0, 6, 7, 8, 9],
    [0, 1, 2, 0, 0, 0, 6, 7, 8, 9],
    [0, 1, 2, 0, 0, 0, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 0, 0, 0, 0],
    [0, 1, 2, 3, 4, 5, 0, 0, 0, 0],
    [0, 1, 2, 3, 4, 5, 0, 0, 0, 0],
    [0, 1, 2, 3, 4, 5, 0, 0, 0, 0],
     ])

# Seed
np.random.seed(0)
random.seed(0)

# Generate graph
G = np.random.choice([0.0, 1.0], size=(N, N), p=[0.5, 0.5])
G = G + np.eye(N)
G = (G > 0) * 1.0


# Create combination matrix using averaging rule 
lamb = 0.5
A = np.zeros((N, N))
for i in range(N):
    A[G[i] > 0, i] = (1 - lamb) / (np.sum(G[i]) - 1)
    A[i, i] = lamb
A_dec = decimal_array(A)

# Check that A^i converges
print('Matrix A is convergent?', np.all(np.isclose(np.linalg.matrix_power(A, 1000), np.linalg.matrix_power(A, 1001))))

# Hypotheses
theta = np.array([0.8 * i for i in range(10)])
thetadec = decimal_array(theta)
var = 1
vardec = Decimal(var)
x = np.linspace(-4, 15, 1000)

# Likelihoods
L = [gaussian(x, t, var) for t in theta]

# Initialization
np.random.seed(0)
mu_0 = np.random.rand(N, M)
mu_0 = mu_0 / np.sum(mu_0, axis = 1)[:, None]
mu_0 = decimal_array(mu_0)


# Run partial information algorithm 50 times, while sharing max belief component 
MU_MAX = []
for j in range(50):
    print(j)
    csi = thetadec[0] + np.sqrt(vardec) * decimal_array(np.random.randn(N, N_ITER))
    MU_max = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx='max', partial=True, self_aware=False, id_matrix=id_matrix)

    MU_MAX.append(MU_max)
    

# Plot belief evolution of agent 1
ag = 0
fig, ax = plt.subplots(1, 1, 
                       figsize=(6, 2.5), 
                       gridspec_kw={'bottom': 0.25, 'top': 0.8})

ax.plot([[MU_MAX[j][i][ag][0]  for j in range(50)] for i in range(N_ITER)], 
        color='C0', 
        linestyle='-', 
        linewidth=1)
ax.tick_params(axis='both', 
               which='major', 
               labelsize=12)
ax.set_ylim([-.1, 1.1])
ax.set_xlim([0, 70])
ax.set_xlabel(r'$i$', 
              fontsize=15)
ax.set_ylabel(r'$\mu_{%d,i}(\theta_0)$' % (ag + 1), 
              fontsize=15)
ax.set_title(r'Partial Information -- Sharing of $\boldsymbol{\theta}_{k,i}^{\sf max}$', 
             fontsize = 16)

fig.savefig(FIG_PATH + 'fig10.pdf', 
            bbox_inches='tight')