import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import os
from functions import *
import random

# Latex settings
mpl.style.use('seaborn-deep')
mpl.rcParams['text.usetex'] = 'True'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
mpl.rcParams['font.family'] = 'serif'

# Figure path
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)
    
# Generate likelihoods and compute KL divergences
x = np.linspace(-4, 10, 1000)
dx = (max(x) - min(x)) / x.shape[0]
L0 = gaussian(x, 1, 1)
L1 = gaussian(x, 2, 1)
L2 = gaussian(x, 5, 1)

DKL(L0, L1, dx)
DKL(L0, (L0 + L2)/2, dx)
DKL(L0, L2, dx)
DKL(L0, (L1 + L0)/2, dx);

# Plot likelihoods
fig, ax = plt.subplots(1,1, figsize=(5,2))
h1 = ax.plot(x,
             L0, 
             color='C0')
h2 = ax.plot(x,
             L1, 
             color='C4')
h3 = ax.plot(x,
             L2, 
             color='C2')
ax.set_ylim(0, 0.5)
ax.set_xlim(-4, 10)
ax.set_xlabel(r'$\xi$', 
              fontsize=16)
ax.set_ylabel(r'$L(\xi|\theta)$', 
              fontsize=20)
ax.set_xticks(np.arange(-4,11,2))
ax.locator_params(axis='y', 
                  nbins = 4)
ax.tick_params(axis='both', 
               which='major', 
               labelsize=16)
ax.legend([r'$\theta=1$',r'$\theta=2$',r'$\theta=3$'], 
          fontsize = 16, 
          handlelength=1, 
          ncol=3, 
          loc='center', 
          bbox_to_anchor=(0.5, -0.6));

plt.savefig(FIG_PATH + 'fig4a.pdf', 
            bbox_inches='tight')


# Setup
N = 10
M = 3
N_ITER = 100

# Seed
np.random.seed(0)
random.seed(0)

# Hypotheses
theta = np.array([1.0, 2.0, 5.0])
thetadec = decimal_array(theta)
var = 1
vardec = Decimal(var)

# Graph
G = np.random.choice([0.0, 1.0],
                     size=(N, N), 
                     p=[0.5,0.5])
G = G + np.eye(N)
G = (G > 0) * 1.0

# Create combination matrix using averaging rule 
lamb = 0.1
A = np.zeros((N, N))
for i in range(N):
    A[G[i] > 0, i] = (1 - lamb) / (np.sum(G[i]) - 1)
    A[i, i] = lamb
A_dec = decimal_array(A)


# Initialization
np.random.seed(0)
mu_0 = np.random.rand(N, M)
mu_0 = mu_0 / np.sum(mu_0, axis = 1)[:, None]
mu_0 = decimal_array(mu_0)

csi=[]
for l in range(N):
    csi.append(thetadec[0] + np.sqrt(vardec) * decimal_array(np.random.randn(N_ITER)))
csi=np.array(csi)


# Run social learning algorithm

# tx=2
MU_ob2 = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=1, partial=True, self_aware=False)

# tx=3
MU_ob3 = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=2, partial=True, self_aware=False)


# Plot likelihood comparison and belief evolution for agent 1, tx=3
ag = 0
fig = plt.figure(figsize=(9, 2))
gs = gridspec.GridSpec(1, 4,
                       width_ratios=[1,1,.01,1], 
                       wspace=0.6)


ax = fig.add_subplot(gs[0, 0])
h1 = ax.plot(x,
             L0, 
             color='C0')
h2 = ax.plot(x,
             L2, 
             color='C2')
ax.set_ylim(0, 0.6)
ax.set_xlim(-4, 10)
ax.set_xlabel(r'$\xi$', 
              fontsize=16)
ax.set_ylabel(r'$L(\xi|\theta)$', 
              fontsize=16)
ax.set_xticks(np.arange(-4, 11, 4))
ax.locator_params(axis='y', 
                  nbins = 4)
ax.tick_params(axis='both', 
               which='major', 
               labelsize=14)
ax.annotate(r'$d(\theta_{{\sf TX}})={0:.2f}$'.format(DKL(L0, L2, dx)), 
            (0.5, 0.47), 
            xycoords=('axes fraction', 'data'), 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontsize=14)


ax = fig.add_subplot(gs[0, 1])
ax.plot(x,
        L0, 
        color='C0')
h3 = ax.plot(x,(L1 + L0) / 2, 
             linestyle='dashed', 
             color='C1')
ax.set_ylim(0, 0.6)
ax.set_xlim(-4, 10)
ax.set_xlabel(r'$\xi$', 
              fontsize=16)
ax.set_ylabel(r'$L(\xi|\theta)$', 
              fontsize=16)
ax.set_xticks(np.arange(-4, 11, 4))
ax.locator_params(axis='y', 
                  nbins = 4)
ax.tick_params(axis='both', 
               which='major', 
               labelsize=14)
ax.annotate( r'$d(\bar{{\theta}}_{{\sf TX}})={0:.2f}$'.format(DKL(L0, (L1 + L0) / 2, dx)), 
            (0.5, 0.47), 
            xycoords=('axes fraction', 'data'), 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontsize=14)
fig.legend(handles=(h1[0],h2[0],h3[0]),
           labels=[r'$\theta_0=1$',r'$\theta_{\sf TX}=3$',r'$\bar{\theta}_{\sf TX}\equiv \{1,2\}$'], 
           fontsize = 16, 
           handlelength=1, 
           ncol=3, 
           loc='center', 
           bbox_to_anchor=(0.5, 0.11));


ax = fig.add_subplot(gs[0, 3])
ax.plot([MU_ob3[i][ag][2] for i in range(21)], 
        linewidth=2, 
        color='C2')
ax.tick_params(axis='both', 
               which='major', 
               labelsize=14)
ax.set_xlim([0, 20])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel(r'$i$', 
              fontsize=16)
ax.set_ylabel(r'$\mu_{1,i}(\theta_{\sf TX})$', 
              fontsize=16)
ax.set_title(r'Belief evolution', 
             fontsize=16)


fig.subplots_adjust(bottom=0.47)
fig.savefig(FIG_PATH + 'fig4c.pdf', 
            bbox_inches='tight')

# Plot likelihood comparison and belief evolution for agent 1, tx=2
fig=plt.figure(figsize=(9,2))
gs = gridspec.GridSpec(1, 4,
                       width_ratios = [1, 1, 0.01, 1], 
                       wspace=.6)
ax = fig.add_subplot(gs[0, 0])
h1=ax.plot(x,
           L0, 
           color='C0')
h2=ax.plot(x,
           L1, 
           color='C4')
ax.set_ylim(0, 0.6)
ax.set_xlim(-4, 10)
ax.set_xlabel(r'$\xi$', 
              fontsize=16)
ax.set_ylabel(r'$L(\xi|\theta)$', 
              fontsize=16)
ax.set_xticks(np.arange(-4,11,4))
ax.locator_params(axis='y', 
                  nbins = 4)
ax.tick_params(axis='both', 
               which='major', 
               labelsize=14)
ax.annotate(r'$d(\theta_{{\sf TX}})={0:.2f}$'.format(DKL(L0, L1, dx)), 
            (0.5, 0.47), 
            xycoords=('axes fraction', 'data'), 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontsize=14)


ax = fig.add_subplot(gs[0, 1])
ax.plot(x,
        L0, 
        color='C0')
h3 = ax.plot(x, (L2 + L0) / 2, 
             linestyle='dashed',
             color='C3')
ax.set_ylim(0,.6)
ax.set_xlim(-4, 10)
ax.set_xlabel(r'$\xi$', fontsize=16)
ax.set_ylabel(r'$L(\xi|\theta)$', fontsize=16)
ax.set_xticks(np.arange(-4,11,4))
ax.locator_params(axis='y', nbins = 4)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.annotate( r'$d(\bar{{\theta}}_{{\sf TX}})={0:.2f}$'.format(DKL(L0, (L2+L0)/2, dx)), 
            (0.5, 0.47), 
            xycoords=('axes fraction', 'data'), 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontsize=14)
fig.legend(handles=(h1[0],h2[0],h3[0]),
           labels=[r'$\theta_0=1$',r'$\theta_{\sf TX}=2$',r'$\bar{\theta}_{\sf TX}\equiv \{1,3\}$'], 
           fontsize = 16, 
           handlelength=1, 
           ncol=3, 
           loc='center', 
           bbox_to_anchor=(0.5, 0.11));


ax = fig.add_subplot(gs[0, 3])
ax.plot([MU_ob2[i][ag][1] for i in range(N_ITER)], 
        linewidth=2, 
        color='C4')
ax.tick_params(axis='both', which='major', 
               labelsize=14)
ax.set_xlabel(r'$i$', 
              fontsize=16)
ax.set_xlim([0, N_ITER])
ax.set_ylim([-0.1, 1.1])
ax.set_ylabel(r'$\mu_{1,i}(\theta_{\sf TX})$', 
              fontsize=16)
ax.set_title(r'Belief evolution', 
             fontsize=16)


fig.subplots_adjust(bottom=0.47)
fig.savefig(FIG_PATH + 'fig4b.pdf', 
            bbox_inches='tight')

