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

# Figure and Data paths
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)
    
# Setup
N = 10
M = 10
N_ITER = 100

# Identifiability matrix
id_matrix = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 0, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 0, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 0, 0, 5, 6, 7, 8, 9],
    [0, 0, 0, 0, 0, 0, 6, 7, 8, 9],
    [0, 0, 0, 0, 0, 0, 0, 7, 8, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     ])

# Seed
np.random.seed(0)
random.seed(0)

# Generate graph
G = np.random.choice([0.0, 1.0],
                     size=(N, N), 
                     p=[0.5, 0.5])
G = G + np.eye(N)
G = (G > 0) * 1.0

# Create combination matrix using averaging rule (self-awareness = 0.7)
lamb = 0.7
A = np.zeros((N, N))
for i in range(N):
    A[G[i] > 0, i]=(1 - lamb)/(np.sum(G[i]) - 1)
    A[i, i] = lamb
A_dec = decimal_array(A)

# Create combination matrix using averaging rule (self-awareness = 0.7)
lamb2 = 0.95
A2 = np.zeros((N, N))
for i in range(N):
    A2[G[i] > 0, i]=(1 - lamb2)/(np.sum(G[i]) - 1)
    A2[i, i] = lamb2
A2_dec = decimal_array(A2)


# Check that A^i converges
print('Matrix A is convergent?', np.all(np.isclose(np.linalg.matrix_power(A, 1000), np.linalg.matrix_power(A, 1001))))

print('Matrix A2 is convergent?', np.all(np.isclose(np.linalg.matrix_power(A2, 1000), np.linalg.matrix_power(A2, 1001))))

# Plot graph
np.random.seed(0)
Gr = nx.from_numpy_array(A)
pos = nx.spring_layout(Gr)

f, ax = plt.subplots(1, 1, figsize=(5, 2.5))
plt.axis('off')
plt.ylim([-1.15, 1.15])
nx.draw_networkx_nodes(Gr, 
                       pos=pos, 
                       node_color='#c0df99', 
                       nodelist=range(N), 
                       node_size=350, 
                       edgecolors='k', 
                       linewidths=0.5)

nx.draw_networkx_labels(Gr, 
                        pos=pos, 
                        labels={i: i+1 for i in range(N)}, 
                        font_size=16, 
                        font_color='black', 
                        alpha=1)
nx.draw_networkx_edges(Gr, 
                       pos=pos, 
                       node_size=350, 
                       alpha=1, 
                       arrowsize=6, 
                       width=1)
plt.tight_layout()
plt.gca().xaxis.set_major_locator(plt.NullLocator()) # Trick to remove white space in bottom
plt.savefig(FIG_PATH + 'fig6.pdf', 
            bbox_inches = 'tight')

# Hypotheses
theta = np.array([0.5 * i for i in range(10)])
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

csi = []
for l in range(N):
    csi.append(thetadec[0] + np.sqrt(vardec) * decimal_array(np.random.randn(N_ITER)))
csi = np.array(csi)

# Case 1: lambda = 0.7

# tx=0
MU_ob_0 = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=0, partial = True, self_aware=False, id_matrix=id_matrix)

MU_ob_0sa = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=0, partial = True, self_aware=True, id_matrix=id_matrix)

# tx=1
MU_ob_1 = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=2, partial=True, self_aware=False, id_matrix=id_matrix)

MU_ob_1sa = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=2, partial=True, self_aware=True, id_matrix=id_matrix)

# tx=2
MU_ob_2 = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=5, partial=True, self_aware=False, id_matrix=id_matrix)

MU_ob_2sa = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, tx=5, partial=True, self_aware=True, id_matrix=id_matrix)

# SL
MU_sl = partial_info(mu_0, csi, thetadec, vardec, A_dec, N_ITER, partial=False, self_aware=False, id_matrix=id_matrix)


# Case 2: lambda = 0.95

# tx=0
MU_ob_02 = partial_info(mu_0, csi, thetadec, vardec, A2_dec, N_ITER, tx=0, partial=True, self_aware=False, id_matrix=id_matrix)

MU_ob_0sa2 = partial_info(mu_0, csi, thetadec, vardec, A2_dec, N_ITER, tx=0, partial=True, self_aware=True, id_matrix=id_matrix)

# tx=1
MU_ob_12 = partial_info(mu_0, csi, thetadec, vardec, A2_dec, N_ITER, tx=2, partial=True, self_aware=False, id_matrix=id_matrix)

MU_ob_1sa2 = partial_info(mu_0, csi, thetadec, vardec, A2_dec, N_ITER, tx=2, partial=True, self_aware=True, id_matrix=id_matrix)

# tx=2
MU_ob_22 = partial_info(mu_0, csi, thetadec, vardec, A2_dec, N_ITER, tx=5, partial=True, self_aware=False, id_matrix=id_matrix)

MU_ob_2sa2 = partial_info(mu_0, csi, thetadec, vardec, A2_dec, N_ITER, tx=5, partial=True, self_aware=True, id_matrix=id_matrix)

# SL
MU_sl2 = partial_info(mu_0, csi, thetadec, vardec, A2_dec, N_ITER, partial=False, self_aware=False, id_matrix=id_matrix)


# Plot belief evolution for agent 5, lambda = 0.7
ag = 4


fig, ax = plt.subplots(1, 3, 
                       figsize=(6, 3.5), 
                       gridspec_kw={'bottom': 0.4, 'top': 0.8})

ax[0].plot([MU_ob_0[i][ag][0] for i in range(N_ITER)], 
           color='C0', 
           linewidth=2)
ax[0].plot([MU_ob_0sa[i][ag][0] for i in range(N_ITER)], 
           color='C0', 
           linestyle='dashed', 
           linewidth=2)
ax[0].plot([MU_sl[i][ag][0] for i in range(N_ITER)], 
           color='k', 
           linestyle=':', 
           linewidth=2)
ax[0].tick_params(axis='both', 
                  which='major', 
                  labelsize=14)
ax[0].set_ylim([-0.1, 1.1])
ax[0].set_xlim([0, 100])
ax[0].set_xlabel(r'$i$', 
                 fontsize=16)
ax[0].set_title(r'$\theta_{\sf TX}=1$', 
                fontsize=16)
ax[0].set_ylabel(r'$\mu_{%d,i}(\theta_{\sf TX})$' % (ag + 1), 
                 fontsize=16)


ax[1].plot([MU_ob_1[i][ag][2] for i in range(N_ITER)], 
           color='C1', 
           linewidth=2)
ax[1].plot([MU_ob_1sa[i][ag][2] for i in range(N_ITER)], 
           color='C1', 
           linestyle='dashed', 
           linewidth=2)
ax[1].plot([MU_sl[i][ag][2] for i in range(N_ITER)], 
           color='k', 
           linestyle=':', 
           linewidth=2)
ax[1].tick_params(axis='both', 
                  which='major', 
                  labelsize=14)
ax[1].set_yticklabels([])
ax[1].set_ylim([-0.1, 1.1])
ax[1].set_xlim([0, 100])
ax[1].set_xlabel(r'$i$', 
                 fontsize=16)
ax[1].set_title(r'$\theta_{\sf TX}=3$', 
                fontsize=16)

ax[2].plot([MU_ob_2[i][ag][5] for i in range(N_ITER)], 
           linewidth=2, 
           color='C2')
ax[2].plot([MU_ob_2sa[i][ag][5] for i in range(N_ITER)], 
           color='C2', 
           linestyle='dashed', 
           linewidth=2)
ax[2].plot([MU_sl[i][ag][5] for i in range(N_ITER)], 
           color='k', 
           linestyle=':', 
           linewidth=2)
ax[2].tick_params(axis='both', 
                  which='major', 
                  labelsize=14)
ax[2].set_ylim([-0.1, 1.1])
ax[2].set_xlim([0, 100])
ax[2].set_yticklabels([])
ax[2].set_xlabel(r'$i$', 
                 fontsize=16)
ax[2].set_title(r'$\theta_{\sf TX}=6$', 
                fontsize=16)
fig.text(0.15, 0.08, 
         'Solid Line: w/o SA \qquad Dashed Line: w/ SA \n Dotted Line: Social learning (full information)', 
         ha='left', 
         fontsize=16,
         bbox=dict(facecolor='w'))

fig.savefig(FIG_PATH + 'fig7a.pdf', bbox_inches='tight')

# Plot belief evolution for agent 5, lambda = 0.95
ag = 4

fig, ax = plt.subplots(1, 3, 
                       figsize=(6, 3.5), 
                       gridspec_kw={'bottom': 0.4, 'top': 0.8})
ax[0].plot([MU_ob_02[i][ag][0] for i in range(N_ITER)], 
           color='C0', 
           linewidth=2)
ax[0].plot([MU_ob_0sa2[i][ag][0] for i in range(N_ITER)], 
           color='C0', 
           linestyle='dashed', 
           linewidth=2)
ax[0].plot([MU_sl2[i][ag][0] for i in range(N_ITER)], 
           color='k', 
           linestyle=':', 
           linewidth=2)
ax[0].tick_params(axis='both', 
                  which='major', 
                  labelsize=14)
ax[0].set_ylim([-0.1, 1.1])
ax[0].set_xlim([0, 100])
ax[0].set_xlabel(r'$i$', 
                 fontsize=16)
ax[0].set_title(r'$\theta_{\sf TX}=1$', 
                fontsize=16)
ax[0].set_ylabel(r'$\mu_{%d,i}(\theta_{\sf TX})$' % (ag+1), 
                 fontsize=16)


ax[1].plot([MU_ob_12[i][ag][2] for i in range(N_ITER)], 
           color='C1', 
           linewidth=2)
ax[1].plot([MU_ob_1sa2[i][ag][2] for i in range(N_ITER)], 
           color='C1', 
           linestyle='dashed', 
           linewidth=2)
ax[1].plot([MU_sl2[i][ag][2] for i in range(N_ITER)], 
           color='k', 
           linestyle=':', 
           linewidth=2)
ax[1].tick_params(axis='both', 
                  which='major', 
                  labelsize=14)
ax[1].set_yticklabels([])
ax[1].set_ylim([-0.1, 1.1])
ax[1].set_xlim([0, 100])
ax[1].set_xlabel(r'$i$', 
                 fontsize=16)
ax[1].set_title(r'$\theta_{\sf TX}=3$', 
                fontsize=16)


ax[2].plot([MU_ob_22[i][ag][5] for i in range(N_ITER)], 
           linewidth=2, 
           color='C2')
ax[2].plot([MU_ob_2sa2[i][ag][5] for i in range(N_ITER)], 
           color='C2', 
           linestyle='dashed', 
           linewidth=2)
ax[2].plot([MU_sl2[i][ag][5] for i in range(N_ITER)], 
           color='k', 
           linestyle=':', 
           linewidth=2)
ax[2].tick_params(axis='both', 
                  which='major', 
                  labelsize=14)
ax[2].set_ylim([-0.1, 1.1])
ax[2].set_xlim([0, 100])
ax[2].set_yticklabels([])
ax[2].set_xlabel(r'$i$', 
                 fontsize=16)
ax[2].set_title(r'$\theta_{\sf TX}=6$', 
                fontsize=16)
fig.text(0.15, 0.08, 
         'Solid Line: w/o SA \qquad Dashed Line: w/ SA \n Dotted Line: Social learning (full information)', 
         ha='left', 
         fontsize=16,
         bbox=dict(facecolor='w'))

fig.savefig(FIG_PATH + 'fig7b.pdf', bbox_inches='tight')
