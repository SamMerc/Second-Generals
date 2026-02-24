#############################
#### Importing libraries ####
#############################

from turtle import color

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing temperature values
raw_T_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#Path to store plots
plot_save_path = base_dir+'Plots/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
raw_inputs = raw_T_data[:, :4]
raw_outputs_T = raw_T_data[:, 5:]
raw_outputs_P = raw_P_data[:, 5:]
#Convert raw outputs to log10 scale so we don't have to deal with it later
raw_outputs_P = np.log10(raw_outputs_P/1000)

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs_T.shape[1] #Number of outputs


# Shuffle data
rp = np.random.permutation(N) #random permutation of the indices
# Apply random permutation to shuffle the data
raw_inputs = raw_inputs[rp, :]
raw_outputs_T = raw_outputs_T[rp, :]
raw_outputs_P = raw_outputs_P[rp, :]

## HYPER-PARAMETERS ##
#Definine partitiion for splitting NN dataset
data_partition = [0.7, 0.1, 0.2]

# Variable to show plots or not 
show_plot = True

#Number of nearest neighbors to choose
N_neigbors = np.linspace(5, 1000, 10, dtype=int).tolist()

#Distance metric to use
distance_metric = 'euclidean' #options: 'euclidean', 'mahalanobis', 'logged_euclidean', 'logged_mahalanobis'

#Convert raw inputs for H2 and CO2 pressures to log10 scale so don't have to deal with it later
if 'logged' in distance_metric:
    raw_inputs[:, 0] = np.log10(raw_inputs[:, 0]) #H2
    raw_inputs[:, 1] = np.log10(raw_inputs[:, 1]) #CO2


INPUT_LABELS = [
    r'H$_2$ Pressure (bar)',
    r'CO$_2$ Pressure (bar)',
    r'LoD (days)',
    r'Obliquity (deg)',
]

plot_1 = False
plot_2 = True

refinement_iterations = 20

############################################################
#### Plot curves, covariance matrices and eigenspectrum ####
############################################################

if plot_1:
    # --- P profiles ---
    ## Base 
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, raw_output_P in enumerate(raw_outputs_P):
        ax.plot(raw_output_P, color=cm.get_cmap('coolwarm')(i / (len(raw_outputs_P) - 1)))
    ax.set_xlabel('Index')
    ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
    ax.invert_yaxis()
    # Add colorbar to show index scale
    sm = cm.ScalarMappable(cmap='coolwarm', norm=mcolors.Normalize(vmin=0, vmax=len(raw_outputs_P) - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Profile Index')
    plt.savefig(plot_save_path + 'ALL_P_profiles.pdf')
    plt.show()

    ## Correlations with input parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for j, ax in enumerate(axes.flatten()):
        
        # Normalize based on actual input values
        norm = mcolors.Normalize(vmin=min(raw_inputs[:, j]), vmax=max(raw_inputs[:, j]))
        cmap = cm.get_cmap('coolwarm')

        for i, raw_output_P in enumerate(raw_outputs_P):
            ax.plot(raw_output_P, color=cmap(norm(raw_inputs[i, j])))
        
        if j > 1:ax.set_xlabel('Index')
        if j==0 or j== 2:ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
        ax.invert_yaxis()

        sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=INPUT_LABELS[j])
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(plot_save_path + 'CORRELATED_P_profiles.pdf')
    plt.show()

    # --- T profiles ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, raw_output_T in enumerate(raw_outputs_T):
        ax.plot(raw_output_T)
    ax.set_xlabel('Index')
    ax.set_ylabel('Temperature (K)')
    plt.savefig(plot_save_path + 'ALL_T_profiles.pdf')
    plt.show()

    ## Correlations with input parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for j, ax in enumerate(axes.flatten()):
        
        # Normalize based on actual input values
        norm = mcolors.Normalize(vmin=min(raw_inputs[:, j]), vmax=max(raw_inputs[:, j]))
        cmap = cm.get_cmap('coolwarm')

        for i, raw_output_T in enumerate(raw_outputs_T):
            ax.plot(raw_output_T, color=cmap(norm(raw_inputs[i, j])))
        
        if j > 1:ax.set_xlabel('Index')
        if j==0 or j== 2:ax.set_ylabel(r'Temperature (K)')

        sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=INPUT_LABELS[j])
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(plot_save_path + 'CORRELATED_T_profiles.pdf')
    plt.show()

    # --- P Covariance plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.cov(raw_outputs_P.T), color='blue')
    ax.set_xlabel('Index')
    ax.set_ylabel('Covariance')
    plt.savefig(plot_save_path + 'COV_P_profiles.pdf')
    plt.show()

    # --- T Covariance heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = sns.heatmap(
        np.cov(raw_outputs_T.T),
        cmap='coolwarm',
        ax=ax,
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Covariance', fontsize=11)
    ax.set_xlabel('Index')
    ax.set_ylabel('Index')
    plt.savefig(plot_save_path + 'COV_T_profiles.pdf')
    plt.show()

    # --- SVD Decomposition for T ---
    _, S_T, _ = np.linalg.svd(raw_outputs_T, full_matrices=False)

    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(S_T, color='blue')
    axes.set_xlabel('Component Index')
    axes.set_ylabel('Singular Value')
    axes.set_yscale('log')

    plt.savefig(plot_save_path + 'SVD_T.pdf')
    plt.show()

    # --- SVD Decomposition for P ---
    _, S_P, _ = np.linalg.svd(raw_outputs_P, full_matrices=False)

    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(S_P, color='blue')
    axes.set_xlabel('Component Index')
    axes.set_ylabel('Singular Value')
    axes.set_yscale('log')

    plt.savefig(plot_save_path + 'SVD_P.pdf')
    plt.show()


###############################################
#### Ensemble Conditional Gaussian Process ####
###############################################
# ── JAX KNN ───────────────────────────────────────────────────────────────────
@partial(jit, static_argnames=('k',))
def _mahal_knn_single(X_train, xq, VI, k):
    """Single query point. X_train: (D, N), xq: (D,), returns (k,)"""
    diff = X_train - xq[:, None]                     # (D, N)
    dists_sq = jnp.sum(diff * (VI @ diff), axis=0)   # (N,)
    return jnp.argsort(dists_sq)[:k]

@partial(jit, static_argnames=('k',))
def _mahal_knn_batch(X_train, X_queries, VI, k):
    """Batch of query points. X_queries: (D, Q), returns (Q, k)"""
    def single(xq):
        diff = X_train - xq[:, None]
        dists_sq = jnp.sum(diff * (VI @ diff), axis=0)
        return jnp.argsort(dists_sq)[:k]
    return vmap(single)(X_queries.T)

# ── JAX CGP step ──────────────────────────────────────────────────────────────
@jit
def _cgp_step(Xens_NN, Yens_NN, Xq):
    """Full covariance + forward/backward step for a fixed neighbourhood."""
    Xm = Xens_NN.mean(axis=1, keepdims=True)
    Ym = Yens_NN.mean(axis=1, keepdims=True)
    dX = Xens_NN - Xm
    dY = Yens_NN - Ym

    Cxx = dX @ dX.T
    Cyx = dY @ dX.T
    Cyy = dY @ dY.T
    Cxy = dX @ dY.T

    # eigvalsh is faster than eigvals for symmetric matrices
    rdgx = jnp.maximum(1e-10, jnp.min(jnp.linalg.eigvalsh(Cxx)))
    rdgy = jnp.maximum(1e-10, jnp.min(jnp.linalg.eigvalsh(Cyy)))

    Mf = Cyx @ jnp.linalg.pinv(Cxx + rdgx * jnp.eye(Cxx.shape[0]))
    Mb = Cxy @ jnp.linalg.pinv(Cyy + rdgy * jnp.eye(Cyy.shape[0]))

    YhSel = Yens_NN + Mf @ (Xq - Xens_NN)
    XhSel = Xens_NN + Mb @ (Ym - YhSel)

    Yh     = Ym + Mf @ (Xq - Xm)
    cov_Yh = Cyy - Mf @ Cxy

    return Mf, YhSel, XhSel, Xm, Ym, Yh, cov_Yh

# ── Main function ─────────────────────────────────────────────────────────────
def ens_CGP(Xens, Yens, Xq, N_neighbor, refinement_iterations):
    """
    Parameters:
    Xens: array of input features which compose the ensemble. shape:(D, N) 
    Yens: array of input labels which compose the ensemble. shape:(M, N) 
    Xq: query point for which we want to compute a prediction. shape:(D, 1)
    N_neighbor: int, number of neighbors to use in CGP
    refinement_iterations: int, number of times to refine the neighborhood
    """
    # Convert to JAX arrays once — cheap after first call
    Xens_j = jnp.array(Xens)
    Yens_j = jnp.array(Yens)
    Xq_j   = jnp.array(Xq)
    VI_j   = jnp.linalg.inv(jnp.cov(Xens_j))

    # Initial KNN
    idxs = np.array(_mahal_knn_single(Xens_j, Xq_j.ravel(), VI_j, N_neighbor))

    # Iterative refinement
    for _ in range(refinement_iterations):
        Mf, YhSel, XhSel, Xm, Ym, Yh, cov_Yh = _cgp_step(
            Xens_j[:, idxs], Yens_j[:, idxs], Xq_j
        )

        # Dynamic parts must stay in numpy due to variable shape from np.unique
        idxs2 = np.array(_mahal_knn_batch(Xens_j, XhSel, VI_j, 1)).flatten()
        idxs  = np.unique(idxs2)
        lx    = len(idxs)

        if lx < N_neighbor:
            idxs3 = np.array(_mahal_knn_single(Xens_j, Xq_j.ravel(), VI_j, N_neighbor - lx))
            idxs  = np.unique(np.concatenate([idxs, idxs3]))

    err_Yh = jnp.sqrt(jnp.maximum(0.0, jnp.diag(cov_Yh)))
    return np.array(Yh.flatten()), np.array(err_Yh)
        
if plot_2:

    #Track the bias and variance of the T and P predictions separately as a function of N_neighbors
    bias_T = np.zeros(len(N_neigbors), dtype=float)
    bias_P = np.zeros(len(N_neigbors), dtype=float)
    var_T = np.zeros(len(N_neigbors), dtype=float)
    var_P = np.zeros(len(N_neigbors), dtype=float)
    MSE_T = np.zeros(len(N_neigbors), dtype=float)
    MSE_P = np.zeros(len(N_neigbors), dtype=float)

    for NNidx, N_neighbor in enumerate(tqdm(N_neigbors)):

        guess_T = np.zeros(raw_outputs_T.shape, dtype=float)
        guess_Terr = np.zeros(raw_outputs_T.shape, dtype=float)
        guess_P = np.zeros(raw_outputs_P.shape, dtype=float)
        guess_Perr = np.zeros(raw_outputs_P.shape, dtype=float)

        for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(tqdm(raw_inputs), raw_outputs_T, raw_outputs_P)):

            # Define the training data for CGP (all data points except the query point)
            XTr = np.delete(
                raw_inputs.T, #shape: (4, N)
                query_idx,
                axis=1
                )
            YTr = np.delete(
                            np.hstack([raw_outputs_T, raw_outputs_P]).T,   # shape: (M, N)
                            query_idx,
                            axis=1
                            )

            Yh, err_Yh = ens_CGP(XTr, YTr, query_input.reshape(-1, 1), N_neighbor, refinement_iterations)
            guess_T[query_idx, :] = Yh[:O]
            guess_P[query_idx, :] = Yh[O:]
            guess_Terr[query_idx, :] = err_Yh[:O]
            guess_Perr[query_idx, :] = err_Yh[O:]

            #Diagnostic plot
            # fig, axs = plt.subplot_mosaic([['res_pressure', '.'],
            #                                 ['results', 'res_temperature']],
            #                         figsize=(8, 6),
            #                         width_ratios=(3, 1), height_ratios=(1, 3),
            #                         layout='constrained')        
            # axs['results'].plot(query_output_T, query_output_P, '.', linestyle='-', color='blue', linewidth=2, label='Truth')
            # axs['results'].errorbar(guess_T[query_idx, :], guess_P[query_idx, :], xerr=guess_Terr[query_idx, :], yerr=guess_Perr[query_idx, :], fmt='o', linestyle='-', color='green', linewidth=2, label='Prediction')
            # axs['results'].invert_yaxis()
            # axs['results'].set_ylabel(r'log$_{10}$ Pressure (bar)')
            # axs['results'].set_xlabel('Temperature (K)')
            # axs['results'].legend()
            # axs['results'].grid()

            # axs['res_temperature'].errorbar(guess_T[query_idx, :] - query_output_T, query_output_P, xerr=guess_Terr[query_idx, :], fmt='.', linestyle='-', color='green', linewidth=2)
            # axs['res_temperature'].set_xlabel('Residuals (K)')
            # axs['res_temperature'].invert_yaxis()
            # axs['res_temperature'].grid()
            # axs['res_temperature'].yaxis.tick_right()
            # axs['res_temperature'].yaxis.set_label_position("right")
            # axs['res_temperature'].sharey(axs['results'])

            # axs['res_pressure'].errorbar(query_output_T, guess_P[query_idx, :] - query_output_P, yerr=guess_Perr[query_idx, :], fmt='.', linestyle='-', color='green', linewidth=2)
            # axs['res_pressure'].set_ylabel('Residuals (bar)')
            # axs['res_pressure'].invert_yaxis()
            # axs['res_pressure'].grid()
            # axs['res_pressure'].xaxis.tick_top()
            # axs['res_pressure'].xaxis.set_label_position("top")
            # axs['res_pressure'].sharex(axs['results'])

            # plt.suptitle(rf'H$_2$ : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg')
            # plt.close()

        #Compute bias and variance for T and P predictions
        bias_T[NNidx] = np.mean(guess_T - raw_outputs_T)
        bias_P[NNidx] = np.mean(guess_P - raw_outputs_P)

        var_T[NNidx] = np.mean(guess_Terr**2) #Use the predicted errorbars as a proxy for variance
        var_P[NNidx] = np.mean(guess_Perr**2)

        MSE_T[NNidx] = bias_T[NNidx]**2 + var_T[NNidx]
        MSE_P[NNidx] = bias_P[NNidx]**2 + var_P[NNidx]

    #Plot bias and variance as a function of N_neighbors
    fig, ax = plt.subplots(3, 2, figsize=(8, 6))
    ax[0,0].plot(N_neigbors, bias_T, label='Bias T', color='blue')
    ax[0,1].plot(N_neigbors, bias_P, label='Bias P', color='orange')
    ax[1,0].plot(N_neigbors, var_T, label='Variance T', color='blue', linestyle='--')
    ax[1,1].plot(N_neigbors, var_P, label='Variance P', color='orange', linestyle='--')
    ax[2,0].plot(N_neigbors, MSE_T, label='MSE T', color='blue')
    ax[2,1].plot(N_neigbors, MSE_P, label='MSE P', color='orange')
    ax[0,0].set_xlabel('Number of Neighbors')
    ax[0,1].set_xlabel('Number of Neighbors')
    ax[1,0].set_ylabel('Bias / Variance')
    ax[1,1].set_ylabel('Bias / Variance')
    ax[0,0].set_title('Bias T')
    ax[0,1].set_title('Bias P')
    ax[1,0].set_title('Variance T')
    ax[1,1].set_title('Variance P')
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    plt.savefig(plot_save_path + 'Bias_Variance.pdf')
    plt.show()