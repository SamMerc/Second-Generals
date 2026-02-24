#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
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
#File containing surface temperature map
raw_ST_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')
#Path to store plots
plot_save_path = base_dir+'Plots/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
raw_inputs = raw_ST_data[:, :4] #has shape 46 x 72 = 3,312
raw_outputs = raw_ST_data[:, 5:]

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs.shape[1] #Number of outputs

## HYPER-PARAMETERS ##
#Definine partitiion for splitting NN dataset
data_partition = [0.7, 0.1, 0.2]

# Variable to show plots or not 
show_plot = True

#Number of nearest neighbors to choose
N_neigbors = np.linspace(5, 200, 5, dtype=int).tolist()

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
    # --- ST2D maps ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, raw_output in enumerate(raw_outputs):
        ax.plot(raw_output)
    ax.set_xlabel('Index')
    ax.set_ylabel('Temperature (K)')
    plt.savefig(plot_save_path + 'ALL_ST2D_maps.pdf')
    plt.show()


    ## Correlations with input parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for j, ax in enumerate(axes.flatten()):
        
        # Normalize based on actual input values
        norm = mcolors.Normalize(vmin=min(raw_inputs[:, j]), vmax=max(raw_inputs[:, j]))
        cmap = cm.get_cmap('coolwarm')

        for i, raw_output in enumerate(raw_outputs):
            ax.plot(raw_output, color=cmap(norm(raw_inputs[i, j])))
        
        if j > 1:ax.set_xlabel('Index')
        if j==0 or j== 2:ax.set_ylabel(r'Temperature (K)')

        sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=INPUT_LABELS[j])
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(plot_save_path + 'CORRELATED_ST2D_profiles.pdf')
    plt.show()

    # --- ST2D Covariance heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = sns.heatmap(
        np.cov(raw_outputs.T),
        cmap='coolwarm',
        ax=ax,
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Covariance', fontsize=11)
    ax.set_xlabel('Index')
    ax.set_ylabel('Index')
    plt.savefig(plot_save_path + 'COV_ST2D_profiles.pdf')
    plt.show()

    # --- SVD Decomposition for ST2D ---
    _, S_ST2D, _ = np.linalg.svd(raw_outputs, full_matrices=False)

    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(S_ST2D, color='blue')
    axes.set_xlabel('Component Index')
    axes.set_ylabel('Singular Value')
    axes.set_yscale('log')

    plt.savefig(plot_save_path + 'SVD_ST2D.pdf')
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

    #Track the bias and variance of the ST2D predictions as a function of N_neighbors
    bias = np.zeros(len(N_neigbors), dtype=float)
    var = np.zeros(len(N_neigbors), dtype=float)
    MSE = np.zeros(len(N_neigbors), dtype=float)

    for NNidx, N_neighbor in enumerate(tqdm(N_neigbors)):

        guess_ST2D = np.zeros(raw_outputs.shape, dtype=float)
        guess_ST2Derr = np.zeros(raw_outputs.shape, dtype=float)

        for query_idx, (query_input, query_output) in enumerate(zip(tqdm(raw_inputs), raw_outputs)):

            # Define the training data for CGP (all data points except the query point)
            XTr = np.delete(
                raw_inputs.T, #shape: (4, N)
                query_idx,
                axis=1
                )
            YTr = np.delete(
                            raw_outputs.T,   # shape: (M, N)
                            query_idx,
                            axis=1
                            )

            Yh, err_Yh = ens_CGP(XTr, YTr, query_input.reshape(-1, 1), N_neighbor, refinement_iterations)
            guess_ST2D[query_idx, :] = Yh
            guess_ST2Derr[query_idx, :] = err_Yh

        #Diagnostic plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout='constrained')        
            
            # Compute global vmin/vmax across all datasets
            # vmin = np.min(NN_test_output)
            # vmax = np.max(NN_test_output)
            
            # Plot heatmaps
            ax1.set_title('Data')
            hm1 = sns.heatmap(query_output.reshape((46,72)), ax=ax1)#, cbar=False, vmin=vmin, vmax=vmax)
            cbar = hm1.collections[0].colorbar
            cbar.set_label('Temperature (K)')
            ax2.set_title('GP Model')
            hm2 = sns.heatmap(guess_ST2D[query_idx, :].reshape((46,72)), ax=ax2)#, cbar=False, vmin=vmin, vmax=vmax)
            cbar = hm2.collections[0].colorbar
            cbar.set_label('Temperature (K)')
            ax3.set_title('NN Model')
            hm3 = sns.heatmap(query_output.reshape((46,72)) - guess_ST2D[query_idx, :].reshape((46,72)), ax=ax3)#, cbar=False, vmin=vmin, vmax=vmax)
            cbar = hm3.collections[0].colorbar
            cbar.set_label('Temperature (K)')

            # Shared colorbar (use the last heatmap's mappable)
            # cbar = fig.colorbar(hm3.get_children()[0], ax=[ax1, ax2, ax3], location='right')
            # cbar.set_label("Temperature")
            # Fix longitude ticks
            ax3.set_xticks(np.linspace(0, 72, 5))
            ax3.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
            ax3.set_xlabel('Longitude (degrees)')
            # Fix latitude ticks
            for ax in [ax1, ax2, ax3]:
                ax.set_yticks(np.linspace(0, 46, 5))
                ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
                ax.set_ylabel('Latitude (degrees)')
            plt.suptitle(rf'H$_2$ : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg')
            plt.show()

    #     #Compute bias and variance for T and P predictions
    #     bias_T[NNidx] = np.mean(guess_T - raw_outputs_T)
    #     bias_P[NNidx] = np.mean(guess_P - raw_outputs_P)

    #     var_T[NNidx] = np.mean(guess_Terr**2) #Use the predicted errorbars as a proxy for variance
    #     var_P[NNidx] = np.mean(guess_Perr**2)

    #     MSE_T[NNidx] = bias_T[NNidx]**2 + var_T[NNidx]
    #     MSE_P[NNidx] = bias_P[NNidx]**2 + var_P[NNidx]

    # #Plot bias and variance as a function of N_neighbors
    # fig, ax = plt.subplots(3, 2, figsize=(8, 6))
    # ax[0,0].plot(N_neigbors, bias_T, label='Bias T', color='blue')
    # ax[0,1].plot(N_neigbors, bias_P, label='Bias P', color='orange')
    # ax[1,0].plot(N_neigbors, var_T, label='Variance T', color='blue', linestyle='--')
    # ax[1,1].plot(N_neigbors, var_P, label='Variance P', color='orange', linestyle='--')
    # ax[2,0].plot(N_neigbors, MSE_T, label='MSE T', color='blue')
    # ax[2,1].plot(N_neigbors, MSE_P, label='MSE P', color='orange')
    # ax[0,0].set_xlabel('Number of Neighbors')
    # ax[0,1].set_xlabel('Number of Neighbors')
    # ax[1,0].set_ylabel('Bias / Variance')
    # ax[1,1].set_ylabel('Bias / Variance')
    # ax[0,0].set_title('Bias T')
    # ax[0,1].set_title('Bias P')
    # ax[1,0].set_title('Variance T')
    # ax[1,1].set_title('Variance P')
    # ax[0,0].legend()
    # ax[0,1].legend()
    # ax[1,0].legend()
    # ax[1,1].legend()
    # plt.savefig(plot_save_path + 'Bias_Variance.pdf')
    # plt.show()