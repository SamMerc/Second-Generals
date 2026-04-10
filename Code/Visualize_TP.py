#############################
#### Importing libraries ####
#############################

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
raw_T_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_T.csv', delimiter=',')
raw_T_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_P.csv', delimiter=',')
raw_P_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#Path to store plots
plot_save_path = base_dir+'Plots/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
# Extract the 4 physical inputs and append stellar temperature as 5th column
inputs_3000 = np.hstack([raw_T_data3000[:, :4], np.full((len(raw_T_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_T_data4500[:, :4], np.full((len(raw_T_data4500), 1), 4500.0)])

# Concatenate along the sample axis
raw_inputs    = np.vstack([inputs_3000,            inputs_4500           ])  # (N_3000+N_4500, 5)
raw_outputs_T = np.vstack([raw_T_data3000[:, 5:],  raw_T_data4500[:, 5:]])  # (N_3000+N_4500, O)
raw_outputs_P = np.vstack([raw_P_data3000[:, 5:],  raw_P_data4500[:, 5:]])  # (N_3000+N_4500, O)

#Convert raw outputs to log10 scale so we don't have to deal with it later
raw_outputs_P = np.log10(raw_outputs_P/1000)

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs_T.shape[1] #Number of outputs

# Shuffle data
np.random.seed(3)
rp = np.random.permutation(N) #random permutation of the indices
# Apply random permutation to shuffle the data
raw_inputs = raw_inputs[rp, :]
raw_outputs_T = raw_outputs_T[rp, :]
raw_outputs_P = raw_outputs_P[rp, :]

INPUT_LABELS = [
    r'H$_2$ Pressure (bar)',
    r'CO$_2$ Pressure (bar)',
    r'LoD (days)',
    r'Obliquity (deg)',
    r'T$_{eff}$ (K)',
]

plot_1 = True
plot_2 = False

############################################################
#### Plot curves, covariance matrices and eigenspectrum ####
############################################################

if plot_1:
    # --- P profiles ---
    ## Base 
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, raw_output_P in enumerate(raw_outputs_P):
        ax.plot(raw_output_P)
    ax.set_xlabel('Index')
    ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
    ax.invert_yaxis()
    plt.savefig(plot_save_path + 'ALL_P_profiles.pdf')
    plt.show()

    ## Correlations with input parameters
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for j, ax in zip(range(D), axes.flatten()):
        
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
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for j, ax in zip(range(D), axes.flatten()):
        
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
    var_explained_T = np.cumsum(S_T**2) / np.sum(S_T**2)

    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(S_T, color='blue')
    axes.set_xlabel('Component Index')
    axes.set_ylabel('Singular Value')
    axes.set_yscale('log')
    axestwin = axes.twinx()
    axestwin.set_yscale('log')
    axestwin.plot(var_explained_T, color='red')
    axestwin.set_ylabel('Cumulative variance explained')
    # Find number of components needed to explain threshold % of variance
    threshold = 0.999
    n_components_T = np.searchsorted(var_explained_T, threshold) + 1
    axestwin.axhline(threshold, color='red', linestyle='--', label=f'{threshold*100}% threshold')
    axestwin.axvline(n_components_T-1, color='green', linestyle='--', label=f'K={n_components_T}')
    plt.legend()
    plt.savefig(plot_save_path + 'SVD_T.pdf')
    plt.show()

    # --- SVD Decomposition for P ---
    _, S_P, _ = np.linalg.svd(raw_outputs_P, full_matrices=False)
    var_explained_P = np.cumsum(S_P**2) / np.sum(S_P**2)

    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(S_P, color='blue')
    axes.set_xlabel('Component Index')
    axes.set_ylabel('Singular Value')
    axes.set_yscale('log')
    axestwin = axes.twinx()
    axestwin.set_yscale('log')
    axestwin.plot(var_explained_P, color='red')
    axestwin.set_ylabel('Cumulative variance explained')
    # Find number of components needed to explain threshold % of variance
    threshold = 0.9999
    n_components_P = np.searchsorted(var_explained_P, threshold) + 1
    axestwin.axhline(threshold, color='red', linestyle='--', label=f'{threshold*100}% threshold')
    axestwin.axvline(n_components_P-1, color='green', linestyle='--', label=f'K={n_components_P}')
    plt.legend()
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
@partial(jit, static_argnames=('N_neighbor',))
def _cgp_step_fixed(Xens, Yens, idxs, Xq, VI, N_neighbor):
    """idxs is always shape (N_neighbor,) — no dynamic shapes."""
    Xens_NN = Xens[:, idxs]   # shape always (D, N_neighbor) ← fixed!
    Yens_NN = Yens[:, idxs]   # shape always (M, N_neighbor) ← fixed!

    Xm = Xens_NN.mean(axis=1, keepdims=True)
    Ym = Yens_NN.mean(axis=1, keepdims=True)
    dX = Xens_NN - Xm
    dY = Yens_NN - Ym

    Cxx = dX @ dX.T
    Cyx = dY @ dX.T
    Cyy = dY @ dY.T
    Cxy = dX @ dY.T

    rdgx = jnp.maximum(1e-10, jnp.min(jnp.linalg.eigvalsh(Cxx)))
    rdgy = jnp.maximum(1e-10, jnp.min(jnp.linalg.eigvalsh(Cyy)))

    Mf = Cyx @ jnp.linalg.pinv(Cxx + rdgx * jnp.eye(Cxx.shape[0]))
    Mb = Cxy @ jnp.linalg.pinv(Cyy + rdgy * jnp.eye(Cyy.shape[0]))

    YhSel = Yens_NN + Mf @ (Xq - Xens_NN)
    XhSel = Xens_NN + Mb @ (Ym - YhSel)

    # Fixed-size unique: always returns exactly N_neighbor indices
    idxs2 = _mahal_knn_batch(Xens, XhSel, VI, 1).flatten()   # (N_neighbor,)
    idxs_new = jnp.unique(idxs2, size=N_neighbor,
                          fill_value=-1)                       # (N_neighbor,)

    # Top-up: always pull N_neighbor candidates from Xq, use where idxs_new has fill
    idxs_topup = _mahal_knn_single(Xens, Xq.ravel(), VI, N_neighbor)
    idxs_final = jnp.where(idxs_new >= 0, idxs_new, idxs_topup)

    Yh     = Ym + Mf @ (Xq - Xm)
    cov_Yh = Cyy - Mf @ Cxy

    return idxs_final, Mf, Cxy, Xm, Ym, Yh, cov_Yh

# ── Main function ─────────────────────────────────────────────────────────────
def ens_CGP(Xens_j, Yens_j, Xq, VI_j, N_neighbor, tol=1e-6, max_iter=100):
    """
    Parameters:
    Xens_j: array of input features which compose the ensemble. shape:(D, N) 
    Yens_j: array of input labels which compose the ensemble. shape:(M, N) 
    Xq: query point for which we want to compute a prediction. shape:(D,) or (D,1)
    VI_j: inverse of the covariance matrix for the input ensemble. shape:(D, D)
    N_neighbor: int, number of neighbors to use in CGP
    tol: float, convergence threshold on average relative change in prediction (default 1%)
    max_iter: int, safety cap on number of iterations (default 100)
    """
    Xq_j = jnp.array(Xq.ravel())   # (D,)

    idxs = _mahal_knn_single(Xens_j, Xq_j, VI_j, N_neighbor)

    # Run first iteration to get an initial prediction
    idxs, _, _, _, _, Yh_prev, cov_Yh = _cgp_step_fixed(
        Xens_j, Yens_j, idxs, Xq_j[:, None], VI_j, N_neighbor
    )
    Yh_prev = np.array(Yh_prev.flatten())

    for i in range(max_iter - 1):
        idxs, _, _, _, _, Yh, cov_Yh = _cgp_step_fixed(
            Xens_j, Yens_j, idxs, Xq_j[:, None], VI_j, N_neighbor
        )
        Yh = np.array(Yh.flatten())

        # Average relative change between this and previous prediction
        # Add small epsilon to denominator to avoid division by zero
        rel_change = np.mean(
            np.abs(Yh - Yh_prev) / (np.abs(Yh_prev) + 1e-10)
        )

        if rel_change < tol:
            break

        Yh_prev = Yh

    err_Yh = jnp.sqrt(jnp.maximum(0.0, jnp.diag(cov_Yh)))
    return Yh, np.array(err_Yh), i + 2   # +2 because of the initial iteration before the loop

if plot_2:

    # ── Step 1: pick a single query point ────────────────────────────────────
    query_idx   = 0
    query_input = raw_inputs[query_idx, :]       # (4,)
    query_T     = raw_outputs_T[query_idx, :]    # (O,)
    query_P     = raw_outputs_P[query_idx, :]    # (O,)

    # Remove query point from the pool so it can't appear in any training set
    pool_inputs   = np.delete(raw_inputs,    query_idx, axis=0)   # (N-1, 4)
    pool_outputs_T = np.delete(raw_outputs_T, query_idx, axis=0)  # (N-1, O)
    pool_outputs_P = np.delete(raw_outputs_P, query_idx, axis=0)  # (N-1, O)

    # ── Step 2: find the 5000 closest neighbours to the query point ───────────
    X_pool = pool_inputs.T                                              # (4, N-1)
    VI_pool = jnp.linalg.inv(jnp.cov(jnp.array(X_pool)))
    dists   = np.array(_mahal_knn_single(
                  jnp.array(X_pool),
                  jnp.array(query_input),
                  VI_pool,
                  5000
              ))   # indices of 5000 nearest neighbours in pool

    # Subset pool to these 5000 neighbours
    nbr_inputs    = pool_inputs[dists, :]      # (5000, 4)
    nbr_outputs_T = pool_outputs_T[dists, :]   # (5000, O)
    nbr_outputs_P = pool_outputs_P[dists, :]   # (5000, O)

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    N_sets        = 50
    N_subset      = 1000
    N_neigbors    = [10]

    # ── Step 3+4: for each K, run ens-CGP on 20 random subsets ───────────────
    # predictions[k_idx, set_idx, :] = predicted Y for that K and subset
    predictions_T = np.zeros((len(N_neigbors), N_sets, raw_outputs_T.shape[1]))
    predictions_P = np.zeros((len(N_neigbors), N_sets, raw_outputs_P.shape[1]))

    rng = np.random.default_rng(42)

    for set_idx in tqdm(range(N_sets), desc='Random subsets'):

        # Draw 500 random points from the 1000 neighbours
        subset_idxs = rng.choice(1000, size=N_subset, replace=False)

        XTr = nbr_inputs[subset_idxs, :].T                              # (4, 500)
        YTr = np.hstack([
                  nbr_outputs_T[subset_idxs, :],
                  nbr_outputs_P[subset_idxs, :]
              ]).T                                                        # (M, 500)

        Xens_j = jnp.array(XTr)
        Yens_j = jnp.array(YTr)
        VI_j   = jnp.linalg.inv(jnp.cov(Xens_j))

        for k_idx, N_neighbor in enumerate(N_neigbors):
            Yh, _, it = ens_CGP(
                Xens_j, Yens_j,
                query_input,
                VI_j,
                N_neighbor,
            )
            print(f'Number of iterations: {it}')
            predictions_T[k_idx, set_idx, :] = Yh[:O]
            predictions_P[k_idx, set_idx, :] = Yh[O:]

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.plot(query_T, query_P, color='blue')
            ax.plot(predictions_T[k_idx, set_idx, :], predictions_P[k_idx, set_idx, :], color='green')
            ax.invert_yaxis()
            plt.show()

    # ── Step 5: compute bias and variance across the 20 predictions ───────────
    # For a fixed query point and fixed K:
    # E[ŷ] = mean over the 20 training sets  → axis=1
    # Bias  = E[ŷ] - truth                   → scalar per K per output level
    # Var   = E[(ŷ - E[ŷ])²]                 → scalar per K per output level
    # Then average over output levels to get one number per K

    bias_T = np.zeros(len(N_neigbors))
    bias_P = np.zeros(len(N_neigbors))
    var_T  = np.zeros(len(N_neigbors))
    var_P  = np.zeros(len(N_neigbors))
    mse_T  = np.zeros(len(N_neigbors))
    mse_P  = np.zeros(len(N_neigbors))

    for k_idx in range(len(N_neigbors)):

        mean_pred_T = predictions_T[k_idx].mean(axis=0)   # (O,) — E[ŷ] over sets
        mean_pred_P = predictions_P[k_idx].mean(axis=0)

        # Bias: mean prediction minus truth, averaged over output levels
        bias_T[k_idx] = np.mean(mean_pred_T - query_T)
        bias_P[k_idx] = np.mean(mean_pred_P - query_P)

        # Variance: mean over sets of squared deviation from mean prediction,
        # then averaged over output levels
        var_T[k_idx] = np.mean(
            np.mean((predictions_T[k_idx] - mean_pred_T)**2, axis=0)
        )
        var_P[k_idx] = np.mean(
            np.mean((predictions_P[k_idx] - mean_pred_P)**2, axis=0)
        )

        mse_T[k_idx] = bias_T[k_idx]**2 + var_T[k_idx]
        mse_P[k_idx] = bias_P[k_idx]**2 + var_P[k_idx]

    # ── Step 6: plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 8))

    for col, (b, v, m, label, color) in enumerate([
        (np.abs(bias_T), var_T, mse_T, 'T (K)',         'blue'),
        (np.abs(bias_P), var_P, mse_P, 'P (log10 bar)', 'orange'),
    ]):
        ax[0, col].plot(N_neigbors, b,  color=color, marker='o')
        ax[0, col].axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax[0, col].set_ylabel(f'Bias {label}')

        ax[1, col].plot(N_neigbors, v,  color=color, marker='o', linestyle='--')
        ax[1, col].set_ylabel(f'Variance {label}')

        ax[2, col].plot(N_neigbors, m,  color=color, marker='o')
        ax[2, col].set_ylabel(f'MSE {label}')
        ax[2, col].set_xlabel('Number of Neighbours K')

    plt.suptitle(
        rf'Query point — H$_2$: {query_input[0]:.2f} bar, '
        rf'CO$_2$: {query_input[1]:.2f} bar, '
        rf'LoD: {query_input[2]:.0f} days, '
        rf'Obliquity: {query_input[3]:.0f}°',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(plot_save_path + 'Bias_Variance.pdf')
    plt.close()