#############################
#### Importing libraries ####
#############################
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from matplotlib import animation
torch.set_float32_matmul_precision('high')

##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing surface temperature map
raw_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_ST2D.csv', delimiter=',')
raw_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')
#Path to store plots
plot_save_path = base_dir+'Plots/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
# Extract the 4 physical inputs and append stellar temperature as 5th column
inputs_3000 = np.hstack([raw_data3000[:, :4], np.full((len(raw_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_data4500[:, :4], np.full((len(raw_data4500), 1), 4500.0)])

# Concatenate along the sample axis
raw_inputs    = np.vstack([inputs_3000,            inputs_4500           ])  # (N_3000+N_4500, 5)
raw_outputs = np.vstack([raw_data3000[:, 5:],  raw_data4500[:, 5:]])  # (N_3000+N_4500, O)

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs.shape[1] #Number of outputs

# Shuffle data
np.random.seed(3)
rp = np.random.permutation(N) #random permutation of the indices
# Apply random permutation to shuffle the data
raw_inputs = raw_inputs[rp, :]
raw_outputs = raw_outputs[rp, :]

## HYPER-PARAMETERS ##
#Definine partitiion for splitting NN dataset
data_partition = [0.7, 0.1, 0.2]

# Variable to show plots or not 
show_plot = True

#Number of nearest neighbors to choose
N_neigbors = np.linspace(5, 200, 5, dtype=int).tolist()

#Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 96
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")

INPUT_LABELS = [
    r'H$_2$ Pressure (bar)',
    r'CO$_2$ Pressure (bar)',
    r'LoD (days)',
    r'Obliquity (deg)',
    r'T$_{eff}$ (K)',
]

plot_1 = False
plot_2 = True
check_cache = True

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
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for j, ax in zip(range(D), axes.flatten()):
        
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

    var_explained_ST2D = np.cumsum(S_ST2D**2) / np.sum(S_ST2D**2)

    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(S_ST2D, color='blue')
    axes.set_xlabel('Component Index')
    axes.set_ylabel('Singular Value')
    axes.set_yscale('log')
    axestwin = axes.twinx()
    axestwin.set_yscale('log')
    axestwin.plot(var_explained_ST2D, color='red')
    axestwin.set_ylabel('Cumulative variance explained')
    # Find number of components needed to explain threshold % of variance
    threshold = 0.9999
    n_components_ST2D = np.searchsorted(var_explained_ST2D, threshold) + 1
    axestwin.axhline(threshold, color='red', linestyle='--', label=f'{threshold*100}% threshold')
    axestwin.axvline(n_components_ST2D-1, color='green', linestyle='--', label=f'K={n_components_ST2D}')
    plt.legend()
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
    def single(xq):
        diff = X_train - xq[:, None]
        dists_sq = jnp.sum(diff * (VI @ diff), axis=0)
        return jnp.argsort(dists_sq)[:k]
    return vmap(single)(X_queries.T)

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
def ens_CGP(Xens_j, Yens_j, Xq, VI_j, N_neighbor, tol=1e-6, max_iter=1000):
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

    rel_change_history = []

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

        # Oscillation detection: count how many times the current value
        # has appeared in the full history
        n_repeats = np.sum(np.isclose(rel_change_history, rel_change, rtol=1e-3))
        if n_repeats >= 5:
            break

        rel_change_history.append(rel_change)

        Yh_prev = Yh

    err_Yh = jnp.sqrt(jnp.maximum(0.0, jnp.diag(cov_Yh)))
    return Yh, np.array(err_Yh), i + 2   # +2 because of the initial iteration before the loop
        
if plot_2:

    if check_cache and os.path.exists(base_dir+'Model_Storage/gp_ST_cache_Nn4_seed3.npz'):
        print(f'  Loading cached GP outputs from:\n  {base_dir+'Model_Storage/gp_ST_cache_Nn4_seed3.npz'}')
        cache = np.load(base_dir+'Model_Storage/gp_ST_cache_Nn4_seed3.npz')
        GP_outputs     = cache['GP_outputs']
        GP_outputs_err = cache['GP_outputs_err']
        GP_bias = GP_outputs - raw_outputs

        # Plot the errors bars of ens-CGP predictions for each query point and compare that
        # to the bias from the ens-CGP predictions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        for query_idx in range(1, len(raw_inputs), 100):
            query_output = raw_outputs[query_idx]
            guess_ST2D = GP_outputs[query_idx, :]
            guess_ST2Derr = GP_outputs_err[query_idx, :]
            guess_bias = GP_bias[query_idx, :]

            ax1.plot(guess_ST2Derr, label='ens-CGP errorbar', alpha=0.5, color='green')
            ax2.plot(guess_bias, label='ens-CGP errorbar', alpha=0.5, color='orange')
            ax1.set_ylabel('ens-CGP errorbar (K)')
            ax2.set_ylabel('ens-CGP bias (K)')
            ax2.set_xlabel('Index')
        plt.savefig(plot_save_path + 'Bias_vs_Error.pdf')
        plt.close()

        # Plot a map of the scaling, to see where it is most affected
        med_diff_map = np.median(GP_bias-GP_outputs_err, axis=0)
        min_diff_map = np.min(GP_bias-GP_outputs_err, axis=0)
        max_diff_map = np.max(GP_bias-GP_outputs_err, axis=0)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        hm1 = sns.heatmap(med_diff_map.reshape((46, 72)), ax=ax1, cmap='coolwarm')
        hm2 = sns.heatmap(min_diff_map.reshape((46, 72)), ax=ax2, cmap='coolwarm')
        hm3 = sns.heatmap(max_diff_map.reshape((46, 72)), ax=ax3, cmap='coolwarm')
        hm1.collections[0].colorbar.set_label('Temperature (K)')
        hm2.collections[0].colorbar.set_label('Temperature (K)')
        hm3.collections[0].colorbar.set_label('Temperature (K)')
        ax3.set_xlabel('Longitude (degrees)')
        ax1.set_ylabel('Latitude (degrees)')
        ax2.set_ylabel('Latitude (degrees)')
        ax3.set_ylabel('Latitude (degrees)')
        ax1.set_title('Median Difference')
        ax2.set_title('Min Difference')
        ax3.set_title('Max Difference')
        plt.savefig(plot_save_path + 'Scaling_Map.pdf')
        plt.close()

        # ── Representative maps: query points closest to the median / min / max ─────
        # of (bias - errorbar), summarized per query point as its mean over pixels.
        diff_matrix    = GP_bias - GP_outputs_err                     # (N, O)
        diff_per_query = np.mean(diff_matrix, axis=1)                 # (N,)

        median_idx = int(np.argmin(np.abs(diff_per_query - np.median(diff_per_query))))
        min_idx    = int(np.argmin(diff_per_query))
        max_idx    = int(np.argmax(diff_per_query))

        rep_indices = {'Median': median_idx, 'Min': min_idx, 'Max': max_idx}

        fig, axes = plt.subplots(3, 4, figsize=(17, 12), sharex=True)
        for row, (label, idx) in enumerate(rep_indices.items()):
            data_map  = raw_outputs[idx].reshape((46, 72))
            model_map = GP_outputs[idx].reshape((46, 72))
            error_map = GP_outputs_err[idx].reshape((46, 72))
            bias_map = GP_bias[idx].reshape((46, 72))

            ax_data, ax_model, ax_bias, ax_error = axes[row, 0], axes[row, 1], axes[row, 2], axes[row, 3]

            hm1 = sns.heatmap(data_map, ax=ax_data)
            hm1.collections[0].colorbar.set_label('Temperature (K)')
            ax_data.set_title(f'{label} Diff — Data (idx={idx})')

            hm2 = sns.heatmap(model_map, ax=ax_model)
            hm2.collections[0].colorbar.set_label('Temperature (K)')
            ax_model.set_title(f'{label} Diff — GP Model (idx={idx})')

            hm3 = sns.heatmap(error_map, ax=ax_error)
            hm3.collections[0].colorbar.set_label('Temperature (K)')
            ax_error.set_title(f'{label} Diff — Errorbar (idx={idx})')

            hm4 = sns.heatmap(bias_map, ax=ax_bias)
            hm4.collections[0].colorbar.set_label('Temperature (K)')
            ax_bias.set_title(f'{label} Diff — Bias (idx={idx})')

            for ax in (ax_data, ax_model, ax_bias, ax_error):
                ax.set_yticks(np.linspace(0, 46, 5))
                ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
                ax.set_ylabel('Latitude (degrees)')

        for ax in axes[-1, :]:
            ax.set_xticks(np.linspace(0, 72, 5))
            ax.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
            ax.set_xlabel('Longitude (degrees)')

        plt.tight_layout()
        plt.savefig(plot_save_path + 'Representative_Diff_Maps.pdf')
        plt.close()

        # ── GIF: spread of (bias - errorbar) maps across the dataset ─────────────────
        # Subsample every 100th query point (same cadence as the Bias_vs_Error plot
        # above), then order those frames by diff magnitude so the animation
        # progresses from smallest to largest spread instead of jumping around.
        gif_indices      = np.arange(1, N, 100)
        gif_diff_summary = diff_per_query[gif_indices]
        gif_order        = gif_indices[np.argsort(gif_diff_summary)]

        # Per-frame vmin/vmax (rather than one global scale) so each map's own
        # contrast is visible instead of being washed out by the most extreme frame.
        fig, ax = plt.subplots(figsize=(8, 6))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        def _update_diff_frame(frame_idx):
            ax.clear()
            cbar_ax.clear()
            diff_map = diff_matrix[frame_idx].reshape((46, 72))
            frame_vmin = diff_map.min()
            frame_vmax = diff_map.max()
            sns.heatmap(diff_map, ax=ax, cmap='coolwarm', center=0,
                        vmin=frame_vmin, vmax=frame_vmax,
                        cbar=True, cbar_ax=cbar_ax)
            cbar_ax.set_ylabel('Bias - Errorbar (K)')
            ax.set_title(f'idx={frame_idx}, mean diff={diff_per_query[frame_idx]:.2f} K')
            ax.set_xticks(np.linspace(0, 72, 5))
            ax.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
            ax.set_xlabel('Longitude (degrees)')
            ax.set_yticks(np.linspace(0, 46, 5))
            ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
            ax.set_ylabel('Latitude (degrees)')

        ani = animation.FuncAnimation(fig, _update_diff_frame, frames=gif_order)
        ani.save(plot_save_path + 'Bias_Error_Spread.gif', writer='pillow', fps=6)
        plt.close()

    else:
        print(f'  Cache not found. Building GP outputs for all query points and N_neighbors...')

        #Track the bias and variance of the ST2D predictions as a function of N_neighbors
        bias = np.zeros(len(N_neigbors), dtype=float)
        var = np.zeros(len(N_neigbors), dtype=float)
        MSE = np.zeros(len(N_neigbors), dtype=float)

        for NNidx, N_neighbor in enumerate(tqdm(N_neigbors)):

            guess_ST2D = np.zeros(raw_outputs.shape, dtype=float)
            guess_ST2Derr = np.zeros(raw_outputs.shape, dtype=float)

            for query_idx, (query_input, query_output) in enumerate(zip(tqdm(raw_inputs), raw_outputs)):

                # Define the training data for CGP (all data points except the query point)
                XTr = np.delete(raw_inputs.T, query_idx, axis=1)
                YTr = np.delete(raw_outputs.T, query_idx, axis=1)

                Yh, err_Yh, _ = ens_CGP(
                                    jnp.array(XTr),
                                    jnp.array(YTr),
                                    query_input, 
                                    jnp.linalg.inv(jnp.cov(XTr)),
                                    N_neighbor, 
                    )
                guess_ST2D[query_idx, :] = Yh
                guess_ST2Derr[query_idx, :] = err_Yh

                # Diagnostic plot
                if show_plot:
                    IMG_H, IMG_W = 46, 72

                    data_map  = query_output.reshape((IMG_H, IMG_W))
                    model_map = guess_ST2D[query_idx, :].reshape((IMG_H, IMG_W))
                    resid_map = data_map - model_map

                    # Gradient of the GP prediction: treat the lat/lon grid as a
                    # plain x-y grid, wrapping edges around to the opposite side
                    # of the map (same convention used for the CNN smoothness
                    # penalty in Code/ST2D/ST2D_GP_CNN.py).
                    dSdx = np.roll(model_map, -1, axis=1) - model_map   # d/dx, periodic in longitude
                    dSdy = np.roll(model_map, -1, axis=0) - model_map   # d/dy, periodic in latitude

                    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
                        5, 1, figsize=(8, 13), sharex=True, layout='constrained'
                    )

                    ax1.set_title('Data')
                    hm1 = sns.heatmap(data_map, ax=ax1)
                    hm1.collections[0].colorbar.set_label('Temperature (K)')

                    ax2.set_title('GP Model')
                    hm2 = sns.heatmap(model_map, ax=ax2)
                    hm2.collections[0].colorbar.set_label('Temperature (K)')

                    ax3.set_title('Residual (Data - GP Model)')
                    hm3 = sns.heatmap(resid_map, ax=ax3)
                    hm3.collections[0].colorbar.set_label('Temperature (K)')

                    ax4.set_title(r'GP Model $\partial S/\partial x$')
                    hm4 = sns.heatmap(dSdx, ax=ax4, cmap='coolwarm', center=0)
                    hm4.collections[0].colorbar.set_label('Temperature Gradient (K)')

                    ax5.set_title(r'GP Model $\partial S/\partial y$')
                    hm5 = sns.heatmap(dSdy, ax=ax5, cmap='coolwarm', center=0)
                    hm5.collections[0].colorbar.set_label('Temperature Gradient (K)')

                    ax5.set_xticks(np.linspace(0, IMG_W, 5))
                    ax5.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
                    ax5.set_xlabel('Longitude (degrees)')

                    for ax in [ax1, ax2, ax3, ax4, ax5]:
                        ax.set_yticks(np.linspace(0, IMG_H, 5))
                        ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
                        ax.set_ylabel('Latitude (degrees)')

                    plt.suptitle(
                        rf'H$_2$ : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, '
                        rf'LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg'
                    )
                    plt.show()

            bias[NNidx] = np.mean(guess_ST2D - raw_outputs)
            var[NNidx] = np.mean(guess_ST2Derr**2)
            MSE[NNidx] = bias[NNidx]**2 + var[NNidx]

        # #Plot bias and variance as a function of N_neighbors
        fig, ax = plt.subplots(3, 1, figsize=(8, 6))
        ax[0].plot(N_neigbors, bias, label='Bias STD2D', color='orange')
        ax[1].plot(N_neigbors, var, label='Variance STD2D', color='orange', linestyle='--')
        ax[2].plot(N_neigbors, MSE, label='MSE STD2D', color='orange')
        ax[0].set_xlabel('Number of Neighbors')
        ax[1].set_xlabel('Number of Neighbors')
        ax[2].set_xlabel('Number of Neighbors')
        ax[0].set_ylabel('Bias')
        ax[1].set_ylabel('Variance')
        ax[2].set_ylabel('MSE')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.savefig(plot_save_path + 'Bias_Variance.pdf')
        plt.show()