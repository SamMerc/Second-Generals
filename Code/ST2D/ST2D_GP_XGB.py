#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from tqdm import tqdm
from kneed import KneeLocator
import xgboost as xgb
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import glob
from sklearn.decomposition import PCA


##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
def find_threshold_round(rmse_per_round, pct=0.95):
    """Find the round that captures pct of total improvement."""
    initial_rmse = rmse_per_round[0]
    final_rmse   = rmse_per_round[-1]
    total_improvement = initial_rmse - final_rmse
    target_rmse = initial_rmse - pct * total_improvement
    return np.argmax(rmse_per_round <= target_rmse) + 1
#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing surface temperature map
raw_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_ST2D.csv', delimiter=',')
raw_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/GP_ST_XGB/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/GP_ST_XGB/'
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
shuffle_seed = 3
np.random.seed(shuffle_seed)
rp = np.random.permutation(N) #random permutation of the indices
# Apply random permutation to shuffle the data
raw_inputs = raw_inputs[rp, :]
raw_outputs = raw_outputs[rp, :]

## HYPER-PARAMETERS ##

#Number of nearest neighbors to choose
N_neighbor = 4

#Whether to show the plots over the for loop or not 
show_plot = False

#Distance metric to use
distance_metric = 'euclidean' #options: 'euclidean', 'mahalanobis', 'logged_euclidean', 'logged_mahalanobis'

#Convert raw inputs for H2 and CO2 pressures to log10 scale so don't have to deal with it later
if 'logged' in distance_metric:
    raw_inputs[:, 0] = np.log10(raw_inputs[:, 0]) #H2
    raw_inputs[:, 1] = np.log10(raw_inputs[:, 1]) #CO2


import psutil, os
print(f"N={N}, O={O}, D={D}")
n_xgb_features = D + 2 * O
X_features_bytes = N * n_xgb_features * 8  # float64
print(f"XGBoost feature matrix: {X_features_bytes / 1e6:.1f} MB raw")
print(f"Estimated XGBoost peak (~7x): {7 * X_features_bytes / 1e9:.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

############################
#### Plotting functions ####
############################
def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def create_rotating_sphere_gif(flux_map, error_map, lon_edges=None, lat_edges=None,
                                error_scale=0.3, cmap='inferno', figsize=(10, 8),
                                frames_per_axis=90, n_cycles=2, fps=30,
                                rotation_axes='xyz',
                                filename='sphere_rotation.gif',
                                title='Surface Temperature Map'):
    """
    Create a rotating 3D sphere GIF.

    Parameters
    ----------
    rotation_axes : str, any combination of 'x', 'y', 'z'
        e.g. 'x'   → rotates around x only
             'xy'  → alternates x, y, x, y, ...
             'xyz' → cycles x, y, z, x, y, z, ...
             'xz'  → alternates x, z, x, z, ...
    """
    n_lat, n_lon = flux_map.shape

    if lon_edges is None:
        lon_edges = np.linspace(0, 360, n_lon + 1)
    if lat_edges is None:
        lat_edges = np.linspace(-90, 90, n_lat + 1)

    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_grid, lat_grid = np.meshgrid(np.radians(lon_c), np.radians(lat_c))

    # Radius modulated by error
    error_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-10)
    R = 1.0 + error_scale * error_norm

    # Base spherical coordinates
    x0 = R * np.cos(lat_grid) * np.cos(lon_grid)
    y0 = R * np.cos(lat_grid) * np.sin(lon_grid)
    z0 = R * np.sin(lat_grid)

    # Colour mapping
    norm = Normalize(vmin=np.nanmin(flux_map), vmax=np.nanmax(flux_map))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    face_colors = mapper.to_rgba(flux_map)

    # Wrap longitude to close the sphere
    x0 = np.hstack([x0, x0[:, 0:1]])
    y0 = np.hstack([y0, y0[:, 0:1]])
    z0 = np.hstack([z0, z0[:, 0:1]])
    face_colors = np.concatenate([face_colors, face_colors[:, 0:1, :]], axis=1)

    # Build rotation axis map
    axis_map = {
        'x': (rotation_matrix_x, 'X'),
        'y': (rotation_matrix_y, 'Y'),
        'z': (rotation_matrix_z, 'Z'),
    }

    # Parse rotation_axes string into ordered list
    selected_axes = [(axis_map[a][0], axis_map[a][1]) for a in rotation_axes.lower()]

    # Build rotation schedule
    n_selected = len(selected_axes)
    n_total_rotations = n_selected * n_cycles
    n_frames = frames_per_axis * n_total_rotations
    angles = np.linspace(0, 2 * np.pi, frames_per_axis, endpoint=False)

    cumulative_rotation = np.eye(3)
    frame_rotations = []
    frame_labels = []

    for cycle in range(n_cycles):
        for rot_func, label in selected_axes:
            for angle in angles:
                R_current = cumulative_rotation @ rot_func(angle)
                frame_rotations.append(R_current)
                frame_labels.append(label)
            cumulative_rotation = cumulative_rotation @ rot_func(2 * np.pi)

    # Set up figure
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    ax.view_init(elev=20, azim=30)

    surf = [ax.plot_surface(x0, y0, z0, facecolors=face_colors, rstride=1, cstride=1,
                            antialiased=True, shade=False)]

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    cbar = fig.colorbar(mapper, ax=ax, shrink=0.6, pad=0.08, label='Flux / Temperature')
    cbar.ax.yaxis.set_tick_params(color='black')
    cbar.ax.yaxis.label.set_color('black')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='black')

    coords = np.stack([x0.ravel(), y0.ravel(), z0.ravel()], axis=0)

    def update(frame):
        surf[0].remove()
        R_mat = frame_rotations[frame]
        rotated = R_mat @ coords
        x_rot = rotated[0].reshape(x0.shape)
        y_rot = rotated[1].reshape(y0.shape)
        z_rot = rotated[2].reshape(z0.shape)
        surf[0] = ax.plot_surface(x_rot, y_rot, z_rot, facecolors=face_colors,
                                   rstride=1, cstride=1, antialiased=True, shade=False)
        ax.set_title(f'{title}  —  Rotating around {frame_labels[frame]}-axis',
                     fontsize=14, pad=20, color='black')
        return [surf[0]]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    duration = n_frames / fps
    axes_str = ', '.join([s[1] for s in selected_axes])
    print(f"Saving {filename}")
    print(f"  Axes: {axes_str} × {n_cycles} cycles = {n_total_rotations} rotations")
    print(f"  {n_frames} frames at {fps} fps = {duration:.1f}s")
    anim.save(plot_save_path+filename, writer=PillowWriter(fps=fps), dpi=150)
    print(f"Saved: {filename}")
    plt.close(fig)

    return anim

def plot_2d_map_with_errors(flux_map, error_map, lon_edges=None, lat_edges=None,
                            cmap='inferno', figsize=(14, 5)):
    """
    Side-by-side: flux map + error map
    Plus a combined version with error contours overlaid.
    """
    n_lat, n_lon = flux_map.shape

    if lon_edges is None:
        lon_edges = np.linspace(0, 360, n_lon + 1)
    if lat_edges is None:
        lat_edges = np.linspace(-90, 90, n_lat + 1)

    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Flux map
    im1 = axes[0].pcolormesh(lon_edges, lat_edges, flux_map, cmap=cmap, shading='flat')
    axes[0].set_title('Flux / Temperature', fontsize=13)
    axes[0].set_xlabel('Longitude (°)')
    axes[0].set_ylabel('Latitude (°)')
    fig.colorbar(im1, ax=axes[0])

    # Panel 2: Error map
    im2 = axes[1].pcolormesh(lon_edges, lat_edges, error_map, cmap='viridis', shading='flat')
    axes[1].set_title('Uncertainty', fontsize=13)
    axes[1].set_xlabel('Longitude (°)')
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    return fig, axes


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




################################
### Build/Load training set ####
################################

print('BUILDING GP TRAINING SET')

# --- Define a cache path tied to the key hyperparameters ---
gp_cache_path = (
    base_dir
    + f'Model_Storage/gp_ST_cache_Nn{N_neighbor}_seed{shuffle_seed}.npz'
)
matching_files = glob.glob(base_dir+'Model_Storage/gp_ST_cache_*.npz')

if os.path.exists(gp_cache_path):
    # ── Load from cache ───────────────────────────────────────
    print(f'  Loading cached GP outputs from:\n  {gp_cache_path}')
    cache = np.load(gp_cache_path)
    GP_outputs    = cache['GP_outputs']
    GP_outputs_err = cache['GP_outputs_err']

elif matching_files:
    # ── Cache mismatch warning ────────────────────────────────
    raise RuntimeError(
        f'WARNING: A GP cache with different hyperparameters was found:\n'
        f'  {matching_files}\n'
        f'Delete it or update your hyperparameters to match.'
    )
else:
    # ── Compute and cache GP outputs ───────────────────────────
    print(f'  No cache found. Computing GP outputs and saving to:\n  {gp_cache_path}')

    # Initialize array to store NN inputs / GP outputs
    GP_outputs = np.zeros(raw_outputs.shape, dtype=float)
    GP_outputs_err = np.zeros(raw_outputs.shape, dtype=float)

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

        Yh, err_Yh, it = ens_CGP(
                            jnp.array(XTr),
                            jnp.array(YTr),
                            query_input, 
                            jnp.linalg.inv(jnp.cov(XTr)),
                            N_neighbor, 
            )
        GP_outputs[query_idx, :] = Yh
        GP_outputs_err[query_idx, :] = err_Yh
    
    # Save to cache so the loop is skipped next time
    np.savez(
        gp_cache_path,
        GP_outputs=GP_outputs,
        GP_outputs_err=GP_outputs_err,
    )
    print(f'  GP outputs cached to:\n  {gp_cache_path}')

#Diagnostic plot
if show_plot:

    #Convert shape
    plot_query_output = query_output.reshape((46, 72))
    plot_model_output = Yh.reshape((46, 72))
    plot_error_output = err_Yh.reshape((46, 72))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, layout='constrained')        
    # Compute global vmin/vmax across all datasets
    vmin = np.min(query_output)
    vmax = np.max(query_output)
    
    # Plot heatmaps
    ax1.set_title('Data')
    hm1 = sns.heatmap(plot_query_output, ax=ax1)
    cbar = hm1.collections[0].colorbar
    cbar.set_label('Temperature (K)')
    ax2.set_title('Model')
    hm2 = sns.heatmap(plot_model_output, ax=ax2)
    cbar = hm2.collections[0].colorbar
    cbar.set_label('Temperature (K)')

    ax2.set_xticks(np.linspace(0, 72, 5))
    ax2.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
    ax2.set_xlabel('Longitude (degrees)')
    
    # Fix latitude ticks
    for ax in [ax1, ax2]:
        ax.set_yticks(np.linspace(0, 46, 5))
        ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
        ax.set_ylabel('Latitude (degrees)')
    plt.suptitle(rf'H$_2$O : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg, Teff : {query_input[4]} K, Number of iterations: {it}')
    plt.show()

    # create_rotating_sphere_gif(plot_model_output, plot_error_output, rotation_axes='xyz', n_cycles=1, filename='sphere_rotation_xyz.gif')

    # create_rotating_sphere_gif(plot_model_output, plot_error_output, rotation_axes='z', n_cycles=1, filename='sphere_rotation_z.gif')

    # plot_2d_map_with_errors(plot_model_output, plot_error_output)




#Plot the residuals
fig, ax = plt.subplots(1, 1, figsize=[12, 8])

for queryidx in range(N):
    if queryidx == 0:ax.plot(GP_outputs[queryidx, :] - raw_outputs[queryidx, :], alpha=0.1, color='green', label=f'Mean : {np.mean(GP_outputs - raw_outputs):.3f} K, Std : {np.std(GP_outputs - raw_outputs):.3f} K')
    else:ax.plot(GP_outputs[queryidx, :] - raw_outputs[queryidx, :], alpha=0.1, color='green')

ax.axhline(0, color='black', linestyle='dashed')
ax.grid()

ax.set_xlabel('Index')
ax.set_ylabel('Temperature')

ax.legend()

# plt.savefig(plot_save_path+f'/res_GP_NN.pdf', bbox_inches='tight')
plt.close()





############################
#### XGBoost Correction ####
############################

print('TRAINING XGBOOST RESIDUAL CORRECTOR')

# ══════════════════════════════════════════════════════════════════════════════
# ── STEP 1: PCA on outputs ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

n_components = 2

# Fit PCA on the GP outputs (or raw_outputs - both should give similar PCs)
pca_outputs = PCA(n_components=n_components)
pca_outputs.fit(raw_outputs)

print(f"\nPCA on outputs (O={O} → {n_components} PCs):")
print(f"  Explained variance: {pca_outputs.explained_variance_ratio_}")
print(f"  Cumulative:         {np.cumsum(pca_outputs.explained_variance_ratio_)}")

# Transform outputs to PC space
raw_outputs_pc = pca_outputs.transform(raw_outputs)          # (N, n_components)
GP_outputs_pc = pca_outputs.transform(GP_outputs)            # (N, n_components)

# ── Optional: PCA on GP uncertainties (or just keep them as-is) ──────────────
# Since uncertainties are always positive and may not follow same PC structure,
# you have two options:

# Option A: Also apply PCA to uncertainties
GP_outputs_err_pc = pca_outputs.transform(GP_outputs_err)    # (N, n_components)

# Option B: Keep full uncertainties (if memory allows) or use their norm/mean
# GP_outputs_err_summary = np.mean(GP_outputs_err, axis=1, keepdims=True)  # (N, 1)

# ══════════════════════════════════════════════════════════════════════════════
# ── STEP 2: Build feature matrix in PC space ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

X_features_pc = np.hstack([
    raw_inputs,           # (N, D=5)
    GP_outputs_pc,        # (N, n_components=2)
    GP_outputs_err_pc,    # (N, n_components=2)
])  # Final shape: (N, 5 + 2 + 2) = (N, 9)

# ── Feature matrix ────────────────────────────────────────────────────────────
X_features = np.hstack([
    raw_inputs,        # (N, D)
    GP_outputs,        # (N, O)
    GP_outputs_err,    # (N, O)
]) #final : (N, D + 4*O)

print(f"\nFeature matrix size reduction:")
print(f"  Original: {X_features.shape} = {X_features.nbytes/1e6:.1f} MB")
print(f"  PC space: {X_features_pc.shape} = {X_features_pc.nbytes/1e6:.1f} MB")
print(f"  Reduction factor: {X_features.shape[1] / X_features_pc.shape[1]:.1f}x")

# ── Residuals in PC space ─────────────────────────────────────────────────────
residuals_pc = raw_outputs_pc - GP_outputs_pc   # (N, n_components)

# ── Train/test split ──────────────────────────────────────────────────────────
(X_train_pc, X_test_pc,
 resid_train_pc, resid_test_pc,
 out_train_pc, out_test_pc) = train_test_split(
    X_features_pc, residuals_pc,
    raw_outputs_pc,
    test_size=0.2, random_state=42
)

# Also split the full-dimensional outputs for final evaluation
_, out_test_full = train_test_split(
    raw_outputs, test_size=0.2, random_state=42
)

# ── Single multi-output XGBoost ───────────────────────────────────────────────
max_depth = int(np.log2(X_features_pc.shape[1]))
print(f"\nUsing max_depth={max_depth} for {X_features_pc.shape[1]} input features")

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=max_depth,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=10,
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1,
    multi_strategy='multi_output_tree',  # native multi-output mode
)

xgb_model.fit(
    X_train_pc, resid_train_pc,
    eval_set=[(X_test_pc, resid_test_pc)],
    verbose=False,
)

n_trees = xgb_model.best_iteration + 1
print(f"Trees used: {n_trees}")

# ── Extract CGP predictions from feature matrix ───────────────────────────────
# Extract CGP predictions in PC space from feature matrix
cgp_test_pc = X_test_pc[:, raw_inputs.shape[1]:raw_inputs.shape[1]+n_components]

# ── Compute RMSE at each round ────────────────────────────────────────────────
rounds = np.arange(1, n_trees + 1)
rmse = lambda pred, truth: np.sqrt(np.mean((pred - truth)**2))

rmse_per_round = np.zeros(len(rounds))

for r in tqdm(rounds, desc='Computing per-round RMSE'):
    # Predict residuals in PC space
    pred_resid_pc = xgb_model.predict(X_test_pc, iteration_range=(0, r))
    
    # Corrected prediction in PC space
    pred_pc = cgp_test_pc + pred_resid_pc
    
    # Transform back to original space
    pred_full = pca_outputs.inverse_transform(pred_pc)
    
    # Compute RMSE in original space
    rmse_per_round[r-1] = rmse(pred_full, out_test_full)

# ── Knee points ───────────────────────────────────────────────────────────────
knee = KneeLocator(rounds, rmse_per_round, curve='convex', direction='decreasing')
print(f"Knee: {knee.knee} trees")

# ── 1-sigma rule ──────────────────────────────────────────────────────────────
best   = rmse_per_round[n_trees - 1]
std    = np.std(rmse_per_round[max(0, n_trees-20):n_trees])
conservative = np.argmax(rmse_per_round <= best + std) + 1
print(f"1-sigma rule: {conservative} trees")

# ── 95% threshold ─────────────────────────────────────────────────────────────
round_95 = find_threshold_round(rmse_per_round, pct=0.95)
round_99 = find_threshold_round(rmse_per_round, pct=0.99)
print(f"95% improvement: {round_95} trees")
print(f"99% improvement: {round_99} trees")

# ── RMSE plot ─────────────────────────────────────────────────────────────────
cgp_test_full = pca_outputs.inverse_transform(cgp_test_pc)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax1.plot(rounds, rmse_per_round, color='blue', linewidth=1.5)
ax1.axhline(rmse(cgp_test_full, out_test_full), color='grey', linestyle=':', label='CGP only')
ax1.axvline(knee.knee,  color='green', linewidth=2, linestyle='--', label=f'Knee @ {knee.knee}')
ax1.axvline(round_95,   color='black', linewidth=2, linestyle='--', label=f'95% thresh. @ {round_95}')
ax1.axvline(round_99,   color='red', linewidth=2, linestyle='--', label=f'99% thresh. @ {round_99}')
ax1.set_xlabel('Number of trees')
ax1.set_ylabel('RMSE ST2D (K)')
ax1.set_title('ST2D RMSE vs boosting round')
ax1.legend(); ax1.grid()

plt.tight_layout()
plt.savefig(plot_save_path + 'RMS_vs_XGBit.pdf')

# ── NN depth guidance ─────────────────────────────────────────────────────────
max_trees = round_99
print(f"\nXGBoost converged in {max_trees} trees of depth {max_depth}")
print(f"Suggested NN hidden layers : ~{max_trees // 10}")
print(f"Suggested neurons per layer: ~{O}")

# ── Final predictions at knee point ──────────────────────────────────────
pred_resid_knee_pc = xgb_model.predict(X_test_pc, iteration_range=(0, knee.knee))
final_pc = cgp_test_pc + pred_resid_knee_pc
final_full = pca_outputs.inverse_transform(final_pc)

# ── Residual plot ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12, 5))

for qid in range(int(0.2 * N)):
    ax1.plot(cgp_test_full[qid,:] - out_test_full[qid,:], color='green', alpha=0.3)
    ax2.plot(final_full[qid,:]    - out_test_full[qid,:], color='blue',  alpha=0.3)

# Add labelled line for legend stats
ax1.plot([], [], color='green', label=f'CGP.     Mean={np.mean(cgp_test_full-out_test_full):.4f}, Std={np.std(cgp_test_full-out_test_full):.4f}, RMSE={rmse(cgp_test_full,out_test_full):.4f}')
ax2.plot([], [], color='blue',  label=f'CGP+XGB. Mean={np.mean(final_full-out_test_full):.4f}, Std={np.std(final_full-out_test_full):.4f}, RMSE={rmse(final_full,out_test_full):.4f}')

ax1.set_ylabel('Residuals ST2D (K)')
ax2.set_xlabel('Index')

for ax in [ax1, ax2]:
    ax.axhline(0, color='black', linestyle='--')
    ax.grid(); ax.legend()

plt.tight_layout()
plt.savefig(plot_save_path + 'CGP_XGB_residuals.pdf')
plt.show()