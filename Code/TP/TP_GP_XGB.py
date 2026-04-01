#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from tqdm import tqdm
from kneed import KneeLocator
import xgboost as xgb
from sklearn.model_selection import train_test_split
import glob


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
#File containing temperature values
raw_T_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_T.csv', delimiter=',')
raw_T_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_P.csv', delimiter=',')
raw_P_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/GP_XGB/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/GP_XGB/'
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
shuffle_seed = 3
np.random.seed(shuffle_seed)
rp = np.random.permutation(N) #random permutation of the indices
# Apply random permutation to shuffle the data
raw_inputs = raw_inputs[rp, :]
raw_outputs_T = raw_outputs_T[rp, :]
raw_outputs_P = raw_outputs_P[rp, :]

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
    + f'Model_Storage/gp_cache_Nn{N_neighbor}_seed{shuffle_seed}.npz'
)
matching_files = glob.glob(base_dir+'Model_Storage/gp_cache_*.npz')

if os.path.exists(gp_cache_path):
    # ── Load from cache ───────────────────────────────────────
    print(f'  Loading cached GP outputs from:\n  {gp_cache_path}')
    cache = np.load(gp_cache_path)
    GP_outputs_T    = cache['GP_outputs_T']
    GP_outputs_P    = cache['GP_outputs_P']
    GP_outputs_Terr = cache['GP_outputs_Terr']
    GP_outputs_Perr = cache['GP_outputs_Perr']

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
    GP_outputs_T = np.zeros(raw_outputs_T.shape, dtype=float)
    GP_outputs_P = np.zeros(raw_outputs_P.shape, dtype=float)
    GP_outputs_Terr = np.zeros(raw_outputs_T.shape, dtype=float)
    GP_outputs_Perr = np.zeros(raw_outputs_P.shape, dtype=float)

    for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(tqdm(raw_inputs), raw_outputs_T, raw_outputs_P)):

        #Define ensemble without the query point
        XTr = np.delete(
                    raw_inputs.T,
                    query_idx,
                    axis=1
                    )                           # (D, N)
        
        YTr = np.delete(
            np.hstack([raw_outputs_T, raw_outputs_P]).T,   # shape: (2O, N)
            query_idx,
            axis=1
            )                                   # (O, N)

        Xens_j = jnp.array(XTr)
        Yens_j = jnp.array(YTr)
        VI_j   = jnp.linalg.inv(jnp.cov(Xens_j))

        #Call the ens-CGP
        Yh, Yh_err, it = ens_CGP(
                    Xens_j, Yens_j,
                    query_input,
                    VI_j,
                    N_neighbor,
                )
        
        #Store outputs
        GP_outputs_T[query_idx, :] = Yh[:O]
        GP_outputs_Terr[query_idx, :] = Yh_err[:O]
        GP_outputs_P[query_idx, :] = Yh[O:]
        GP_outputs_Perr[query_idx, :] = Yh_err[O:]

    # Save to cache so the loop is skipped next time
    np.savez(
        gp_cache_path,
        GP_outputs_T=GP_outputs_T,
        GP_outputs_P=GP_outputs_P,
        GP_outputs_Terr=GP_outputs_Terr,
        GP_outputs_Perr=GP_outputs_Perr,
    )
    print(f'  GP outputs cached to:\n  {gp_cache_path}')

#Diagnostic plot
if show_plot:
    for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(tqdm(raw_inputs), raw_outputs_T, raw_outputs_P)):
        #Plot TP profiles
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        #ax1 : prediction, truth and prediction errorbars in T
        ax1.errorbar(GP_outputs_T[query_idx, :], GP_outputs_P[query_idx, :], xerr=GP_outputs_Terr[query_idx, :], fmt='.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=2, alpha=0.4, label='Prediction')
        ax1.fill_betweenx(GP_outputs_P[query_idx, :], GP_outputs_T[query_idx, :]-GP_outputs_Terr[query_idx, :], GP_outputs_T[query_idx, :]+GP_outputs_Terr[query_idx, :], color='green', zorder=2, alpha=0.2)
        
        #ax2 : prediction, truth and prediction errorbars in P
        ax2.errorbar(GP_outputs_T[query_idx, :], GP_outputs_P[query_idx, :], yerr=GP_outputs_Perr[query_idx, :], fmt='.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=2, alpha=0.4, label='Prediction')
        ax2.fill_between(GP_outputs_T[query_idx, :], GP_outputs_P[query_idx, :]-GP_outputs_Perr[query_idx, :], GP_outputs_P[query_idx, :]+GP_outputs_Perr[query_idx, :], color='green', zorder=2, alpha=0.2)

        for ax in [ax1, ax2]:

            ax.plot(query_output_T, query_output_P, '.', linestyle='-', color='blue', linewidth=2, zorder=3, markersize=10, label='Truth')

            ax.invert_yaxis()
            
            if ax == ax1 : ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
            ax.set_xlabel('Temperature (K)')
            
            ax.grid()
            ax.legend()        

        plt.suptitle(rf'H$_2$ : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg, Teff : {query_input[4]} K, Number of iterations: {it}')
        plt.subplots_adjust(wspace=0.2)
        plt.show()

#Plot the residuals
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=[12, 8])

for queryidx in range(N):
    if queryidx == 0:ax1.plot(GP_outputs_T[queryidx, :] - raw_outputs_T[queryidx, :], alpha=0.1, color='green', label=f'Mean : {np.mean(GP_outputs_T - raw_outputs_T):.3f} K, Std : {np.std(GP_outputs_T - raw_outputs_T):.3f} K')
    else:ax1.plot(GP_outputs_T[queryidx, :] - raw_outputs_T[queryidx, :], alpha=0.1, color='green')
    if queryidx ==0 : ax2.plot(GP_outputs_P[queryidx, :] - raw_outputs_P[queryidx, :], alpha=0.1, color='green', label=f'Mean : {np.mean(GP_outputs_P - raw_outputs_P):.3f} bar, Std : {np.std(GP_outputs_P - raw_outputs_P):.3f} bar')
    else:ax2.plot(GP_outputs_P[queryidx, :] - raw_outputs_P[queryidx, :], alpha=0.1, color='green')

for ax in [ax1, ax2]:
    ax.axhline(0, color='black', linestyle='dashed')
    ax.grid()

ax2.set_xlabel('Index')

ax1.set_ylabel('Temperature')
ax2.set_ylabel('log$_{10}$ Pressure (bar)')

ax1.legend()
ax2.legend()
plt.subplots_adjust(hspace=0.1, bottom=0.25)

# plt.savefig(plot_save_path+f'/res_GP_NN.pdf', bbox_inches='tight')
plt.show()





############################
#### XGBoost Correction ####
############################

print('TRAINING XGBOOST RESIDUAL CORRECTOR')

# ── Residuals ─────────────────────────────────────────────────────────────────
residuals_T = raw_outputs_T - GP_outputs_T   # (N, O)
residuals_P = raw_outputs_P - GP_outputs_P   # (N, O)
residuals   = np.hstack([residuals_T, residuals_P])  # (N, 2*O)

# ── Feature matrix ────────────────────────────────────────────────────────────
X_features = np.hstack([
    raw_inputs,        # (N, D)
    GP_outputs_T,      # (N, O)
    GP_outputs_P,      # (N, O)
    GP_outputs_Terr,   # (N, O)
    GP_outputs_Perr,   # (N, O)
]) #final : (N, D + 4*O)

# ── Train/test split ──────────────────────────────────────────────────────────
(X_train, X_test,
 resid_train, resid_test,
 _,   out_T_test,
 _,   out_P_test) = train_test_split(
    X_features, residuals,
    raw_outputs_T, raw_outputs_P,
    test_size=0.2, random_state=42
)

# ── Single multi-output XGBoost ───────────────────────────────────────────────
max_depth = int(np.log2(D + 4*O))
print(f"Using max_depth={max_depth} for {D + 4*O} input features")

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
    X_train, resid_train,
    eval_set=[(X_test, resid_test)],
    verbose=False,
)

n_trees = xgb_model.best_iteration + 1
print(f"Trees used: {n_trees}")

# ── Extract CGP predictions from feature matrix ───────────────────────────────
cgp_T_test = X_test[:, D:D+O]
cgp_P_test = X_test[:, D+O:D+2*O]

# ── Compute RMSE at each round ────────────────────────────────────────────────
rounds = np.arange(1, n_trees + 1)
rmse = lambda pred, truth: np.sqrt(np.mean((pred - truth)**2))

rmse_T_per_round = np.zeros(len(rounds))
rmse_P_per_round = np.zeros(len(rounds))

for r in tqdm(rounds, desc='Computing per-round RMSE'):
    pred_resid = xgb_model.predict(X_test, iteration_range=(0, r))  # (N_test, 2*O)
    pred_T = cgp_T_test + pred_resid[:, :O]
    pred_P = cgp_P_test + pred_resid[:, O:]
    rmse_T_per_round[r-1] = rmse(pred_T, out_T_test)
    rmse_P_per_round[r-1] = rmse(pred_P, out_P_test)

# ── Knee points ───────────────────────────────────────────────────────────────
knee_T = KneeLocator(rounds, rmse_T_per_round, curve='convex', direction='decreasing')
knee_P = KneeLocator(rounds, rmse_P_per_round, curve='convex', direction='decreasing')
print(f"Knee — T: {knee_T.knee} trees, P: {knee_P.knee} trees")

# ── 1-sigma rule ──────────────────────────────────────────────────────────────
for label, rmse_per_round, n_t in [('T', rmse_T_per_round, n_trees), ('P', rmse_P_per_round, n_trees)]:
    best   = rmse_per_round[n_t - 1]
    std    = np.std(rmse_per_round[max(0, n_t-20):n_t])
    conservative = np.argmax(rmse_per_round <= best + std) + 1
    print(f"1-sigma rule {label}: {conservative} trees")

# ── 95% threshold ─────────────────────────────────────────────────────────────
round_95_T = find_threshold_round(rmse_T_per_round, pct=0.95)
round_95_P = find_threshold_round(rmse_P_per_round, pct=0.95)
print(f"95% improvement — T: {round_95_T} trees, P: {round_95_P} trees")

# ── RMSE plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax1.plot(rounds, rmse_T_per_round, color='blue', linewidth=1.5)
ax1.axhline(rmse(cgp_T_test, out_T_test), color='grey', linestyle=':', label='CGP only')
ax1.axvline(knee_T.knee,  color='green', linewidth=2, linestyle='--', label=f'Knee @ {knee_T.knee}')
ax1.axvline(round_95_T,   color='black', linewidth=2, linestyle='--', label=f'95% thresh. @ {round_95_T}')
ax1.set_xlabel('Number of trees')
ax1.set_ylabel('RMSE T (K)')
ax1.set_title('T RMSE vs boosting round')
ax1.legend(); ax1.grid()

ax2.plot(rounds, rmse_P_per_round, color='orange', linewidth=1.5)
ax2.axhline(rmse(cgp_P_test, out_P_test), color='grey', linestyle=':', label='CGP only')
ax2.axvline(knee_P.knee,  color='green', linewidth=2, linestyle='--', label=f'Knee @ {knee_P.knee}')
ax2.axvline(round_95_P,   color='black', linewidth=2, linestyle='--', label=f'95% thresh. @ {round_95_P}')
ax2.set_xlabel('Number of trees')
ax2.set_ylabel(r'RMSE P (log$_{10}$ bar)')
ax2.set_title('P RMSE vs boosting round')
ax2.legend(); ax2.grid()

plt.tight_layout()
plt.savefig(plot_save_path + 'RMS_vs_XGBit.pdf')

# ── NN depth guidance ─────────────────────────────────────────────────────────
max_trees = max(knee_T.knee, knee_P.knee)
print(f"\nXGBoost converged in {max_trees} trees of depth {max_depth}")
print(f"Suggested NN hidden layers : ~{max_trees // 10}")
print(f"Suggested neurons per layer: ~{2 * O}")

# ── Corrected predictions at knee point ──────────────────────────────────────
knee_round = max(knee_T.knee, knee_P.knee)
pred_resid_knee = xgb_model.predict(X_test, iteration_range=(0, knee_round))
final_T = cgp_T_test + pred_resid_knee[:, :O]
final_P = cgp_P_test + pred_resid_knee[:, O:]

# ── Residual plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 5))

for qid in range(int(0.2 * N)):
    axes[0,0].plot(cgp_T_test[qid,:] - out_T_test[qid,:], color='green', alpha=0.3)
    axes[1,0].plot(cgp_P_test[qid,:] - out_P_test[qid,:], color='green', alpha=0.3)
    axes[0,1].plot(final_T[qid,:]    - out_T_test[qid,:], color='blue',  alpha=0.3)
    axes[1,1].plot(final_P[qid,:]    - out_P_test[qid,:], color='blue',  alpha=0.3)

# Add labelled line for legend stats
axes[0,0].plot([], [], color='green', label=f'CGP.     Mean={np.mean(cgp_T_test-out_T_test):.4f}, Std={np.std(cgp_T_test-out_T_test):.4f}, RMSE={rmse(cgp_T_test,out_T_test):.4f}')
axes[1,0].plot([], [], color='green', label=f'CGP.     Mean={np.mean(cgp_P_test-out_P_test):.4f}, Std={np.std(cgp_P_test-out_P_test):.4f}, RMSE={rmse(cgp_P_test,out_P_test):.4f}')
axes[0,1].plot([], [], color='blue',  label=f'CGP+XGB. Mean={np.mean(final_T-out_T_test):.4f}, Std={np.std(final_T-out_T_test):.4f}, RMSE={rmse(final_T,out_T_test):.4f}')
axes[1,1].plot([], [], color='blue',  label=f'CGP+XGB. Mean={np.mean(final_P-out_P_test):.4f}, Std={np.std(final_P-out_P_test):.4f}, RMSE={rmse(final_P,out_P_test):.4f}')

axes[0,0].set_ylabel('Residuals T (K)')
axes[1,0].set_ylabel(r'Residuals P (log$_{10}$ bar)')
axes[1,0].set_xlabel('Index')
axes[1,1].set_xlabel('Index')

for ax in axes.ravel():
    ax.axhline(0, color='black', linestyle='--')
    ax.grid(); ax.legend()

plt.tight_layout()
plt.savefig(plot_save_path + 'CGP_XGB_residuals.pdf')
plt.show()