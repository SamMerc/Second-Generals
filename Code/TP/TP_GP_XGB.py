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
#Path to store model
model_save_path = base_dir+'Model_Storage/GP_XGB/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/GP_XGB/'
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




############################
#### Build training set ####
############################

print('BUILDING GP TRAINING SET')

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
                )                           # (4, N)
    
    YTr = np.delete(
        np.hstack([raw_outputs_T, raw_outputs_P]).T,   # shape: (M, N)
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

    #Diagnostic plot
    if show_plot:

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

        plt.suptitle(rf'H$_2$ : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg, Number of iterations: {it}')
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





import xgboost as xgb
from sklearn.model_selection import train_test_split

############################
#### XGBoost Correction ####
############################

print('TRAINING XGBOOST RESIDUAL CORRECTOR')

# ── Residuals ─────────────────────────────────────────────────────────────────
residuals_T = raw_outputs_T - GP_outputs_T   # (N, O)
residuals_P = raw_outputs_P - GP_outputs_P   # (N, O)

# ── Feature matrix ────────────────────────────────────────────────────────────
X_features = np.hstack([
    raw_inputs,       # (N, 4)
    GP_outputs_T,      # (N, O) — CGP predictions
    GP_outputs_P,      # (N, O)
    GP_outputs_Terr,   # (N, O) — CGP uncertainties
    GP_outputs_Perr,   # (N, O)
])

# ── Train/test split ──────────────────────────────────────────────────────────
(X_train, X_test,
 resid_T_train, resid_T_test,
 resid_P_train, resid_P_test,
 out_T_train,   out_T_test,
 out_P_train,   out_P_test) = train_test_split(
    X_features, residuals_T, residuals_P,
    raw_outputs_T, raw_outputs_P,
    test_size=0.2, random_state=42
)

# ── Fit directly to residuals — no PCA needed ─────────────────────────────────
xgb_T = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=10,
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1,
)

xgb_P = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=10,
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1,
)

xgb_T.fit(
    X_train, resid_T_train,
    eval_set=[(X_test, resid_T_test)],
    verbose=False,
)

xgb_P.fit(
    X_train, resid_P_train,
    eval_set=[(X_test, resid_P_test)],
    verbose=False,
)

n_trees_T = xgb_T.best_iteration + 1
n_trees_P = xgb_P.best_iteration + 1
print(f"Trees used — T: {n_trees_T}, P: {n_trees_P}")

# ── Corrected predictions ─────────────────────────────────────────────────────
cgp_T_test = X_test[:, 4:4+O]
cgp_P_test = X_test[:, 4+O:4+2*O]

final_T = cgp_T_test + xgb_T.predict(X_test)
final_P = cgp_P_test + xgb_P.predict(X_test)

# ── Evaluate ──────────────────────────────────────────────────────────────────
rmse = lambda pred, truth: np.sqrt(np.mean((pred - truth)**2))

print(f"RMSE T — CGP: {rmse(cgp_T_test, out_T_test):.4f}  |  CGP+XGB: {rmse(final_T, out_T_test):.4f}")
print(f"RMSE P — CGP: {rmse(cgp_P_test, out_P_test):.4f}  |  CGP+XGB: {rmse(final_P, out_P_test):.4f}")

# ── NN depth guidance ─────────────────────────────────────────────────────────
max_trees = max(n_trees_T, n_trees_P)
print(f"\nXGBoost converged in {max_trees} trees of depth {4}")
print(f"Suggested NN hidden layers : ~{max_trees // 10}")
print(f"Suggested neurons per layer: ~{2 * O}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 5))

for ax, cgp, final, truth, label in zip(
    axes,
    [cgp_T_test, cgp_P_test],
    [final_T,    final_P],
    [out_T_test, out_P_test],
    ['T (K)',    r'P (log$_{10}$ bar)']
):
    
    for queryidx in range(int(0.2 * N)):
        if queryidx == 0:
            ax.plot(cgp[queryidx,:]   - truth[queryidx,:], alpha=0.1, color='blue',  label=f'CGP. Mean = {np.mean(cgp   - truth):.3f}, Std = {np.std(cgp   - truth):.3f}, RMSE = {rmse(cgp,   truth):.4f}')
            ax.plot(final[queryidx,:]   - truth[queryidx,:], alpha=0.1, color='green', label=f'CGP+XGB. Mean = {np.mean(final   - truth):.3f}, Std = {np.std(final   - truth):.3f}, RMSE = {rmse(final,   truth):.4f}')
        else:
            ax.plot(cgp[queryidx,:]   - truth[queryidx,:], alpha=0.1, color='blue')
            ax.plot(final[queryidx,:]   - truth[queryidx,:], alpha=0.1, color='green')

    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel(f'Residuals {label}')
    ax.set_xlabel('Index')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.savefig(plot_save_path + 'CGP_XGB_residuals.pdf')
plt.show()