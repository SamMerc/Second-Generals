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
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
from jax import jit, vmap
from functools import partial
import jax.numpy as jnp
from time import time
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import glob
torch.set_float32_matmul_precision('high')
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import warnings
warnings.filterwarnings('ignore')  # suppress Lightning verbosity during search

##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/home/merci228/WORK/2G_ML/'
#File containing temperature values
raw_T_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_T.csv', delimiter=',')
raw_T_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_P.csv', delimiter=',')
raw_P_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/Hyperparam_tuning_LRinit_NNdepth_NNwidth_L2_BS/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/Hyperparam_tuning_LRinit_NNdepth_NNwidth_L2_BS/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
# Extract the 4 physical inputs and append stellar temperature as 5th column
inputs_3000 = np.hstack([raw_T_data3000[:, :4], np.full((len(raw_T_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_T_data4500[:, :4], np.full((len(raw_T_data4500), 1), 4500.0)])

# # Concatenate along the sample axis
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

## HYPER-PARAMETERS for ens-CGP ##

#Number of nearest neighbors to choose
N_neighbor = 4

#Definine partitiion for splitting NN dataset
data_partition = [0.7, 0.1, 0.2]

#Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 96
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")

#Mode
run_mode = 'search'   # runs Optuna hyperparameter search
# run_mode = 'train'    # trains final model with best params, saves checkpoint
# run_mode = 'evaluate' # loads checkpoint, runs diagnostic plots








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


# Optuna callback to print trial summary after each trial completes
def print_trial_summary(study, trial):
    print(f'\n--- Trial {trial.number} finished ---')
    print(f'  Value (mean val loss): {trial.value:.6f}')
    print(f'  Params: {trial.params}')
    print(f'  Best so far: {study.best_value:.6f} (trial {study.best_trial.number})')
    print(f'  Trials completed: {len(study.trials)}')



###################
#### Build MLP ####
###################
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))   # ← skip connection

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, generator=None):
        super().__init__()
        if generator is not None:
            torch.manual_seed(generator.initial_seed())

        # First layer : Project input to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Stack fully-connected with dimension hidden_dim
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        # Project to output
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)

# PyTorch Lightning DataModule
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs, test_inputs, test_outputs, batch_size, rng):
        super().__init__()

        # Standardizing the output
        ## Create scaler
        out_scaler_T = StandardScaler()
        out_scaler_P = StandardScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        out_scaler_T.fit(train_outputs[:, :O].cpu().numpy())
        out_scaler_P.fit(train_outputs[:, O:].cpu().numpy())

        ## Transform all datasets and convert back to tensors
        train_T_scaled = torch.tensor(out_scaler_T.transform(train_outputs[:, :O].cpu().numpy()), dtype=torch.float32)
        train_P_scaled = torch.tensor(out_scaler_P.transform(train_outputs[:, O:].cpu().numpy()), dtype=torch.float32)

        valid_T_scaled = torch.tensor(out_scaler_T.transform(valid_outputs[:, :O].cpu().numpy()), dtype=torch.float32)
        valid_P_scaled = torch.tensor(out_scaler_P.transform(valid_outputs[:, O:].cpu().numpy()), dtype=torch.float32)

        test_T_scaled = torch.tensor(out_scaler_T.transform(test_outputs[:, :O].cpu().numpy()), dtype=torch.float32)
        test_P_scaled = torch.tensor(out_scaler_P.transform(test_outputs[:, O:].cpu().numpy()), dtype=torch.float32)
        
        # Concatenate
        train_outputs = torch.cat([train_T_scaled, train_P_scaled], dim=1)
        valid_outputs = torch.cat([valid_T_scaled, valid_P_scaled], dim=1)
        test_outputs = torch.cat([test_T_scaled, test_P_scaled], dim=1)

        # Store the scaler if you need to inverse transform later
        self.out_scaler_T = out_scaler_T
        self.out_scaler_P = out_scaler_P
        
        # --- Input scaling: one scaler per block ---
        # Block indices: [phys(D) | GP_T(O) | GP_P(O) | GP_Terr(O) | GP_Perr(O)]
        i0, i1, i2, i3, i4 = 0, D, D+O, D+2*O, D+3*O

        in_scaler_phys = StandardScaler()
        in_scaler_T    = StandardScaler()
        in_scaler_P    = StandardScaler()
        in_scaler_Terr = StandardScaler()
        in_scaler_Perr = StandardScaler()

        in_scaler_phys.fit(train_inputs[:, i0:i1].cpu().numpy())
        in_scaler_T.fit(   train_inputs[:, i1:i2].cpu().numpy())
        in_scaler_P.fit(   train_inputs[:, i2:i3].cpu().numpy())
        in_scaler_Terr.fit(train_inputs[:, i3:i4].cpu().numpy())
        in_scaler_Perr.fit(train_inputs[:, i4:  ].cpu().numpy())

        def scale_inputs(X):
            X = X.cpu().numpy()
            return torch.tensor(np.hstack([
                in_scaler_phys.transform(X[:, i0:i1]),
                in_scaler_T.transform(   X[:, i1:i2]),
                in_scaler_P.transform(   X[:, i2:i3]),
                in_scaler_Terr.transform(X[:, i3:i4]),
                in_scaler_Perr.transform(X[:, i4:  ]),
            ]), dtype=torch.float32)

        self.train_inputs  = scale_inputs(train_inputs)
        self.valid_inputs  = scale_inputs(valid_inputs)
        self.test_inputs   = scale_inputs(test_inputs)

        # Store all scalers for inference
        self.in_scaler_phys = in_scaler_phys
        self.in_scaler_T    = in_scaler_T
        self.in_scaler_P    = in_scaler_P
        self.in_scaler_Terr = in_scaler_Terr
        self.in_scaler_Perr = in_scaler_Perr

        # Storing it and passing it to loaders
        self.train_outputs = train_outputs
        self.valid_outputs = valid_outputs
        self.test_outputs = test_outputs
        self.batch_size = batch_size
        self.rng = rng
    
    def train_dataloader(self):
        dataset = TensorDataset(self.train_inputs, self.train_outputs)
        return DataLoader(
         dataset,
         batch_size=self.batch_size, 
         shuffle=True, 
         generator=self.rng,
         pin_memory=True,
        #  persistent_workers=True,
         )
    
    def val_dataloader(self):
        dataset = TensorDataset(self.valid_inputs, self.valid_outputs)
        return DataLoader(
         dataset,
         batch_size=self.batch_size, 
         generator=self.rng,
         pin_memory=True,
        #  persistent_workers=True,
         )

    def test_dataloader(self):
        dataset = TensorDataset(self.test_inputs, self.test_outputs)
        return DataLoader(
         dataset,
         batch_size=self.batch_size, 
         generator=self.rng,
         pin_memory=True,
        #  persistent_workers=True,
         )




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


# Targets are residuals: truth - GP prediction
residuals_T = raw_outputs_T - GP_outputs_T   # (N, O)
residuals_P = raw_outputs_P - GP_outputs_P   # (N, O)

# PyTorch Lightning Module
class RegressionModule(pl.LightningModule):
    def __init__(self, model, optimizer, learning_rate, weight_decay=0.0,
                 reg_coeff_l1=0.0, reg_coeff_l2=0.0, smoothness_coeff=0.0,
                 lr_patience=10, lr_factor=0.5, lr_min=1e-7):
        super().__init__()
        self.model            = model
        self.learning_rate    = learning_rate
        self.reg_coeff_l1     = reg_coeff_l1
        self.reg_coeff_l2     = reg_coeff_l2
        self.smoothness_coeff = smoothness_coeff
        self.weight_decay     = weight_decay
        self.loss_fn          = nn.MSELoss()
        self.optimizer_class  = optimizer
        self.lr_patience      = lr_patience
        self.lr_factor        = lr_factor
        self.lr_min           = lr_min
    
    def compute_weight_regularization(self):
        """
        Compute L1 and L2 regularization on model weights (parameters).
        """
        if self.reg_coeff_l1 == 0 and self.reg_coeff_l2 == 0:
            return torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)
        
        l1_penalty = torch.tensor(0., device=self.device)
        l2_penalty = torch.tensor(0., device=self.device)
        
        for param in self.model.parameters():
            if self.reg_coeff_l1 > 0:
                l1_penalty += torch.sum(torch.abs(param))
            if self.reg_coeff_l2 > 0:
                l2_penalty += torch.sum(param ** 2)
        
        return self.reg_coeff_l1 * l1_penalty, self.reg_coeff_l2 * l2_penalty

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        X, y = batch
        if self.smoothness_coeff > 0:
            X.requires_grad_(True)
        pred = self(X)
        
        # Base loss: ||y - s||
        mse = self.loss_fn(pred, y)
        
        # Add weight regularization (L1/L2 on network parameters)
        l1_penalty, l2_penalty = self.compute_weight_regularization()
        loss = mse + l1_penalty + l2_penalty

        # Compute gradients for smoothness penalty
        if self.smoothness_coeff > 0:
            # Compute gradient of loss w.r.t. inputs
            grad_loss = torch.autograd.grad(
                outputs=mse,
                inputs=X,
                create_graph=True,
                retain_graph=True,
            )[0]
            
            # Penalize large input gradients (smoother decision boundaries)
            smoothness_penalty = self.smoothness_coeff * torch.mean(grad_loss ** 2)
            loss += smoothness_penalty

        # Log metrics
        self.log('train_mse', mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)

        # Log metrics
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
        ) 
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss',   # ReduceLROnPlateau needs a metric to watch
                'interval': 'epoch',
                'frequency': 1,
            }
        }

if run_mode == 'search':

    ##################################
    #### Optuna Objective Function ###
    ##################################

    PARTITION_SEEDS = [4]          # Stage 1: single seed
    # PARTITION_SEEDS = [4, 7, 13] # Stage 2: uncomment for multi-seed validation

    # Fixed hyperparameters (not searched)
    BATCH_SEED   = 5
    NN_SEED      = 6
    LR_PATIENCE  = 50
    LR_FACTOR    = 0.7
    LR_MIN       = 1e-7
    N_EPOCHS     = 5000
    ES_PATIENCE  = 100   # tighter than final model for faster search

    def build_data_module(partition_seed, batch_size):
        """Rebuild data splits with a given partition seed."""
        _partition_rng = torch.Generator()
        _partition_rng.manual_seed(partition_seed)

        _batch_rng = torch.Generator()
        _batch_rng.manual_seed(BATCH_SEED)

        _train_idx, _valid_idx, _test_idx = torch.utils.data.random_split(
            range(N), data_partition, generator=_partition_rng
        )

        # --- Inputs ---
        _train_inputs = torch.cat([
            torch.tensor(raw_inputs[_train_idx],        dtype=torch.float32),
            torch.tensor(GP_outputs_T[_train_idx],      dtype=torch.float32),
            torch.tensor(GP_outputs_P[_train_idx],      dtype=torch.float32),
            torch.tensor(GP_outputs_Terr[_train_idx],   dtype=torch.float32),
            torch.tensor(GP_outputs_Perr[_train_idx],   dtype=torch.float32),
        ], dim=1)
        _valid_inputs = torch.cat([
            torch.tensor(raw_inputs[_valid_idx],        dtype=torch.float32),
            torch.tensor(GP_outputs_T[_valid_idx],      dtype=torch.float32),
            torch.tensor(GP_outputs_P[_valid_idx],      dtype=torch.float32),
            torch.tensor(GP_outputs_Terr[_valid_idx],   dtype=torch.float32),
            torch.tensor(GP_outputs_Perr[_valid_idx],   dtype=torch.float32),
        ], dim=1)
        _test_inputs = torch.cat([
            torch.tensor(raw_inputs[_test_idx],         dtype=torch.float32),
            torch.tensor(GP_outputs_T[_test_idx],       dtype=torch.float32),
            torch.tensor(GP_outputs_P[_test_idx],       dtype=torch.float32),
            torch.tensor(GP_outputs_Terr[_test_idx],    dtype=torch.float32),
            torch.tensor(GP_outputs_Perr[_test_idx],    dtype=torch.float32),
        ], dim=1)

        # --- Outputs (residuals) ---
        _train_outputs = torch.cat([
            torch.tensor(residuals_T[_train_idx], dtype=torch.float32),
            torch.tensor(residuals_P[_train_idx], dtype=torch.float32),
        ], dim=1)
        _valid_outputs = torch.cat([
            torch.tensor(residuals_T[_valid_idx], dtype=torch.float32),
            torch.tensor(residuals_P[_valid_idx], dtype=torch.float32),
        ], dim=1)
        _test_outputs = torch.cat([
            torch.tensor(residuals_T[_test_idx],  dtype=torch.float32),
            torch.tensor(residuals_P[_test_idx],  dtype=torch.float32),
        ], dim=1)

        return CustomDataModule(
            _train_inputs, _train_outputs,
            _valid_inputs, _valid_outputs,
            _test_inputs,  _test_outputs,
            batch_size, _batch_rng,
        )


    def objective(trial):
        # ── Sample hyperparameters ──────────────────────────────────
        lr_init    = trial.suggest_float('lr_init',  1e-4, 1e-2,  log=True)
        nn_depth   = trial.suggest_categorical('nn_depth', [4, 8, 16, 32])
        nn_width   = trial.suggest_categorical('nn_width', [209, 313, 418])
        reg_l2     = trial.suggest_float('reg_l2',   1e-6,  1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

        val_losses = []

        for seed_idx, p_seed in enumerate(PARTITION_SEEDS):

            # Fresh data module for this partition seed
            _data_module = build_data_module(p_seed, batch_size)

            # Fresh model with fixed NN seed for reproducibility across trials
            pl.seed_everything(NN_SEED, workers=True)
            _model = NeuralNetwork(D + 4*O, nn_width, 2*O, nn_depth)

            _lightning_module = RegressionModule(
                model=_model,
                optimizer=Adam,
                learning_rate=lr_init,
                reg_coeff_l1=0.0,
                reg_coeff_l2=reg_l2,
                weight_decay=0.0,
                smoothness_coeff=0.0,
                lr_patience=LR_PATIENCE,
                lr_factor=LR_FACTOR,
                lr_min=LR_MIN,
            )

            # Pruning callback — reports valid_loss to Optuna each epoch
            pruning_callback = PyTorchLightningPruningCallback(
                trial, monitor='valid_loss'
            )
            early_stopping = EarlyStopping(
                monitor='valid_loss',
                patience=ES_PATIENCE,
                mode='min',
            )
            checkpoint = ModelCheckpoint(
                monitor='valid_loss',
                mode='min',
                save_top_k=1,
                # Store per trial/seed to avoid collisions
                dirpath=model_save_path + f'optuna_trial{trial.number}_seed{p_seed}/',
            )

            _trainer = Trainer(
                max_epochs=N_EPOCHS,
                callbacks=[pruning_callback, early_stopping, checkpoint],
                enable_progress_bar=False,   # cleaner output
                logger=False,                # no CSV logging during search
                deterministic=True,
                enable_checkpointing=True,
            )

            try:
                _trainer.fit(_lightning_module, datamodule=_data_module)
            except optuna.exceptions.TrialPruned:
                raise  # let Optuna handle pruning cleanly

            best_val = checkpoint.best_model_score
            if best_val is None:
                raise optuna.exceptions.TrialPruned()  # safety: treat failed runs as pruned

            val_losses.append(best_val.item())

        return float(np.mean(val_losses))


    ##################################
    #### Run the study ####
    ##################################

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,    # don't prune until 5 trials complete
        n_warmup_steps=50,     # don't prune before epoch 50 (let LR schedule warm up)
        interval_steps=10,     # check every 10 epochs
    )

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        storage=f'sqlite:///{model_save_path}optuna_study.db',
        study_name='tp_profile_nn_stage1',
        load_if_exists=True,   # safely resume if interrupted
    )

    study.optimize(
        objective,
        n_trials=50,
        timeout=None,          # no wall-clock limit; rely on n_trials
        gc_after_trial=True,   # free GPU memory between trials
        callbacks=[print_trial_summary],
    )

    ##################################
    #### Report results ####
    ##################################

    print('\n=== Optuna Search Complete ===')
    print(f'Best trial: {study.best_trial.number}')
    print(f'Best val loss: {study.best_value:.6f}')
    print('Best hyperparameters:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')

    # Save full results to CSV for inspection
    results_df = study.trials_dataframe()
    results_df.to_csv(model_save_path + 'optuna_results_stage1.csv', index=False)
    print(f'\nFull results saved to {model_save_path}optuna_results_stage1.csv')

    # Show top 10 trials
    print('\nTop 10 trials:')
    top10 = results_df.sort_values('value').head(10)[
        ['number', 'value', 'params_lr_init', 'params_nn_depth',
        'params_nn_width', 'params_reg_l2', 'params_batch_size']
    ]
    print(top10.to_string(index=False))

elif run_mode == 'train':
    # --- Final training block ---
    # Use best params from study (paste them in manually or load from CSV)
    # Constructs data_module once with final partition_seed
    # Saves best checkpoint + best_ckpt_path.txt
    # Outputs: best model checkpoint

elif run_mode == 'evaluate':
    # --- Diagnostic plots block ---
    # Loads from best_ckpt_path.txt
    # Runs test set evaluation
    # Generates all plots    



    ##########################
    #### Diagnostic plots ####
    ##########################
    # Loss curves
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3, 1]}, figsize=(10, 6))

    # Calculate number of batches per epoch
    actual_epochs = len(eval_losses)  # one entry per epoch
    n_batches = len(train_losses) // actual_epochs  # batches per epoch
    n_batches = max(1, n_batches)     # safety guard

    # Create x-axis in terms of epochs (0 to n_epochs)
    x_all = np.linspace(0, actual_epochs, len(train_losses))
    x_epoch = np.arange(actual_epochs + 1)

    # Plot transparent background showing all batch losses
    ax1.plot(x_all, train_losses, alpha=0.3, color='C0', linewidth=0.5)
    ax1.plot(x_all, eval_losses, alpha=0.3, color='C1', linewidth=0.5)

    # Plot solid lines showing epoch-level losses (every n_batches steps)
    train_epoch = [train_losses[0]] + train_losses[n_batches-1::n_batches]
    eval_epoch  = [eval_losses[0]]  + eval_losses[n_batches-1::n_batches]
    ax1.plot(x_epoch, train_epoch, label="Train", color='C0', linewidth=2, marker='o')
    ax1.plot(x_epoch, eval_epoch, label="Validation", color='C1', linewidth=2, marker='o')

    # Same for difference plot
    diff_epoch  = np.abs(np.array(train_epoch) - np.array(eval_epoch))

    ax2.plot(x_epoch, diff_epoch, color='C2', linewidth=2, marker='o')

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax2.set_ylabel("Loss Diff.")
    ax1.legend()
    ax1.grid()
    ax2.grid()
    plt.subplots_adjust(hspace=0)
    plt.savefig(plot_save_path+'/loss.pdf')
    plt.close()

    #Comparing GP predicted T-P profiles vs NN predicted T-P profiles vs true T-P profiles with residuals
    substep = 100

    # Get the scalers from data module
    out_scaler_T = data_module.out_scaler_T
    out_scaler_P = data_module.out_scaler_P
    in_scaler_phys = data_module.in_scaler_phys
    in_scaler_T    = data_module.in_scaler_T
    in_scaler_P    = data_module.in_scaler_P
    in_scaler_Terr = data_module.in_scaler_Terr
    in_scaler_Perr = data_module.in_scaler_Perr

    #Converting tensors to numpy arrays if this isn't already done
    if (type(NN_test_outputs_T) != np.ndarray):
        NN_test_outputs_T = NN_test_outputs_T.cpu().numpy()
        NN_test_outputs_P = NN_test_outputs_P.cpu().numpy()

    GP_res_T = np.zeros(NN_test_outputs_P.shape, dtype=float)
    GP_res_P = np.zeros(NN_test_outputs_P.shape, dtype=float)
    NN_res_T = np.zeros(NN_test_outputs_P.shape, dtype=float)
    NN_res_P = np.zeros(NN_test_outputs_P.shape, dtype=float)

    for NN_test_idx, (NN_test_input, GP_test_output_T, GP_test_output_P,
                    GP_test_err_T, GP_test_err_P,
                    NN_test_output_T, NN_test_output_P,
                    true_T, true_P) in enumerate(zip(
        NN_test_inputs_phys,
        NN_test_inputs_T,   NN_test_inputs_P,
        NN_test_inputs_Terr, NN_test_inputs_Perr,
        NN_test_outputs_T,  NN_test_outputs_P,
        NN_test_true_T,     NN_test_true_P
    )):

        scaled_input = torch.tensor(np.hstack([
            in_scaler_phys.transform(NN_test_input.numpy().reshape(1, -1)),
            in_scaler_T.transform(   GP_test_output_T.numpy().reshape(1, -1)),
            in_scaler_P.transform(   GP_test_output_P.numpy().reshape(1, -1)),
            in_scaler_Terr.transform(GP_test_err_T.numpy().reshape(1, -1)),
            in_scaler_Perr.transform(GP_test_err_P.numpy().reshape(1, -1)),
        ]), dtype=torch.float32)

        NN_pred_output = model(scaled_input).detach().numpy()
        
        #Inverse scaling - NN predicts the residuals not the profile
        pred_resid_T = out_scaler_T.inverse_transform(NN_pred_output[:, :O].reshape(1, -1)).flatten()
        pred_resid_P = out_scaler_P.inverse_transform(NN_pred_output[:, O:].reshape(1, -1)).flatten()

        # Final prediction = GP prediction + NN residual correction
        NN_pred_output_T = GP_test_output_T.numpy() + pred_resid_T
        NN_pred_output_P = GP_test_output_P.numpy() + pred_resid_P

        #Convert to numpy
        true_T_np = true_T.cpu().numpy()
        true_P_np = true_P.cpu().numpy()
        NN_test_input = NN_test_input.cpu().numpy()

        #Storing residuals 
        GP_res_T[NN_test_idx, :] = GP_test_output_T.numpy() - true_T_np
        GP_res_P[NN_test_idx, :] = GP_test_output_P.numpy() - true_P_np
        NN_res_T[NN_test_idx, :] = NN_pred_output_T - true_T_np
        NN_res_P[NN_test_idx, :] = NN_pred_output_P - true_P_np

        #Plotting
        if (NN_test_idx % substep == 0):
            fig, axs = plt.subplot_mosaic([['res_pressure', '.'],
                                        ['results', 'res_temperature']],
                                figsize=(8, 6),
                                width_ratios=(3, 1), height_ratios=(1, 3),
                                layout='constrained')        
            axs['results'].plot(true_T_np, true_P_np, '.', linestyle='-', color='blue', linewidth=2, label='Truth')
            axs['results'].plot(NN_pred_output_T, NN_pred_output_P, color='green', linewidth=2, label='NN prediction')
            axs['results'].plot(GP_test_output_T, GP_test_output_P, color='red', linewidth=2, label='GP prediction')
            axs['results'].invert_yaxis()
            axs['results'].set_ylabel(r'log$_{10}$ Pressure (bar)')
            axs['results'].set_xlabel('Temperature (K)')
            axs['results'].legend()
            axs['results'].grid()

            axs['res_temperature'].plot(NN_res_T[NN_test_idx, :], true_P_np, '.', linestyle='-', color='green', linewidth=2)
            axs['res_temperature'].plot(GP_res_T[NN_test_idx, :], true_P_np, '.', linestyle='-', color='red', linewidth=2)
            axs['res_temperature'].set_xlabel('Residuals (K)')
            axs['res_temperature'].invert_yaxis()
            axs['res_temperature'].grid()
            axs['res_temperature'].axvline(0, color='black', linestyle='dashed', zorder=2)
            axs['res_temperature'].yaxis.tick_right()
            axs['res_temperature'].yaxis.set_label_position("right")
            axs['res_temperature'].sharey(axs['results'])

            axs['res_pressure'].plot(true_T_np, NN_res_P[NN_test_idx, :], '.', linestyle='-', color='green', linewidth=2)
            axs['res_pressure'].plot(true_T_np, GP_res_P[NN_test_idx, :], '.', linestyle='-', color='red', linewidth=2)
            axs['res_pressure'].set_ylabel('Residuals (bar)')
            axs['res_pressure'].invert_yaxis()
            axs['res_pressure'].grid()
            axs['res_pressure'].axhline(0, color='black', linestyle='dashed', zorder=2)
            axs['res_pressure'].xaxis.tick_top()
            axs['res_pressure'].xaxis.set_label_position("top")
            axs['res_pressure'].sharex(axs['results'])

            plt.suptitle(rf'H$_2$ : {NN_test_input[0]} bar, CO$_2$ : {NN_test_input[1]} bar, LoD : {NN_test_input[2]:.0f} days, Obliquity : {NN_test_input[3]} deg, Teff : {NN_test_input[4]} K')
            plt.savefig(plot_save_path+f'/pred_vs_actual_n.{NN_test_idx}.pdf')  
            plt.close()


    #Plot residuals
    fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2, 2, sharex=True, figsize=[12, 8])
    ax1.plot(GP_res_T.T, alpha=0.1, color='green')
    ax2.plot(GP_res_P.T, alpha=0.1, color='green')
    ax3.plot(NN_res_T.T, alpha=0.1, color='blue')
    ax4.plot(NN_res_P.T, alpha=0.1, color='blue')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0, color='black', linestyle='dashed')
        ax.grid()
    ax2.set_xlabel('Index')
    ax4.set_xlabel('Index')
    ax1.set_ylabel('Temperature')
    ax2.set_ylabel('log$_{10}$ Pressure (bar)')
    ax3.set_ylabel('Temperature')
    ax4.set_ylabel('log$_{10}$ Pressure (bar)')
    plt.subplots_adjust(hspace=0.1, bottom=0.25)

    # Add statistics text at the bottom
    stats_text = (
        f"--- GP Residuals ---\n"
        f"Temperature Residuals : Median = {np.median(GP_res_T):.2f} K, Std = {np.std(GP_res_T):.2f} K\n"
        f"Pressure Residuals : Median = {np.median(GP_res_P):.2f} $log_{{10}}$ bar, Std = {np.std(GP_res_P):.2f} $log_{{10}}$ bar\n"
        f"\n"
        f"--- NN Residuals ---\n"
        f"Temperature Residuals : Median = {np.median(NN_res_T):.2f} K, Std = {np.std(NN_res_T):.2f} K\n"
        f"Pressure Residuals : Median = {np.median(NN_res_P):.3f} $log_{{10}}$ bar, Std = {np.std(NN_res_P):.2f} $log_{{10}}$ bar"
    )

    fig.text(0.1, 0.05, stats_text, fontsize=10, family='monospace',
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(plot_save_path+f'/res_GP_NN.pdf', bbox_inches='tight')
    plt.close()