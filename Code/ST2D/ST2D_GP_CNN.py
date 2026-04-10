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
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tqdm import tqdm
from jax import jit, vmap
from functools import partial
import jax.numpy as jnp
from time import time
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import glob
torch.set_float32_matmul_precision('high')

##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
# Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir): os.mkdir(dir)

# Base directory
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'

# File containing surface temperature map
raw_data3000 = np.loadtxt(base_dir + 'Data/bt-3000k/training_data_ST2D.csv', delimiter=',')
raw_data4500 = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')

# Path to store model
model_save_path = base_dir + 'Model_Storage/GP_ST_ResCNN/'
check_and_make_dir(model_save_path)

# Path to store plots
plot_save_path = base_dir + 'Plots/GP_ST_ResCNN/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
# Extract the 4 physical inputs and append stellar temperature as 5th column
inputs_3000 = np.hstack([raw_data3000[:, :4], np.full((len(raw_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_data4500[:, :4], np.full((len(raw_data4500), 1), 4500.0)])

# Concatenate along the sample axis
raw_inputs  = np.vstack([inputs_3000,           inputs_4500          ])
raw_outputs = np.vstack([raw_data3000[:, 5:],   raw_data4500[:, 5:] ])

# Storing useful quantities
N = raw_inputs.shape[0]
D = raw_inputs.shape[1]
O = raw_outputs.shape[1]

# Map geometry
IMG_H, IMG_W = 46, 72
assert O == IMG_H * IMG_W, f"Output dim {O} != {IMG_H}x{IMG_W}"

# Shuffle data
shuffle_seed = 3
np.random.seed(shuffle_seed)
rp = np.random.permutation(N)
raw_inputs  = raw_inputs[rp, :]
raw_outputs = raw_outputs[rp, :]

## HYPER-PARAMETERS for ens-CGP ##

#Number of nearest neighbors to choose
N_neighbor = 4

## HYPER-PARAMETERS for CNN ##
data_partition = [0.7, 0.1, 0.2]

#Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 96
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")

# Seeds
partition_seed = 4
partition_rng = torch.Generator()
partition_rng.manual_seed(partition_seed)

#Defining the noise seed for the generating of batches from the partitioned data
batch_seed = 5
batch_rng = torch.Generator()
batch_rng.manual_seed(batch_seed)

#Defining the noise seed for the neural network initialization
NN_seed = 6
NN_rng = torch.Generator()
NN_rng.manual_seed(NN_seed)

# Variable to show plots or not 
show_plot = False

# CNN architecture
cnn_hidden_channels = 64   # number of feature maps in residual blocks
cnn_depth = 32             # number of residual conv blocks

# Optimizer learning rate schedule - ReduceLROnPlateau
lr_init      = 1e-3   # initial LR — ReduceLROnPlateau will reduce from here
lr_patience  = 50     # epochs to wait before reducing LR
lr_factor    = 0.7    # multiply LR by this when plateauing
lr_min       = 1e-7   # floor

# Regularization
regularization_coeff_l1 = 0.0
regularization_coeff_l2 = 5e-5
smoothness_coeff = 1e-1
weight_decay = 0.0

# Training
batch_size = 32
n_epochs = 100
early_stopping_patience = 10
run_mode = 'use'


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







###################
#### Build CNN ####
###################
class ResidualConvBlock(nn.Module):
    """2-conv residual block with skip connection (mirrors MLP ResidualBlock)."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))   # ← skip connection


class ResidualCNN(nn.Module):
    """
    Mirrors the MLP architecture pattern:
      input_proj → N x ResidualBlock → output_proj

    Input channels:
      - D channels : physical params broadcast to constant (D, H, W) feature maps
      - 1 channel  : GP prediction   reshaped to (1, H, W)
      - 1 channel  : GP uncertainty  reshaped to (1, H, W)
      Total: D + 2

    Output channels:
      - 1 channel  : residual correction map (1, H, W)
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 depth, img_height, img_width, generator=None):
        super().__init__()
        if generator is not None:
            torch.manual_seed(generator.initial_seed())

        self.img_height = img_height
        self.img_width  = img_width

        # Input projection: map input channels to hidden channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        # Stack of residual conv blocks
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(hidden_channels) for _ in range(depth)]
        )
        # Output projection: map hidden channels to output channels
        self.output_proj = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (batch, D+2, H, W)
        returns : (batch, 1, H, W)
        """
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


# ── CNN-aware DataModule ──────────────────────────────────────────────────────
class CNNDataModule(pl.LightningDataModule):
    """
    Scales inputs block-wise (phys, GP pred, GP err), then
    reshapes into multi-channel images for the CNN.

    Input layout (flat): [phys(D) | GP_pred(O) | GP_err(O)]
    CNN input shape    : (batch, D+2, H, W)
      - channels 0..D-1 : each physical param broadcast to a constant HxW map
      - channel D       : GP prediction  reshaped to (H, W)
      - channel D+1     : GP uncertainty reshaped to (H, W)
    CNN output shape   : (batch, 1, H, W)  ← residual correction
    """
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs,
                 test_inputs, test_outputs, batch_size, rng,
                 n_phys, img_height, img_width):
        super().__init__()
        self.batch_size  = batch_size
        self.rng         = rng
        self.n_phys      = n_phys
        self.img_height  = img_height
        self.img_width   = img_width
        O_flat = img_height * img_width  # 3312

        # ── Block indices in the flat input vector ────────────
        i0 = 0
        i1 = n_phys           # end of phys block
        i2 = i1 + O_flat      # end of GP pred block
        # i3 = i2 + O_flat    # end of GP err block (= total)

        # ── Output scaler (fit on training residuals) ─────────
        out_scaler = StandardScaler()
        out_scaler.fit(train_outputs.cpu().numpy())

        train_outputs = torch.tensor(
            out_scaler.transform(train_outputs.cpu().numpy()), dtype=torch.float32)
        valid_outputs = torch.tensor(
            out_scaler.transform(valid_outputs.cpu().numpy()), dtype=torch.float32)
        test_outputs  = torch.tensor(
            out_scaler.transform(test_outputs.cpu().numpy()),  dtype=torch.float32)
        self.out_scaler = out_scaler

        # ── Input scalers (one per block) ─────────────────────
        in_scaler_phys = StandardScaler()
        in_scaler_pred = StandardScaler()
        in_scaler_err  = StandardScaler()

        in_scaler_phys.fit(train_inputs[:, i0:i1].cpu().numpy())
        in_scaler_pred.fit(train_inputs[:, i1:i2].cpu().numpy())
        in_scaler_err.fit( train_inputs[:, i2:  ].cpu().numpy())

        self.in_scaler_phys = in_scaler_phys
        self.in_scaler_pred = in_scaler_pred
        self.in_scaler_err  = in_scaler_err

        def scale_flat(X):
            X_np = X.cpu().numpy()
            return torch.tensor(np.hstack([
                in_scaler_phys.transform(X_np[:, i0:i1]),
                in_scaler_pred.transform(X_np[:, i1:i2]),
                in_scaler_err.transform( X_np[:, i2:  ]),
            ]), dtype=torch.float32)

        # ── Scale, then reshape to multi-channel images ───────
        def to_cnn_input(X_flat):
            """(N, D+2*O) → (N, D+2, H, W)"""
            N_samples = X_flat.shape[0]
            phys = X_flat[:, i0:i1]                                     # (N, D)
            pred = X_flat[:, i1:i2].reshape(N_samples, 1, img_height, img_width)  # (N, 1, H, W)
            err  = X_flat[:, i2:  ].reshape(N_samples, 1, img_height, img_width)  # (N, 1, H, W)

            # Broadcast each physical param to a constant (H, W) map
            phys_maps = phys[:, :, None, None].expand(
                N_samples, n_phys, img_height, img_width
            )  # (N, D, H, W)

            return torch.cat([phys_maps, pred, err], dim=1)  # (N, D+2, H, W)

        self.train_inputs = to_cnn_input(scale_flat(train_inputs))
        self.valid_inputs = to_cnn_input(scale_flat(valid_inputs))
        self.test_inputs  = to_cnn_input(scale_flat(test_inputs))

        # ── Reshape outputs to (N, 1, H, W) ──────────────────
        def to_cnn_output(Y_flat):
            return Y_flat.reshape(-1, 1, img_height, img_width)

        self.train_outputs = to_cnn_output(train_outputs)
        self.valid_outputs = to_cnn_output(valid_outputs)
        self.test_outputs  = to_cnn_output(test_outputs)

    def train_dataloader(self):
        dataset = TensorDataset(self.train_inputs, self.train_outputs)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.rng,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = TensorDataset(self.valid_inputs, self.valid_outputs)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            generator=self.rng,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = TensorDataset(self.test_inputs, self.test_outputs)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            generator=self.rng,
            pin_memory=True,
        )


# ── Instantiate the model ────────────────────────────────────────────────────
#  Input channels: D physical params + 1 GP prediction + 1 GP uncertainty = D+2
#  Output channels: 1 (residual correction map)
input_channels  = D + 2
output_channels = 1

model = ResidualCNN(
    input_channels=input_channels,
    hidden_channels=cnn_hidden_channels,
    output_channels=output_channels,
    depth=cnn_depth,
    img_height=IMG_H,
    img_width=IMG_W,
    generator=NN_rng,
)
summary(model, input_size=(1, input_channels, IMG_H, IMG_W))


################################
### Build/Load training set ####
################################

print('BUILDING GP TRAINING SET')

# --- Define a cache path tied to the key hyperparameters ---
gp_cache_path = (
    base_dir
    + f'Model_Storage/gp_ST_cache_Nn{N_neighbor}_seed{shuffle_seed}.npz'
)
matching_files = glob.glob(base_dir + 'Model_Storage/gp_ST_cache_*.npz')

if os.path.exists(gp_cache_path):
    print(f'  Loading cached GP outputs from:\n  {gp_cache_path}')
    cache = np.load(gp_cache_path)
    GP_outputs     = cache['GP_outputs']
    GP_outputs_err = cache['GP_outputs_err']

elif matching_files:
    raise RuntimeError(
        f'WARNING: A GP cache with different hyperparameters was found:\n'
        f'  {matching_files}\n'
        f'Delete it or update your hyperparameters to match.'
    )
else:
    print(f'  No cache found. Computing GP outputs and saving to:\n  {gp_cache_path}')

    GP_outputs     = np.zeros(raw_outputs.shape, dtype=float)
    GP_outputs_err = np.zeros(raw_outputs.shape, dtype=float)

    for query_idx, (query_input, query_output) in enumerate(zip(tqdm(raw_inputs), raw_outputs)):
        XTr = np.delete(raw_inputs.T, query_idx, axis=1)
        YTr = np.delete(raw_outputs.T, query_idx, axis=1)

        Yh, err_Yh, it = ens_CGP(
            jnp.array(XTr),
            jnp.array(YTr),
            query_input,
            jnp.linalg.inv(jnp.cov(XTr)),
            N_neighbor,
        )
        GP_outputs[query_idx, :]     = Yh
        GP_outputs_err[query_idx, :] = err_Yh

    np.savez(
        gp_cache_path,
        GP_outputs=GP_outputs,
        GP_outputs_err=GP_outputs_err,
    )
    print(f'  GP outputs cached to:\n  {gp_cache_path}')

# Diagnostic plot
if show_plot:
    plot_query_output = query_output.reshape((IMG_H, IMG_W))
    plot_model_output = Yh.reshape((IMG_H, IMG_W))
    plot_error_output = err_Yh.reshape((IMG_H, IMG_W))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, layout='constrained')
    vmin = np.min(query_output)
    vmax = np.max(query_output)

    ax1.set_title('Data')
    hm1 = sns.heatmap(plot_query_output, ax=ax1)
    cbar = hm1.collections[0].colorbar
    cbar.set_label('Temperature (K)')
    ax2.set_title('Model')
    hm2 = sns.heatmap(plot_model_output, ax=ax2)
    cbar = hm2.collections[0].colorbar
    cbar.set_label('Temperature (K)')

    ax2.set_xticks(np.linspace(0, IMG_W, 5))
    ax2.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
    ax2.set_xlabel('Longitude (degrees)')

    for ax in [ax1, ax2]:
        ax.set_yticks(np.linspace(0, IMG_H, 5))
        ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
        ax.set_ylabel('Latitude (degrees)')

    plt.suptitle(
        rf'H$_2$O : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, '
        rf'LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg, '
        rf'Teff : {query_input[4]} K, Number of iterations: {it}'
    )
    plt.show()


# ── Targets are residuals: truth − GP prediction ─────────────────────────────
residuals = raw_outputs - GP_outputs  # (N, O)

# ── Split into train / valid / test ──────────────────────────────────────────
train_idx, valid_idx, test_idx = torch.utils.data.random_split(
    range(N), data_partition, generator=partition_rng
)

# --- Inputs (flat): [phys | GP_pred | GP_err] ---
NN_train_inputs_phys = torch.tensor(raw_inputs[train_idx],       dtype=torch.float32)
NN_valid_inputs_phys = torch.tensor(raw_inputs[valid_idx],       dtype=torch.float32)
NN_test_inputs_phys  = torch.tensor(raw_inputs[test_idx],        dtype=torch.float32)

NN_train_inputs_pred = torch.tensor(GP_outputs[train_idx],       dtype=torch.float32)
NN_valid_inputs_pred = torch.tensor(GP_outputs[valid_idx],       dtype=torch.float32)
NN_test_inputs_pred  = torch.tensor(GP_outputs[test_idx],        dtype=torch.float32)

NN_train_inputs_err  = torch.tensor(GP_outputs_err[train_idx],   dtype=torch.float32)
NN_valid_inputs_err  = torch.tensor(GP_outputs_err[valid_idx],   dtype=torch.float32)
NN_test_inputs_err   = torch.tensor(GP_outputs_err[test_idx],    dtype=torch.float32)

# --- Outputs (residuals) ---
NN_train_outputs = torch.tensor(residuals[train_idx], dtype=torch.float32)
NN_valid_outputs = torch.tensor(residuals[valid_idx], dtype=torch.float32)
NN_test_outputs  = torch.tensor(residuals[test_idx],  dtype=torch.float32)

# --- Plotting purposes: truth ---
NN_test_true = torch.tensor(raw_outputs[test_idx], dtype=torch.float32)

# --- Concatenate flat: [phys | GP_pred | GP_err] ---
NN_train_inputs = torch.cat([NN_train_inputs_phys, NN_train_inputs_pred, NN_train_inputs_err], dim=1)
NN_valid_inputs = torch.cat([NN_valid_inputs_phys, NN_valid_inputs_pred, NN_valid_inputs_err], dim=1)
NN_test_inputs  = torch.cat([NN_test_inputs_phys,  NN_test_inputs_pred,  NN_test_inputs_err],  dim=1)

# ── Create DataModule (handles scaling + reshape to multi-channel images) ────
data_module = CNNDataModule(
    NN_train_inputs, NN_train_outputs,
    NN_valid_inputs, NN_valid_outputs,
    NN_test_inputs,  NN_test_outputs,
    batch_size, batch_rng,
    n_phys=D,
    img_height=IMG_H,
    img_width=IMG_W,
)


###################################
#### Define optimization block ####
###################################
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

        # Base loss
        mse = self.loss_fn(pred, y)

        # Weight regularization
        l1_penalty, l2_penalty = self.compute_weight_regularization()
        loss = mse + l1_penalty + l2_penalty

        # Smoothness penalty
        if self.smoothness_coeff > 0:
            grad_loss = torch.autograd.grad(
                outputs=mse,
                inputs=X,
                create_graph=True,
                retain_graph=True,
            )[0]
            smoothness_penalty = self.smoothness_coeff * torch.mean(grad_loss ** 2)
            loss += smoothness_penalty

        self.log('train_mse',  mse,  on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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
                'monitor': 'valid_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


######################
#### Run training ####
######################
lightning_module = RegressionModule(
    model=model,
    optimizer=Adam,
    learning_rate=lr_init,
    reg_coeff_l1=regularization_coeff_l1,
    reg_coeff_l2=regularization_coeff_l2,
    weight_decay=weight_decay,
    smoothness_coeff=smoothness_coeff,
    lr_patience=lr_patience,
    lr_factor=lr_factor,
    lr_min=lr_min,
)

logger = CSVLogger(model_save_path + 'logs', name='NeuralNetwork')

pl.seed_everything(NN_seed, workers=True)

early_stopping = EarlyStopping(
    monitor='valid_loss',
    patience=early_stopping_patience,
    mode='min',
    verbose=True,
)

trainer = Trainer(
    max_epochs=n_epochs,
    logger=logger,
    deterministic=True,
    enable_checkpointing=True,
    callbacks=[
        ModelCheckpoint(
            dirpath=model_save_path,
            save_top_k=1,
            monitor='valid_loss',
            mode='min',
        ),
        early_stopping,
    ],
    enable_progress_bar=True,
)

t0 = time()

if run_mode == 'use':
    last_ckpt = None
    if os.path.exists(model_save_path + 'last.ckpt'):
        last_ckpt = model_save_path + 'last.ckpt'

    trainer.fit(lightning_module, datamodule=data_module, ckpt_path=last_ckpt)

    # Get the best checkpoint path from ModelCheckpoint callback
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")

    # Save best path for later loading
    with open(model_save_path + 'best_ckpt_path.txt', 'w') as f:
        f.write(best_model_path)

    finish_time_s    = time() - t0
    finish_time_min  = finish_time_s / 60
    finish_time_hrs  = finish_time_s / 3600
    finish_time_days = finish_time_s / (3600 * 24)
    print(f"Done! In {finish_time_s:.3f} s / {finish_time_min:.3f} min / "
          f"{finish_time_hrs:.3f} hrs / {finish_time_days:.3f} days")

else:
    with open(model_save_path + 'best_ckpt_path.txt', 'r') as f:
        best_ckpt_path = f.read().strip()

    lightning_module = RegressionModule.load_from_checkpoint(
        best_ckpt_path,
        model=model,
        optimizer=Adam,
        learning_rate=lr_init,
        reg_coeff_l1=regularization_coeff_l1,
        reg_coeff_l2=regularization_coeff_l2,
        weight_decay=weight_decay,
        smoothness_coeff=smoothness_coeff,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        lr_min=lr_min,
    )
    print("Model loaded!")

model = lightning_module.model
model.cpu()
model.eval()

if run_mode == 'use':
    trainer.test(lightning_module, datamodule=data_module)

# --- Accessing Training History ---
log_dir = model_save_path + 'logs/NeuralNetwork'
versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
latest_version = sorted(versions)[-1]
csv_path = os.path.join(log_dir, latest_version, 'metrics.csv')

metrics_df = pd.read_csv(csv_path)

train_losses = metrics_df[metrics_df['train_mse_epoch'].notna()]['train_mse_epoch'].tolist()
eval_losses  = metrics_df[metrics_df['valid_loss'].notna()]['valid_loss'].tolist()


##########################
#### Diagnostic plots ####
##########################
# ── Loss curves ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6)
)

actual_epochs = len(eval_losses)
n_batches = len(train_losses) // actual_epochs
n_batches = max(1, n_batches)

x_all   = np.linspace(0, actual_epochs, len(train_losses))
x_epoch = np.arange(actual_epochs + 1)

ax1.plot(x_all, train_losses, alpha=0.3, color='C0', linewidth=0.5)
ax1.plot(x_all, eval_losses,  alpha=0.3, color='C1', linewidth=0.5)

train_epoch = [train_losses[0]] + train_losses[n_batches - 1::n_batches]
eval_epoch  = [eval_losses[0]]  + eval_losses[n_batches - 1::n_batches]
ax1.plot(x_epoch, train_epoch, label="Train",      color='C0', linewidth=2, marker='o')
ax1.plot(x_epoch, eval_epoch,  label="Validation",  color='C1', linewidth=2, marker='o')

diff_epoch = np.abs(np.array(train_epoch) - np.array(eval_epoch))
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
plt.savefig(plot_save_path + '/loss.pdf')
plt.close()


# ── Comparing GP vs CNN vs truth ─────────────────────────────────────────────
substep = 100

out_scaler     = data_module.out_scaler
in_scaler_phys = data_module.in_scaler_phys
in_scaler_pred = data_module.in_scaler_pred
in_scaler_err  = data_module.in_scaler_err

if isinstance(NN_test_outputs, torch.Tensor):
    NN_test_outputs = NN_test_outputs.cpu().numpy()

GP_res = np.zeros(NN_test_outputs.shape, dtype=float)
NN_res = np.zeros(NN_test_outputs.shape, dtype=float)

for NN_test_idx, (NN_test_input, GP_test_output, GP_test_err,
                  NN_test_output, true) in enumerate(zip(
    NN_test_inputs_phys, NN_test_inputs_pred, NN_test_inputs_err,
    NN_test_outputs, NN_test_true
)):
    # ── Scale flat inputs ─────────────────────────────────────
    phys_np = NN_test_input.numpy().reshape(1, -1)
    pred_np = GP_test_output.numpy().reshape(1, -1)
    err_np  = GP_test_err.numpy().reshape(1, -1)

    phys_scaled = in_scaler_phys.transform(phys_np)   # (1, D)
    pred_scaled = in_scaler_pred.transform(pred_np)    # (1, O)
    err_scaled  = in_scaler_err.transform(err_np)      # (1, O)

    # ── Reshape to multi-channel CNN input (1, D+2, H, W) ────
    phys_maps = torch.tensor(
        phys_scaled, dtype=torch.float32
    )[:, :, None, None].expand(1, D, IMG_H, IMG_W)                      # (1, D, H, W)

    pred_map = torch.tensor(
        pred_scaled.reshape(1, 1, IMG_H, IMG_W), dtype=torch.float32
    )                                                                     # (1, 1, H, W)

    err_map = torch.tensor(
        err_scaled.reshape(1, 1, IMG_H, IMG_W), dtype=torch.float32
    )                                                                     # (1, 1, H, W)

    cnn_input = torch.cat([phys_maps, pred_map, err_map], dim=1)          # (1, D+2, H, W)

    # ── Predict ───────────────────────────────────────────────
    with torch.no_grad():
        cnn_pred = model(cnn_input)                                        # (1, 1, H, W)

    # Flatten to (1, O) for inverse scaling
    pred_resid_flat = cnn_pred.numpy().reshape(1, -1)
    pred_resid = out_scaler.inverse_transform(pred_resid_flat).flatten()   # (O,)

    # Final prediction = GP prediction + CNN residual correction
    CNN_pred_output = GP_test_output.numpy() + pred_resid

    # Convert to numpy
    true_np         = true.cpu().numpy()
    NN_test_input_np = NN_test_input.cpu().numpy()

    # Store residuals
    GP_res[NN_test_idx, :] = GP_test_output.numpy() - true_np
    NN_res[NN_test_idx, :] = CNN_pred_output - true_np

    # ── Plot individual maps ──────────────────────────────────
    if NN_test_idx % substep == 0:
        plot_true        = true_np.reshape((IMG_H, IMG_W))
        plot_cnn_pred    = CNN_pred_output.reshape((IMG_H, IMG_W))
        plot_gp_pred     = GP_test_output.numpy().reshape((IMG_H, IMG_W))
        plot_res_GP      = GP_res[NN_test_idx, :].reshape((IMG_H, IMG_W))
        plot_res_CNN     = NN_res[NN_test_idx, :].reshape((IMG_H, IMG_W))

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            5, 1, figsize=(8, 12), sharex=True, layout='constrained'
        )

        ax1.set_title('Data')
        hm1 = sns.heatmap(plot_true, ax=ax1)
        hm1.collections[0].colorbar.set_label('Temperature (K)')

        ax2.set_title('GP Model')
        hm2 = sns.heatmap(plot_gp_pred, ax=ax2)
        hm2.collections[0].colorbar.set_label('Temperature (K)')

        ax3.set_title('CNN Model')
        hm3 = sns.heatmap(plot_cnn_pred, ax=ax3)
        hm3.collections[0].colorbar.set_label('Temperature (K)')

        ax4.set_title('GP Residuals')
        hm4 = sns.heatmap(plot_res_GP, ax=ax4)
        hm4.collections[0].colorbar.set_label('Temperature (K)')

        ax5.set_title('CNN Residuals')
        hm5 = sns.heatmap(plot_res_CNN, ax=ax5)
        hm5.collections[0].colorbar.set_label('Temperature (K)')

        ax5.set_xticks(np.linspace(0, IMG_W, 5))
        ax5.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
        ax5.set_xlabel('Longitude (degrees)')

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_yticks(np.linspace(0, IMG_H, 5))
            ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
            ax.set_ylabel('Latitude (degrees)')

        plt.suptitle(
            rf'H$_2$ : {NN_test_input_np[0]} bar, CO$_2$ : {NN_test_input_np[1]} bar, '
            rf'LoD : {NN_test_input_np[2]:.0f} days, Obliquity : {NN_test_input_np[3]} deg, '
            rf'Teff : {NN_test_input_np[4]} K'
        )
        plt.savefig(plot_save_path + f'/pred_vs_actual_n.{NN_test_idx}.pdf')
        plt.close()


# ── Summary residual plot ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[12, 8])

for qid in range(len(NN_test_outputs)):
    ax1.plot(GP_res[qid, :], alpha=0.1, color='green')
    ax2.plot(NN_res[qid, :], alpha=0.1, color='blue')

for ax in [ax1, ax2]:
    ax.axhline(0, color='black', linestyle='dashed')
    ax.grid()

ax1.set_xlabel('Pixel Index')
ax2.set_xlabel('Pixel Index')
ax1.set_ylabel('Temperature Residual (K)')
ax2.set_ylabel('Temperature Residual (K)')
ax1.set_title('GP Residuals')
ax2.set_title('CNN Residuals')

plt.subplots_adjust(hspace=0.1, bottom=0.25)

stats_text = (
    f"--- GP Residuals ---\n"
    f"Temperature: Median = {np.median(GP_res):.2f} K, "
    f"Std = {np.std(GP_res):.2f} K, "
    f"RMSE = {np.sqrt(np.mean(GP_res**2)):.2f} K\n"
    f"\n"
    f"--- CNN Residuals ---\n"
    f"Temperature: Median = {np.median(NN_res):.2f} K, "
    f"Std = {np.std(NN_res):.2f} K, "
    f"RMSE = {np.sqrt(np.mean(NN_res**2)):.2f} K"
)

fig.text(0.1, 0.05, stats_text, fontsize=10, family='monospace',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(plot_save_path + '/res_GP_CNN.pdf', bbox_inches='tight')
plt.close()