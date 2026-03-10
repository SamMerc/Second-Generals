#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, SGD
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import scipy
from torchinfo import summary
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
from jax import jit, vmap
from functools import partial
import jax.numpy as jnp


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
model_save_path = base_dir+'Model_Storage/updatedensCGP_deeperMLP/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/updatedensCGP_deeperMLP/'
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

## HYPER-PARAMETERS for ens-CGP ##

#Number of nearest neighbors to choose
N_neighbor = 4

#Distance metric to use
distance_metric = 'euclidean' #options: 'euclidean', 'mahalanobis', 'logged_euclidean', 'logged_mahalanobis'

#Convert raw inputs for H2 and CO2 pressures to log10 scale so don't have to deal with it later
if 'logged' in distance_metric:
    raw_inputs[:, 0] = np.log10(raw_inputs[:, 0]) #H2
    raw_inputs[:, 1] = np.log10(raw_inputs[:, 1]) #CO2


## HYPER-PARAMETERS for NN ##
#Definine partitiion for splitting NN dataset
data_partition = [0.7, 0.1, 0.2]

#Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 96
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")

#Defining the noise seed for the random partitioning of the training data
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

#Neural network width and depth
nn_width = 102
nn_depth = 10

#Optimizer learning rate
learning_rate = 1e-3

#Regularization coefficient
regularization_coeff_l1 = 0.0
regularization_coeff_l2 = 0.0

#Smoothness constraint coefficient
smoothness_coeff = 0.001

#Weight decay 
weight_decay = 0.0

#Batch size 
batch_size = 200

#Number of epochs 
n_epochs = 10000

#Mode for optimization
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






###################
#### Build MLP ####
###################
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, generator=None):
        super().__init__()
        # Set seed if generator provided
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        # Pack all layers into a Sequential container
        self.linear_relu_stack = nn.Sequential(*layers)
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
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
        
        # Standardize the input
        ## Create scaler
        in_scaler_T = StandardScaler()
        in_scaler_P = StandardScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        in_scaler_T.fit(train_inputs[:, :O].cpu().numpy())
        in_scaler_P.fit(train_inputs[:, O:].cpu().numpy())

        ## Transform all datasets and convert back to tensors
        train_T_scaled = torch.tensor(in_scaler_T.transform(train_inputs[:, :O].cpu().numpy()), dtype=torch.float32)
        train_P_scaled = torch.tensor(in_scaler_P.transform(train_inputs[:, O:].cpu().numpy()), dtype=torch.float32)

        valid_T_scaled = torch.tensor(in_scaler_T.transform(valid_inputs[:, :O].cpu().numpy()), dtype=torch.float32)
        valid_P_scaled = torch.tensor(in_scaler_P.transform(valid_inputs[:, O:].cpu().numpy()), dtype=torch.float32)

        test_T_scaled = torch.tensor(in_scaler_T.transform(test_inputs[:, :O].cpu().numpy()), dtype=torch.float32)
        test_P_scaled = torch.tensor(in_scaler_P.transform(test_inputs[:, O:].cpu().numpy()), dtype=torch.float32)
        
        # Concatenate
        train_inputs = torch.cat([train_T_scaled, train_P_scaled], dim=1)
        valid_inputs = torch.cat([valid_T_scaled, valid_P_scaled], dim=1)
        test_inputs = torch.cat([test_T_scaled, test_P_scaled], dim=1)

        # Store the scaler if you need to inverse transform later
        self.in_scaler_T = in_scaler_T
        self.in_scaler_P = in_scaler_P

        # Storing it and passing it to loaders
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.valid_inputs = valid_inputs
        self.valid_outputs = valid_outputs
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs
        self.batch_size = batch_size
        self.rng = rng
    
    def train_dataloader(self):
        dataset = TensorDataset(self.train_inputs, self.train_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=self.rng)
    
    def val_dataloader(self):
        dataset = TensorDataset(self.valid_inputs, self.valid_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, generator=self.rng)

    def test_dataloader(self):
        dataset = TensorDataset(self.test_inputs, self.test_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, generator=self.rng)

model = NeuralNetwork(2*O, nn_width, 2*O, nn_depth, generator=NN_rng)
summary(model)




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

# argets are residuals: truth - GP prediction
residuals_T = raw_outputs_T - GP_outputs_T   # (N, O)
residuals_P = raw_outputs_P - GP_outputs_P   # (N, O)

# Split training dataset into training, validation, and testing, and format it correctly

## Retrieving indices of data partitions
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(N), data_partition, generator=partition_rng)

## Generate the data partitions
### Training
NN_train_inputs_T = torch.tensor(GP_outputs_T[train_idx], dtype=torch.float32)
NN_train_inputs_P = torch.tensor(GP_outputs_P[train_idx], dtype=torch.float32)
NN_train_outputs_T = torch.tensor(residuals_T[train_idx], dtype=torch.float32)
NN_train_outputs_P = torch.tensor(residuals_P[train_idx], dtype=torch.float32)
### Validation
NN_valid_inputs_T = torch.tensor(GP_outputs_T[valid_idx], dtype=torch.float32)
NN_valid_inputs_P = torch.tensor(GP_outputs_P[valid_idx], dtype=torch.float32)
NN_valid_outputs_T = torch.tensor(residuals_T[valid_idx], dtype=torch.float32)
NN_valid_outputs_P = torch.tensor(residuals_P[valid_idx], dtype=torch.float32)
### Testing
NN_test_inputs_T = torch.tensor(GP_outputs_T[test_idx], dtype=torch.float32)
NN_test_inputs_P = torch.tensor(GP_outputs_P[test_idx], dtype=torch.float32)
NN_test_outputs_T = torch.tensor(residuals_T[test_idx], dtype=torch.float32)
NN_test_outputs_P = torch.tensor(residuals_P[test_idx], dtype=torch.float32)
NN_test_true_T = torch.tensor(raw_outputs_T[test_idx], dtype=torch.float32)
NN_test_true_P = torch.tensor(raw_outputs_P[test_idx], dtype=torch.float32)
NN_test_og_inputs = torch.tensor(raw_inputs[test_idx], dtype=torch.float32) 

## Concatenating inputs and outputs
NN_train_inputs = torch.cat([
    NN_train_inputs_T,
    NN_train_inputs_P
], dim=1)
NN_train_outputs = torch.cat([
    NN_train_outputs_T,
    NN_train_outputs_P
], dim=1)

NN_valid_inputs = torch.cat([
    NN_valid_inputs_T,
    NN_valid_inputs_P
], dim=1)
NN_valid_outputs = torch.cat([
    NN_valid_outputs_T,
    NN_valid_outputs_P
], dim=1)

NN_test_inputs = torch.cat([
    NN_test_inputs_T,
    NN_test_inputs_P
], dim=1)
NN_test_outputs = torch.cat([
    NN_test_outputs_T,
    NN_test_outputs_P
], dim=1)

# Create DataModule
data_module = CustomDataModule(
    NN_train_inputs, NN_train_outputs,
    NN_valid_inputs, NN_valid_outputs,
    NN_test_inputs, NN_test_outputs,
    batch_size, batch_rng
)






###################################
#### Define optimization block ####
###################################
# PyTorch Lightning Module
class RegressionModule(pl.LightningModule):
    def __init__(self, model, optimizer, learning_rate, weight_decay=0.0, reg_coeff_l1=0.0, reg_coeff_l2=0.0, smoothness_coeff=0.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.reg_coeff_l1 = reg_coeff_l1
        self.reg_coeff_l2 = reg_coeff_l2
        self.smoothness_coeff = smoothness_coeff
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()
        self.optimizer_class = optimizer
    
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
    
    def compute_smoothness_constraint(self, X, output):
        """
        Compute smoothness constraint: ||∇s|| where s is the model output.
        This penalizes rapid changes in output with respect to input.
        """
        if self.smoothness_coeff == 0:
            return torch.tensor(0., device=self.device)
        
        # Clone and enable gradient computation for inputs
        X_grad = X.clone().detach().requires_grad_(True)

        # Temporarily enable gradients
        with torch.enable_grad():
            # Recompute output with gradient tracking
            output_grad = self.model(X_grad)
            
            # Compute gradients of output with respect to input: ∂s/∂x
            grad_outputs = torch.ones_like(output_grad)
            gradients = torch.autograd.grad(
                outputs=output_grad,
                inputs=X_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Smoothness: ||∇s|| = L2 norm of gradient vector for each sample
            # Shape: gradients is (batch_size, input_dim)
            # We want: mean over batch of ||∇s|| for each sample
            smoothness_penalty = torch.mean(torch.norm(gradients, p=2, dim=1))
        
        return self.smoothness_coeff * smoothness_penalty

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        X, y = batch
        pred = self(X)
        
        # Base loss: ||y - s||
        loss = self.loss_fn(pred, y)
        
        # Add weight regularization (L1/L2 on network parameters)
        l1_penalty, l2_penalty = self.compute_weight_regularization()
        loss += l1_penalty + l2_penalty
        
        # Add smoothness constraint (gradient of output w.r.t. input)
        smoothness_penalty = self.compute_smoothness_constraint(X, pred)
        loss += smoothness_penalty

        # Log metrics
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
        return self.optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)








######################
#### Run training ####
######################
# Create Lightning Module
lightning_module = RegressionModule(
    model=model,
    optimizer=Adam,
    learning_rate=learning_rate,
    reg_coeff_l1=regularization_coeff_l1,
    reg_coeff_l2=regularization_coeff_l2,
    weight_decay=weight_decay,
    smoothness_coeff=smoothness_coeff,
)

# Setup logger
logger = CSVLogger(model_save_path+'logs', name='NeuralNetwork')

# Set all seeds for complete reproducibility
pl.seed_everything(NN_seed, workers=True)

# Create Trainer and train
trainer = Trainer(
    max_epochs=n_epochs,
    logger=logger,
    deterministic=True  # For reproducibility
)

if run_mode == 'use':
    
    trainer.fit(lightning_module, datamodule=data_module)
    
    # Save model (PyTorch Lightning style)
    trainer.save_checkpoint(model_save_path + f'{n_epochs}epochs_{weight_decay}WD_{regularization_coeff_l1+regularization_coeff_l2}RC_{smoothness_coeff}SC_{learning_rate}LR_{batch_size}BS.ckpt')
    
    print("Done!")
    
else:
    # Load model
    lightning_module = RegressionModule.load_from_checkpoint(
        model_save_path + f'{n_epochs}epochs_{weight_decay}WD_{regularization_coeff_l1+regularization_coeff_l2}RC_{smoothness_coeff}SC_{learning_rate}LR_{batch_size}BS.ckpt',
        model=model,
        optimizer=Adam,
    learning_rate=learning_rate,
    reg_coeff_l1=regularization_coeff_l1,
    reg_coeff_l2=regularization_coeff_l2,
    weight_decay=weight_decay,
    smoothness_coeff=smoothness_coeff,
    )
    print("Model loaded!")


#Testing model on test dataset
if run_mode == 'use':trainer.test(lightning_module, datamodule=data_module)

# --- Accessing Training History After Training ---
# Find the version directory (e.g., version_0, version_1, etc.)
log_dir = model_save_path+'logs/NeuralNetwork'
versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
latest_version = sorted(versions)[-1]  # Get the latest version
csv_path = os.path.join(log_dir, latest_version, 'metrics.csv')

# Read the metrics
metrics_df = pd.read_csv(csv_path)

# Extract losses per epoch
train_losses = metrics_df[metrics_df['train_loss_epoch'].notna()]['train_loss_epoch'].tolist()
eval_losses = metrics_df[metrics_df['valid_loss'].notna()]['valid_loss'].tolist()







##########################
#### Diagnostic plots ####
##########################
# Loss curves
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3, 1]}, figsize=(10, 6))

# Calculate number of batches per epoch
n_batches = len(train_losses) // n_epochs

# Create x-axis in terms of epochs (0 to n_epochs)
x_all = np.linspace(0, n_epochs, len(train_losses))
x_epoch = np.arange(n_epochs+1)

# Plot transparent background showing all batch losses
ax1.plot(x_all, train_losses, alpha=0.3, color='C0', linewidth=0.5)
ax1.plot(x_all, eval_losses, alpha=0.3, color='C1', linewidth=0.5)

# Plot solid lines showing epoch-level losses (every n_batches steps)
train_epoch = [train_losses[0]] + train_losses[n_batches-1::n_batches]  # Last batch of each epoch
eval_epoch = [eval_losses[0]] + eval_losses[n_batches-1::n_batches]
ax1.plot(x_epoch, train_epoch, label="Train", color='C0', linewidth=2, marker='o')
ax1.plot(x_epoch, eval_epoch, label="Validation", color='C1', linewidth=2, marker='o')

# Same for difference plot
diff_all = np.array(train_losses) - np.array(eval_losses)
diff_epoch = np.array(train_epoch) - np.array(eval_epoch)

ax2.plot(x_all, diff_all, alpha=0.3, color='C2', linewidth=0.5)
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

#Comparing GP predicted T-P profiles vs NN predicted T-P profiles vs true T-P profiles with residuals
substep = 100

# Get the scalers from data module
out_scaler_T = data_module.out_scaler_T
out_scaler_P = data_module.out_scaler_P
in_scaler_T = data_module.in_scaler_T
in_scaler_P = data_module.in_scaler_P

# Move model to CPU for inference to avoid GPU memory issues
model.cpu()
model.eval()

#Converting tensors to numpy arrays if this isn't already done
if (type(NN_test_outputs_T) != np.ndarray):
    NN_test_outputs_T = NN_test_outputs_T.cpu().numpy()
    NN_test_outputs_P = NN_test_outputs_P.cpu().numpy()

GP_res_T = np.zeros(NN_test_outputs_P.shape, dtype=float)
GP_res_P = np.zeros(NN_test_outputs_P.shape, dtype=float)
NN_res_T = np.zeros(NN_test_outputs_P.shape, dtype=float)
NN_res_P = np.zeros(NN_test_outputs_P.shape, dtype=float)

for NN_test_idx, (NN_test_input, GP_test_output_T, GP_test_output_P, 
                  NN_test_output_T, NN_test_output_P,
                  true_T, true_P) in enumerate(zip(
    NN_test_og_inputs,
    NN_test_inputs_T, NN_test_inputs_P,
    NN_test_outputs_T, NN_test_outputs_P,   # these are now residuals
    NN_test_true_T, NN_test_true_P          # these are the actual profiles
)):
    #Retrieve prediction
    scaled_GP_test_output_T = torch.tensor(in_scaler_T.transform(GP_test_output_T.reshape(1, -1)), dtype=torch.float32)
    scaled_GP_test_output_P = torch.tensor(in_scaler_P.transform(GP_test_output_P.reshape(1, -1)), dtype=torch.float32)
    scaled_input = torch.cat([scaled_GP_test_output_T, scaled_GP_test_output_P], dim=1)

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

        plt.suptitle(rf'H$_2$ : {NN_test_input[0]} bar, CO$_2$ : {NN_test_input[1]} bar, LoD : {NN_test_input[2]:.0f} days, Obliquity : {NN_test_input[3]} deg')
        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{NN_test_idx}.pdf')  


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
plt.show()