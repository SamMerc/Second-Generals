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
import shutil
torch.set_float32_matmul_precision('high')
import optuna
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
import warnings
warnings.filterwarnings('ignore')
import matplotlib.colors as mcolors
from matplotlib import ticker
from sklearn.metrics import r2_score
import matplotlib.gridspec as gridspec

##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
def check_and_make_dir(dir):
    if not os.path.isdir(dir): os.mkdir(dir)

base_dir = '/home/merci228/WORK/2G_ML/'
raw_T_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_T.csv', delimiter=',')
raw_T_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
raw_P_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_P.csv', delimiter=',')
raw_P_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')

model_save_path = base_dir+'Model_Storage/Hyperparam_tuning_LRinit_NNdepth_NNwidth_L2_BS/'
check_and_make_dir(model_save_path)
plot_save_path = base_dir+'Plots/Hyperparam_tuning_LRinit_NNdepth_NNwidth_L2_BS/'
check_and_make_dir(plot_save_path)

inputs_3000 = np.hstack([raw_T_data3000[:, :4], np.full((len(raw_T_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_T_data4500[:, :4], np.full((len(raw_T_data4500), 1), 4500.0)])

raw_inputs    = np.vstack([inputs_3000,           inputs_4500          ])
raw_outputs_T = np.vstack([raw_T_data3000[:, 5:], raw_T_data4500[:, 5:]])
raw_outputs_P = np.vstack([raw_P_data3000[:, 5:], raw_P_data4500[:, 5:]])
raw_outputs_P = np.log10(raw_outputs_P / 1000)

N = raw_inputs.shape[0]
D = raw_inputs.shape[1]
O = raw_outputs_T.shape[1]

shuffle_seed = 3
np.random.seed(shuffle_seed)
rp = np.random.permutation(N)
raw_inputs    = raw_inputs[rp, :]
raw_outputs_T = raw_outputs_T[rp, :]
raw_outputs_P = raw_outputs_P[rp, :]

N_neighbor     = 4
data_partition = [0.7, 0.1, 0.2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 96
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")

#############################
#### run_mode selection  ####
#############################
run_mode = 'evaluate'   # 'search1' | 'search2' | 'train' | 'evaluate'

## ── Parameters for 'train' and 'evaluate' modes ──────────────────────────────
## After the Optuna search completes, paste the best params here and switch
## run_mode to 'train', then to 'evaluate'.
FINAL_PARAMS = {
    'lr_init'    : 0.0001881473773986,
    'nn_depth'   : 8,
    'nn_width'   : 313,
    'reg_l2'     : 2.175134924658512e-06,
    'batch_size' : 128,
}
FINAL_PARTITION_SEED = 4
FINAL_BATCH_SEED     = 5
FINAL_NN_SEED        = 6
FINAL_LR_PATIENCE    = 50
FINAL_LR_FACTOR      = 0.7
FINAL_LR_MIN         = 1e-7
FINAL_N_EPOCHS       = 5000
FINAL_ES_PATIENCE    = 200

###############################################
#### Ensemble Conditional Gaussian Process ####
###############################################
@partial(jit, static_argnames=('k',))
def _mahal_knn_single(X_train, xq, VI, k):
    diff = X_train - xq[:, None]
    dists_sq = jnp.sum(diff * (VI @ diff), axis=0)
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
    Xens_NN = Xens[:, idxs]
    Yens_NN = Yens[:, idxs]
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
    idxs2    = _mahal_knn_batch(Xens, XhSel, VI, 1).flatten()
    idxs_new = jnp.unique(idxs2, size=N_neighbor, fill_value=-1)
    idxs_topup = _mahal_knn_single(Xens, Xq.ravel(), VI, N_neighbor)
    idxs_final = jnp.where(idxs_new >= 0, idxs_new, idxs_topup)
    Yh     = Ym + Mf @ (Xq - Xm)
    cov_Yh = Cyy - Mf @ Cxy
    return idxs_final, Mf, Cxy, Xm, Ym, Yh, cov_Yh

def ens_CGP(Xens_j, Yens_j, Xq, VI_j, N_neighbor, tol=1e-6, max_iter=1000):
    Xq_j = jnp.array(Xq.ravel())
    idxs = _mahal_knn_single(Xens_j, Xq_j, VI_j, N_neighbor)
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
        rel_change = np.mean(np.abs(Yh - Yh_prev) / (np.abs(Yh_prev) + 1e-10))
        if rel_change < tol:
            break
        n_repeats = np.sum(np.isclose(rel_change_history, rel_change, rtol=1e-3))
        if n_repeats >= 5:
            break
        rel_change_history.append(rel_change)
        Yh_prev = Yh
    err_Yh = jnp.sqrt(jnp.maximum(0.0, jnp.diag(cov_Yh)))
    return Yh, np.array(err_Yh), i + 2

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
        return self.activation(x + self.block(x))


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, generator=None):
        super().__init__()
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs,
                 test_inputs, test_outputs, batch_size, rng):
        super().__init__()

        out_scaler_T = StandardScaler()
        out_scaler_P = StandardScaler()
        out_scaler_T.fit(train_outputs[:, :O].cpu().numpy())
        out_scaler_P.fit(train_outputs[:, O:].cpu().numpy())

        def scale_outputs(t):
            return torch.cat([
                torch.tensor(out_scaler_T.transform(t[:, :O].cpu().numpy()), dtype=torch.float32),
                torch.tensor(out_scaler_P.transform(t[:, O:].cpu().numpy()), dtype=torch.float32),
            ], dim=1)

        self.out_scaler_T = out_scaler_T
        self.out_scaler_P = out_scaler_P

        # Input scalers — one per block
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
        self.train_outputs = scale_outputs(train_outputs)
        self.valid_outputs = scale_outputs(valid_outputs)
        self.test_outputs  = scale_outputs(test_outputs)

        self.in_scaler_phys = in_scaler_phys
        self.in_scaler_T    = in_scaler_T
        self.in_scaler_P    = in_scaler_P
        self.in_scaler_Terr = in_scaler_Terr
        self.in_scaler_Perr = in_scaler_Perr

        self.batch_size = batch_size
        self.rng = rng

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.train_inputs, self.train_outputs),
                          batch_size=self.batch_size, shuffle=True,
                          generator=self.rng, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.valid_inputs, self.valid_outputs),
                          batch_size=self.batch_size, generator=self.rng,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.test_inputs, self.test_outputs),
                          batch_size=self.batch_size, generator=self.rng,
                          pin_memory=True)


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
        mse = self.loss_fn(pred, y)
        l1_penalty, l2_penalty = self.compute_weight_regularization()
        loss = mse + l1_penalty + l2_penalty
        if self.smoothness_coeff > 0:
            grad_mse = torch.autograd.grad(         # ← differentiate mse, not loss
                outputs=mse,
                inputs=X,
                create_graph=True,
                retain_graph=True,
            )[0]
            smoothness_penalty = self.smoothness_coeff * torch.mean(grad_mse ** 2)
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
            optimizer, mode='min', factor=self.lr_factor,
            patience=self.lr_patience, min_lr=self.lr_min,
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


################################
### Build/Load GP cache ########
################################
print('BUILDING GP TRAINING SET')

gp_cache_path  = base_dir + f'Model_Storage/gp_cache_Nn{N_neighbor}_seed{shuffle_seed}.npz'
matching_files = glob.glob(base_dir + 'Model_Storage/gp_cache_*.npz')

if os.path.exists(gp_cache_path):
    print(f'  Loading cached GP outputs from:\n  {gp_cache_path}')
    cache = np.load(gp_cache_path)
    GP_outputs_T    = cache['GP_outputs_T']
    GP_outputs_P    = cache['GP_outputs_P']
    GP_outputs_Terr = cache['GP_outputs_Terr']
    GP_outputs_Perr = cache['GP_outputs_Perr']

elif matching_files:
    raise RuntimeError(
        f'WARNING: A GP cache with different hyperparameters was found:\n'
        f'  {matching_files}\n'
        f'Delete it or update your hyperparameters to match.'
    )
else:
    print(f'  No cache found. Computing GP outputs and saving to:\n  {gp_cache_path}')
    GP_outputs_T    = np.zeros(raw_outputs_T.shape, dtype=float)
    GP_outputs_P    = np.zeros(raw_outputs_P.shape, dtype=float)
    GP_outputs_Terr = np.zeros(raw_outputs_T.shape, dtype=float)
    GP_outputs_Perr = np.zeros(raw_outputs_P.shape, dtype=float)

    for query_idx, (query_input, query_output_T, query_output_P) in enumerate(
            zip(tqdm(raw_inputs), raw_outputs_T, raw_outputs_P)):
        XTr = np.delete(raw_inputs.T, query_idx, axis=1)
        YTr = np.delete(np.hstack([raw_outputs_T, raw_outputs_P]).T, query_idx, axis=1)
        Xens_j = jnp.array(XTr)
        Yens_j = jnp.array(YTr)
        VI_j   = jnp.linalg.inv(jnp.cov(Xens_j))
        Yh, Yh_err, it = ens_CGP(Xens_j, Yens_j, query_input, VI_j, N_neighbor)
        GP_outputs_T[query_idx, :]    = Yh[:O]
        GP_outputs_Terr[query_idx, :] = Yh_err[:O]
        GP_outputs_P[query_idx, :]    = Yh[O:]
        GP_outputs_Perr[query_idx, :] = Yh_err[O:]

    np.savez(gp_cache_path,
             GP_outputs_T=GP_outputs_T, GP_outputs_P=GP_outputs_P,
             GP_outputs_Terr=GP_outputs_Terr, GP_outputs_Perr=GP_outputs_Perr)
    print(f'  GP outputs cached to:\n  {gp_cache_path}')

# Residuals: what the NN needs to learn
residuals_T = raw_outputs_T - GP_outputs_T
residuals_P = raw_outputs_P - GP_outputs_P


###############################
#### Shared helper function ####
###############################
def build_data_module(partition_seed, batch_size, batch_seed=5):
    """Build a CustomDataModule for a given partition seed and batch size."""
    _partition_rng = torch.Generator()
    _partition_rng.manual_seed(partition_seed)
    _batch_rng = torch.Generator()
    _batch_rng.manual_seed(batch_seed)

    _train_idx, _valid_idx, _test_idx = torch.utils.data.random_split(
        range(N), data_partition, generator=_partition_rng
    )

    def make_inputs(idx):
        return torch.cat([
            torch.tensor(raw_inputs[idx],        dtype=torch.float32),
            torch.tensor(GP_outputs_T[idx],      dtype=torch.float32),
            torch.tensor(GP_outputs_P[idx],      dtype=torch.float32),
            torch.tensor(GP_outputs_Terr[idx],   dtype=torch.float32),
            torch.tensor(GP_outputs_Perr[idx],   dtype=torch.float32),
        ], dim=1)

    def make_outputs(idx):
        return torch.cat([
            torch.tensor(residuals_T[idx], dtype=torch.float32),
            torch.tensor(residuals_P[idx], dtype=torch.float32),
        ], dim=1)

    return (
        CustomDataModule(
            make_inputs(_train_idx),  make_outputs(_train_idx),
            make_inputs(_valid_idx),  make_outputs(_valid_idx),
            make_inputs(_test_idx),   make_outputs(_test_idx),
            batch_size, _batch_rng,
        ),
        _test_idx,   # returned so evaluate mode can index raw arrays
    )


##################################
#### run_mode: search1 ############
##################################
if run_mode == 'search1':

    PARTITION_SEEDS = [4]           # Stage 1: single seed

    SEARCH_BATCH_SEED  = 5
    SEARCH_NN_SEED     = 6
    SEARCH_LR_PATIENCE = 50
    SEARCH_LR_FACTOR   = 0.7
    SEARCH_LR_MIN      = 1e-7
    SEARCH_N_EPOCHS    = 5000
    SEARCH_ES_PATIENCE = 100   # tighter patience for faster search

    def print_trial_summary(study, trial):
        if trial.value is None:
            return   # pruned trial — skip
        print(f'\n--- Trial {trial.number} finished ---')
        print(f'  Value (mean val loss): {trial.value:.6f}')
        print(f'  Params: {trial.params}')
        print(f'  Best so far: {study.best_value:.6f} (trial {study.best_trial.number})')
        print(f'  Trials completed: {len([t for t in study.trials if t.value is not None])}')

    def objective(trial):
        lr_init    = trial.suggest_float('lr_init',  1e-4, 1e-2, log=True)
        nn_depth   = trial.suggest_categorical('nn_depth',   [4, 8, 16, 32])
        nn_width   = trial.suggest_categorical('nn_width',   [209, 313, 418])
        reg_l2     = trial.suggest_float('reg_l2',   1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

        val_losses = []

        for p_seed in PARTITION_SEEDS:

            _data_module, _ = build_data_module(p_seed, batch_size, SEARCH_BATCH_SEED)

            pl.seed_everything(SEARCH_NN_SEED, workers=True)
            _model = NeuralNetwork(D + 4*O, nn_width, 2*O, nn_depth)

            _lightning_module = RegressionModule(
                model=_model,
                optimizer=Adam,
                learning_rate=lr_init,
                reg_coeff_l1=0.0,
                reg_coeff_l2=reg_l2,
                weight_decay=0.0,
                smoothness_coeff=0.0,
                lr_patience=SEARCH_LR_PATIENCE,
                lr_factor=SEARCH_LR_FACTOR,
                lr_min=SEARCH_LR_MIN,
            )

            pruning_callback = PyTorchLightningPruningCallback(trial, monitor='valid_loss')
            early_stopping   = EarlyStopping(monitor='valid_loss', patience=SEARCH_ES_PATIENCE, mode='min')
            checkpoint       = ModelCheckpoint(
                monitor='valid_loss', mode='min', save_top_k=1,
                dirpath=model_save_path + f'optuna_trial{trial.number}_seed{p_seed}/',
            )

            _trainer = Trainer(
                max_epochs=SEARCH_N_EPOCHS,
                callbacks=[pruning_callback, early_stopping, checkpoint],
                enable_progress_bar=False,
                logger=False,
                deterministic=True,
                enable_checkpointing=True,
            )

            try:
                _trainer.fit(_lightning_module, datamodule=_data_module)
            except optuna.exceptions.TrialPruned:
                raise

            best_val = checkpoint.best_model_score
            if best_val is None:
                raise optuna.exceptions.TrialPruned()

            val_losses.append(best_val.item())

        return float(np.mean(val_losses))

    # ── Run study ────────────────────────────────────────────────────────────
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=50,
        interval_steps=10,
    )

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        storage=f'sqlite:///{model_save_path}_stage1.db',
        study_name='tp_profile_nn_stage1',
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=50,
        timeout=None,
        gc_after_trial=True,
        callbacks=[print_trial_summary],
    )

    # ── Report ───────────────────────────────────────────────────────────────
    print('\n=== Optuna Search Complete ===')
    print(f'Best trial   : {study.best_trial.number}')
    print(f'Best val loss: {study.best_value:.6f}')
    print('Best hyperparameters:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')

    results_df = study.trials_dataframe()
    results_df.to_csv(model_save_path + 'optuna_results_stage1.csv', index=False)
    print(f'\nFull results saved to {model_save_path}optuna_results_stage1.csv')

    print('\nTop 10 trials:')
    top10 = results_df.sort_values('value').head(10)[[
        'number', 'value', 'params_lr_init', 'params_nn_depth',
        'params_nn_width', 'params_reg_l2', 'params_batch_size'
    ]]
    print(top10.to_string(index=False))

    # ── Clean up checkpoints from non-best trials ────────────────────────────
    best_trial_number = study.best_trial.number
    for trial in study.trials:
        if trial.number == best_trial_number:
            continue
        for p_seed in PARTITION_SEEDS:
            trial_dir = model_save_path + f'optuna_trial{trial.number}_seed{p_seed}/'
            if os.path.exists(trial_dir):
                shutil.rmtree(trial_dir)
    print(f'\nCheckpoints from non-best trials removed. Best trial ({best_trial_number}) kept.')

##################################
#### run_mode: search2 ###########
##################################
elif run_mode == 'search2':

    STAGE2_PARTITION_SEEDS = [4, 7, 13, 42, 99]  # 5 seeds for robustness evaluation
    STAGE2_BATCH_SEED      = 5
    STAGE2_NN_SEED         = 6
    STAGE2_LR_PATIENCE     = 50
    STAGE2_LR_FACTOR       = 0.7
    STAGE2_LR_MIN          = 1e-7
    STAGE2_N_EPOCHS        = 5000
    STAGE2_ES_PATIENCE     = 200   # relaxed vs Stage 1 for fairer comparison
    #Stage 1 used a tighter patience to speed up the search. 
    #Stage 2 uses your intended final patience so the comparison is fair and reflects how the model
    #will actually behave in train mode.
    STAGE2_TOP_N           = 10

    # ── Load Stage 1 results and extract top N configs ────────────────────────
    stage1_csv = model_save_path + 'optuna_results_stage1.csv'
    assert os.path.exists(stage1_csv), f'Stage 1 results not found at {stage1_csv}'

    results_df = pd.read_csv(stage1_csv)

    # Keep only completed (non-pruned) trials
    completed = results_df[results_df['value'].notna()].copy()
    top_configs = completed.sort_values('value').head(STAGE2_TOP_N).reset_index(drop=True)

    print(f'\n=== Stage 2: Robustness evaluation of top {STAGE2_TOP_N} configs ===')
    print(f'Partition seeds: {STAGE2_PARTITION_SEEDS}')
    print(top_configs[['number', 'value', 'params_lr_init', 'params_nn_depth',
                        'params_nn_width', 'params_reg_l2', 'params_batch_size']].to_string(index=False))

    stage2_results = []

    #Optuna logging
    stage2_study = optuna.create_study(
        direction='minimize',
        storage=f'sqlite:///{model_save_path}optuna_study_stage2.db',
        study_name='tp_profile_nn_stage2',
        load_if_exists=True,
    )

    for rank, row in top_configs.iterrows():
        trial_num  = int(row['number'])
        lr_init    = float(row['params_lr_init'])
        nn_depth   = int(row['params_nn_depth'])
        nn_width   = int(row['params_nn_width'])
        reg_l2     = float(row['params_reg_l2'])
        batch_size = int(row['params_batch_size'])

        print(f'\n--- Stage 2 | Rank {rank+1} | Stage-1 trial {trial_num} ---')
        print(f'  lr={lr_init:.2e}, depth={nn_depth}, width={nn_width}, '
              f'l2={reg_l2:.2e}, bs={batch_size}')

        seed_val_losses = []

        for p_seed in STAGE2_PARTITION_SEEDS:

            data_module, _ = build_data_module(p_seed, batch_size, STAGE2_BATCH_SEED)

            pl.seed_everything(STAGE2_NN_SEED, workers=True)
            _model = NeuralNetwork(D + 4*O, nn_width, 2*O, nn_depth)

            _lightning_module = RegressionModule(
                model=_model,
                optimizer=Adam,
                learning_rate=lr_init,
                reg_coeff_l1=0.0,
                reg_coeff_l2=reg_l2,
                weight_decay=0.0,
                smoothness_coeff=0.0,
                lr_patience=STAGE2_LR_PATIENCE,
                lr_factor=STAGE2_LR_FACTOR,
                lr_min=STAGE2_LR_MIN,
            )

            ckpt_dir = model_save_path + f'stage2_trial{trial_num}_seed{p_seed}/'
            checkpoint_cb = ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor='valid_loss', mode='min', save_top_k=1,
            )
            early_stopping = EarlyStopping(
                monitor='valid_loss', patience=STAGE2_ES_PATIENCE, mode='min',
            )

            trainer = Trainer(
                max_epochs=STAGE2_N_EPOCHS,
                callbacks=[checkpoint_cb, early_stopping],
                enable_progress_bar=False,
                logger=False,
                deterministic=True,
                enable_checkpointing=True,
            )

            trainer.fit(_lightning_module, datamodule=data_module)

            best_val = checkpoint_cb.best_model_score
            val_loss = best_val.item() if best_val is not None else float('nan')
            seed_val_losses.append(val_loss)
            print(f'  seed={p_seed}  ->  val_loss={val_loss:.6f}')

        mean_loss   = float(np.nanmean(seed_val_losses))
        std_loss    = float(np.nanstd(seed_val_losses))
        worst_loss  = float(np.nanmax(seed_val_losses))

        # After the per-seed loop for each config:
        trial = stage2_study.ask()
        trial.suggest_float('lr_init',  lr_init,    lr_init)
        trial.suggest_int(  'nn_depth', nn_depth,   nn_depth)
        trial.suggest_int(  'nn_width', nn_width,   nn_width)
        trial.suggest_float('reg_l2',   reg_l2,     reg_l2)
        trial.suggest_int(  'batch_size', batch_size, batch_size)
        stage2_study.tell(trial, mean_loss)

        print(f'  => mean={mean_loss:.6f}  std={std_loss:.6f}  worst={worst_loss:.6f}')

        stage2_results.append({
            'stage1_trial'    : trial_num,
            'stage1_val_loss' : float(row['value']),
            'mean_val_loss'   : mean_loss,
            'std_val_loss'    : std_loss,
            'worst_val_loss'  : worst_loss,
            'lr_init'         : lr_init,
            'nn_depth'        : nn_depth,
            'nn_width'        : nn_width,
            'reg_l2'          : reg_l2,
            'batch_size'      : batch_size,
            'per_seed_losses' : seed_val_losses,
        })

    # ── Rank by mean val loss ─────────────────────────────────────────────────
    stage2_df = pd.DataFrame(stage2_results).sort_values('mean_val_loss').reset_index(drop=True)
    stage2_df.to_csv(model_save_path + 'stage2_results.csv', index=False)

    print('\n=== Stage 2 Complete ===')
    print('Rankings by mean val loss across seeds:')
    print(stage2_df[['stage1_trial', 'stage1_val_loss', 'mean_val_loss',
                      'std_val_loss', 'worst_val_loss', 'lr_init', 'nn_depth',
                      'nn_width', 'reg_l2', 'batch_size']].to_string(index=False))

    best = stage2_df.iloc[0]
    print(f'\nBest Stage-2 config (Stage-1 trial {int(best["stage1_trial"])}) :')
    print(f'  mean_val_loss = {best["mean_val_loss"]:.6f}  (std={best["std_val_loss"]:.6f})')
    print(f'  lr_init    : {best["lr_init"]}')
    print(f'  nn_depth   : {int(best["nn_depth"])}')
    print(f'  nn_width   : {int(best["nn_width"])}')
    print(f'  reg_l2     : {best["reg_l2"]}')
    print(f'  batch_size : {int(best["batch_size"])}')
    print(f'\nPaste these into FINAL_PARAMS and set run_mode = "train"')
    print(f'Full results saved to {model_save_path}stage2_results.csv')

    # ── Clean up checkpoints from non-best configs ────────────────────────────
    best_trial_num = int(best['stage1_trial'])
    for row in stage2_results:
        if row['stage1_trial'] == best_trial_num:
            continue
        for p_seed in STAGE2_PARTITION_SEEDS:
            ckpt_dir = model_save_path + f'stage2_trial{int(row["stage1_trial"])}_seed{p_seed}/'
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)
    print(f'Checkpoints from non-best Stage-2 configs removed.')

##################################
#### run_mode: train #############
##################################
elif run_mode == 'train':

    p = FINAL_PARAMS
    print(f'\nTraining final model with params: {p}')

    data_module, test_idx = build_data_module(
        FINAL_PARTITION_SEED, p['batch_size'], FINAL_BATCH_SEED
    )

    pl.seed_everything(FINAL_NN_SEED, workers=True)
    _nn_rng = torch.Generator()
    _nn_rng.manual_seed(FINAL_NN_SEED)
    model = NeuralNetwork(D + 4*O, p['nn_width'], 2*O, p['nn_depth'], generator=_nn_rng)
    summary(model)

    lightning_module = RegressionModule(
        model=model,
        optimizer=Adam,
        learning_rate=p['lr_init'],
        reg_coeff_l1=0.0,
        reg_coeff_l2=p['reg_l2'],
        weight_decay=0.0,
        smoothness_coeff=0.0,
        lr_patience=FINAL_LR_PATIENCE,
        lr_factor=FINAL_LR_FACTOR,
        lr_min=FINAL_LR_MIN,
    )

    logger = CSVLogger(model_save_path + 'logs', name='NeuralNetwork')

    early_stopping = EarlyStopping(
        monitor='valid_loss', patience=FINAL_ES_PATIENCE, mode='min', verbose=True,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_save_path + 'final_model/',
        save_top_k=1,
        monitor='valid_loss',
        mode='min',
    )

    trainer = Trainer(
        max_epochs=FINAL_N_EPOCHS,
        logger=logger,
        deterministic=True,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb, early_stopping],
        enable_progress_bar=True,
    )

    t0 = time()
    trainer.fit(lightning_module, datamodule=data_module)

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f'\nBest model path: {best_model_path}')
    with open(model_save_path + 'best_ckpt_path.txt', 'w') as f:
        f.write(best_model_path)

    # Also save the test indices so evaluate mode can reconstruct the test set
    np.save(model_save_path + 'test_idx.npy', np.array(test_idx))

    trainer.test(lightning_module, datamodule=data_module)

    elapsed = time() - t0
    print(f'Done! {elapsed:.1f}s / {elapsed/60:.1f}min / {elapsed/3600:.2f}hrs')


##################################
#### run_mode: evaluate ##########
##################################
elif run_mode == 'evaluate':

    # ── Load checkpoint ───────────────────────────────────────────────────────
    with open(model_save_path + 'best_ckpt_path.txt', 'r') as f:
        best_ckpt_path = f.read().strip()

    p = FINAL_PARAMS
    _nn_rng = torch.Generator()
    _nn_rng.manual_seed(FINAL_NN_SEED)
    _model = NeuralNetwork(D + 4*O, p['nn_width'], 2*O, p['nn_depth'], generator=_nn_rng)

    lightning_module = RegressionModule.load_from_checkpoint(
        best_ckpt_path,
        model=_model,
        optimizer=Adam,
        learning_rate=p['lr_init'],
        reg_coeff_l1=0.0,
        reg_coeff_l2=p['reg_l2'],
        weight_decay=0.0,
        smoothness_coeff=0.0,
        lr_patience=FINAL_LR_PATIENCE,
        lr_factor=FINAL_LR_FACTOR,
        lr_min=FINAL_LR_MIN,
    )

    # Sync model weights from checkpoint then move to CPU for inference
    model = lightning_module.model
    model.cpu()
    model.eval()

    # ── Rebuild data module with same partition seed ───────────────────────────
    data_module, test_idx = build_data_module(
        FINAL_PARTITION_SEED, p['batch_size'], FINAL_BATCH_SEED
    )

    # Verify test indices match what was used during training
    saved_test_idx = np.load(model_save_path + 'test_idx.npy')
    assert np.array_equal(np.array(test_idx), saved_test_idx), \
        "Test indices don't match saved indices — check partition seed!"

    # ── Recover scalers from data module ──────────────────────────────────────
    out_scaler_T   = data_module.out_scaler_T
    out_scaler_P   = data_module.out_scaler_P
    in_scaler_phys = data_module.in_scaler_phys
    in_scaler_T    = data_module.in_scaler_T
    in_scaler_P    = data_module.in_scaler_P
    in_scaler_Terr = data_module.in_scaler_Terr
    in_scaler_Perr = data_module.in_scaler_Perr

    # ── Reconstruct test tensors ──────────────────────────────────────────────
    NN_test_inputs_phys = torch.tensor(raw_inputs[test_idx],        dtype=torch.float32)
    NN_test_inputs_T    = torch.tensor(GP_outputs_T[test_idx],      dtype=torch.float32)
    NN_test_inputs_P    = torch.tensor(GP_outputs_P[test_idx],      dtype=torch.float32)
    NN_test_inputs_Terr = torch.tensor(GP_outputs_Terr[test_idx],   dtype=torch.float32)
    NN_test_inputs_Perr = torch.tensor(GP_outputs_Perr[test_idx],   dtype=torch.float32)
    NN_test_true_T      = torch.tensor(raw_outputs_T[test_idx],     dtype=torch.float32)
    NN_test_true_P      = torch.tensor(raw_outputs_P[test_idx],     dtype=torch.float32)

    # ── Loss curve plot ───────────────────────────────────────────────────────
    log_dir  = model_save_path + 'logs/NeuralNetwork'
    versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    csv_path = os.path.join(log_dir, sorted(versions)[-1], 'metrics.csv')
    metrics_df   = pd.read_csv(csv_path)
    train_losses = metrics_df[metrics_df['train_mse_epoch'].notna()]['train_mse_epoch'].tolist()
    eval_losses  = metrics_df[metrics_df['valid_loss'].notna()]['valid_loss'].tolist()

    actual_epochs = len(eval_losses)
    n_batches     = max(1, len(train_losses) // actual_epochs)
    x_all         = np.linspace(0, actual_epochs, len(train_losses))
    x_epoch       = np.arange(actual_epochs + 1)
    train_epoch   = [train_losses[0]] + train_losses[n_batches-1::n_batches]
    eval_epoch    = [eval_losses[0]]  + eval_losses[n_batches-1::n_batches]
    diff_epoch    = np.abs(np.array(train_epoch) - np.array(eval_epoch))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
    ax1.plot(x_all,   train_losses, alpha=0.3, color='C0', linewidth=0.5)
    ax1.plot(x_all,   eval_losses,  alpha=0.3, color='C1', linewidth=0.5)
    ax1.plot(x_epoch, train_epoch,  label='Train',      color='C0', linewidth=2, marker='o')
    ax1.plot(x_epoch, eval_epoch,   label='Validation', color='C1', linewidth=2, marker='o')
    ax2.plot(x_epoch, diff_epoch,   color='C2', linewidth=2, marker='o')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax2.set_ylabel('|Loss Diff.|')
    ax1.legend()
    ax1.grid()
    ax2.grid()
    plt.subplots_adjust(hspace=0)
    plt.savefig(plot_save_path + '/loss.pdf')
    plt.close()

    # ── Prediction loop ───────────────────────────────────────────────────────
    substep  = 100
    n_test   = len(test_idx)
    GP_res_T = np.zeros((n_test, O), dtype=float)
    GP_res_P = np.zeros((n_test, O), dtype=float)
    NN_pred_T = np.zeros((n_test, O), dtype=float) 
    NN_res_T = np.zeros((n_test, O), dtype=float)
    NN_pred_P = np.zeros((n_test, O), dtype=float) 
    NN_res_P = np.zeros((n_test, O), dtype=float)

    for NN_test_idx, (test_input_phys, gp_T, gp_P, gp_Terr, gp_Perr,
                      true_T, true_P) in enumerate(zip(
            NN_test_inputs_phys, NN_test_inputs_T, NN_test_inputs_P,
            NN_test_inputs_Terr, NN_test_inputs_Perr,
            NN_test_true_T, NN_test_true_P)):

        scaled_input = torch.tensor(np.hstack([
            in_scaler_phys.transform(test_input_phys.numpy().reshape(1, -1)),
            in_scaler_T.transform(   gp_T.numpy().reshape(1, -1)),
            in_scaler_P.transform(   gp_P.numpy().reshape(1, -1)),
            in_scaler_Terr.transform(gp_Terr.numpy().reshape(1, -1)),
            in_scaler_Perr.transform(gp_Perr.numpy().reshape(1, -1)),
        ]), dtype=torch.float32)

        with torch.no_grad():
            nn_pred = model(scaled_input).numpy()

        pred_resid_T = out_scaler_T.inverse_transform(nn_pred[:, :O].reshape(1, -1)).flatten()
        pred_resid_P = out_scaler_P.inverse_transform(nn_pred[:, O:].reshape(1, -1)).flatten()

        nn_pred_T = gp_T.numpy() + pred_resid_T
        nn_pred_P = gp_P.numpy() + pred_resid_P

        true_T_np = true_T.numpy()
        true_P_np = true_P.numpy()
        phys_np   = test_input_phys.numpy()

        GP_res_T[NN_test_idx] = gp_T.numpy()  - true_T_np
        GP_res_P[NN_test_idx] = gp_P.numpy()  - true_P_np
        NN_pred_T[NN_test_idx] = nn_pred_T
        NN_res_T[NN_test_idx] = nn_pred_T      - true_T_np
        NN_pred_P[NN_test_idx] = nn_pred_P
        NN_res_P[NN_test_idx] = nn_pred_P      - true_P_np

        if NN_test_idx % substep == 0:
            fig, axs = plt.subplot_mosaic(
                [['res_pressure', '.'], ['results', 'res_temperature']],
                figsize=(8, 6), width_ratios=(3, 1), height_ratios=(1, 3),
                layout='constrained',
            )
            axs['results'].plot(true_T_np,    true_P_np,    '.', linestyle='-', color='blue',  linewidth=2, label='Truth')
            axs['results'].plot(nn_pred_T,    nn_pred_P,         color='green', linewidth=2, label='NN prediction')
            axs['results'].plot(gp_T.numpy(), gp_P.numpy(),      color='red',   linewidth=2, label='GP prediction')
            axs['results'].invert_yaxis()
            axs['results'].set_ylabel(r'log$_{10}$ Pressure (bar)')
            axs['results'].set_xlabel('Temperature (K)')
            axs['results'].legend()
            axs['results'].grid()

            axs['res_temperature'].plot(NN_res_T[NN_test_idx], true_P_np, '.', linestyle='-', color='green', linewidth=2)
            axs['res_temperature'].plot(GP_res_T[NN_test_idx], true_P_np, '.', linestyle='-', color='red',   linewidth=2)
            axs['res_temperature'].set_xlabel('Residuals (K)')
            axs['res_temperature'].invert_yaxis()
            axs['res_temperature'].grid()
            axs['res_temperature'].axvline(0, color='black', linestyle='dashed', zorder=2)
            axs['res_temperature'].yaxis.tick_right()
            axs['res_temperature'].yaxis.set_label_position('right')
            axs['res_temperature'].sharey(axs['results'])

            axs['res_pressure'].plot(true_T_np, NN_res_P[NN_test_idx], '.', linestyle='-', color='green', linewidth=2)
            axs['res_pressure'].plot(true_T_np, GP_res_P[NN_test_idx], '.', linestyle='-', color='red',   linewidth=2)
            axs['res_pressure'].set_ylabel('Residuals (bar)')
            axs['res_pressure'].invert_yaxis()
            axs['res_pressure'].grid()
            axs['res_pressure'].axhline(0, color='black', linestyle='dashed', zorder=2)
            axs['res_pressure'].xaxis.tick_top()
            axs['res_pressure'].xaxis.set_label_position('top')
            axs['res_pressure'].sharex(axs['results'])

            plt.suptitle(
                rf'H$_2$: {phys_np[0]} bar, CO$_2$: {phys_np[1]} bar, '
                rf'LoD: {phys_np[2]:.0f} days, Obliquity: {phys_np[3]} deg, '
                rf'Teff: {phys_np[4]} K'
            )
            plt.savefig(plot_save_path + f'/pred_vs_actual_n.{NN_test_idx}.pdf')
            plt.close()

    # ── Residuals summary plot ────────────────────────────────────────────────
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, figsize=[12, 8])
    ax1.plot(GP_res_T.T, alpha=0.1, color='green')
    ax2.plot(GP_res_P.T, alpha=0.1, color='green')
    ax3.plot(NN_res_T.T, alpha=0.1, color='blue')
    ax4.plot(NN_res_P.T, alpha=0.1, color='blue')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0, color='black', linestyle='dashed')
        ax.grid()
    ax2.set_xlabel('Index')
    ax4.set_xlabel('Index')
    ax1.set_ylabel('Temperature (K)')
    ax2.set_ylabel(r'log$_{10}$ Pressure (bar)')
    ax3.set_ylabel('Temperature (K)')
    ax4.set_ylabel(r'log$_{10}$ Pressure (bar)')
    ax1.set_title('GP residuals')
    ax3.set_title('NN residuals')
    plt.subplots_adjust(hspace=0.1, bottom=0.25)

    stats_text = (
        f"--- GP Residuals ---\n"
        f"Temperature : Median = {np.median(GP_res_T):.2f} K,          Std = {np.std(GP_res_T):.2f} K\n"
        f"Pressure    : Median = {np.median(GP_res_P):.4f} log10 bar,  Std = {np.std(GP_res_P):.4f} log10 bar\n"
        f"\n"
        f"--- NN Residuals ---\n"
        f"Temperature : Median = {np.median(NN_res_T):.2f} K,          Std = {np.std(NN_res_T):.2f} K\n"
        f"Pressure    : Median = {np.median(NN_res_P):.4f} log10 bar,  Std = {np.std(NN_res_P):.4f} log10 bar"
    )
    fig.text(0.1, 0.05, stats_text, fontsize=10, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig(plot_save_path + '/res_GP_NN.pdf', bbox_inches='tight')
    plt.close()

    # ── Figure 1: RMSE histograms ─────────────────────────────────────────────
    # Per-sample RMSE across the profile axis (axis=1)
    GP_rmse_T = np.sqrt(np.mean(GP_res_T**2, axis=1))
    GP_rmse_P = np.sqrt(np.mean(GP_res_P**2, axis=1))
    NN_rmse_T = np.sqrt(np.mean(NN_res_T**2, axis=1))
    NN_rmse_P = np.sqrt(np.mean(NN_res_P**2, axis=1))
    FS=12

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    ax[0].hist(GP_rmse_T, bins=40, color='C2', alpha=0.7, density=True, linewidth=0.5, histtype='step')
    ax[0].axvline(np.median(GP_rmse_T), color='black',  linestyle='--', linewidth=2,
                label=f'Median = {np.median(GP_rmse_T):.3f}')
    
    ax[0].hist(NN_rmse_T, bins=40, color='C0', alpha=0.7, density=True, linewidth=0.5, histtype='step')
    ax[0].axvline(np.median(NN_rmse_T), color='black',  linestyle='--', linewidth=2,
                label=f'Median = {np.median(NN_rmse_T):.3f}')
    
    ax[0].set_xlabel('RMSE (K)', fontsize=FS)
    ax[0].set_ylabel('Log Density', fontsize=FS)
    ax[0].set_yscale('log')
    ax[0].legend(fontsize=10)
    ax[0].grid(alpha=0.4)

    ax[1].hist(GP_rmse_P, bins=40, color='C2', alpha=0.7, density=True, linewidth=0.5, histtype='step')
    ax[1].axvline(np.median(GP_rmse_P), color='black',  linestyle='--', linewidth=2,
                label=f'Median = {np.median(GP_rmse_P):.3f}')
    
    ax[1].hist(NN_rmse_P, bins=40, color='C0', alpha=0.7, density=True, linewidth=0.5, histtype='step')
    ax[1].axvline(np.median(NN_rmse_P), color='black',  linestyle='--', linewidth=2,
                label=f'Median = {np.median(NN_rmse_P):.3f}')
    
    ax[1].set_xlabel(r'RMSE ($\log_{10}$ bar)', fontsize=FS)
    ax[1].set_ylabel('Log Density', fontsize=FS)
    ax[1].set_yscale('log')
    ax[1].legend(fontsize=10)
    ax[1].grid(alpha=0.4)

    plt.suptitle('Per-sample RMSE distributions across test set', fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_save_path + '/rmse_histograms.pdf', bbox_inches='tight')
    plt.close()


    # ── Figure 2: RMSE vs input variables ────────────────────────────────────
    # Input variable names and indices into raw_inputs
    input_var_names = [
        r'H$_2$ (bar)',
        r'CO$_2$ (bar)',
        r'LoD (days)',
        r'Obliquity (deg)',
        r'T$_{\rm eff}$ (K)',
    ]
    test_inputs_np = raw_inputs[test_idx]   # shape (n_test, D)

    rmse_sets = [
        ('GP',      'Temperature',  GP_rmse_T, 'C2'),
        ('GP + NN', 'Temperature',  NN_rmse_T, 'C0'),
        ('GP',      'Pressure',     GP_rmse_P, 'C2'),
        ('GP + NN', 'Pressure',     NN_rmse_P, 'C0'),
    ]

    fig, axes = plt.subplots(4, D, figsize=(4 * D, 14), sharey='row')

    for row_idx, (model_label, var_label, rmse_vals, color) in enumerate(rmse_sets):
        for col_idx, var_name in enumerate(input_var_names):
            ax  = axes[row_idx, col_idx]
            x   = test_inputs_np[:, col_idx]
            corr = np.corrcoef(x, rmse_vals)[0, 1]

            ax.scatter(x, rmse_vals, s=8, alpha=0.4, color=color, linewidths=0)

            # Trend line in log or linear space depending on variable range
            x_sort  = np.sort(x)
            z       = np.polyfit(x, rmse_vals, 1)
            ax.plot(x_sort, np.polyval(z, x_sort),
                    color='black', linewidth=1.5, linestyle='--', alpha=0.8)

            ax.set_xlabel(var_name, fontsize=10)
            ax.set_title(f'r = {corr:.2f}', fontsize=10, color='black')
            ax.grid(alpha=0.3)

            if col_idx == 0:
                units = 'K' if 'Temp' in var_label else r'log$_{10}$ bar'
                ax.set_ylabel(
                    f'{model_label}\n{var_label}\nRMSE ({units})',
                    fontsize=9
                )

    plt.suptitle('RMSE vs input variables (Pearson r shown per panel)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(plot_save_path + '/rmse_vs_inputs.pdf', bbox_inches='tight')
    plt.close()

    # ── Figure 3: truth vs predicted scatter plot ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(8,8), sharey='row')
    GP_plot_color = 'sandybrown'
    GPNN_plot_color = 'skyblue'
    FS=12
    plot_alpha=0.15

    def compute_stats(y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        r2   = r2_score(y_true_flat, y_pred_flat)
        rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat)**2))
        me   = np.mean(y_pred_flat - y_true_flat)
        return r2, rmse, me

    for test_elem in range(n_test):
        #GP T
        r2, rmse, me = compute_stats(NN_test_true_T.numpy(), NN_test_inputs_T.numpy())
        axes[0, 0].errorbar(NN_test_inputs_T[test_elem], NN_test_true_T[test_elem],
                            yerr= NN_test_inputs_Terr[test_elem], fmt='o', alpha=plot_alpha,
                           color='orange', zorder=2, label=f"R$^2$={r2:.4f}\nRMSE={rmse:.4f}\nME={me:.4f}" if test_elem == 0 else None)

        #GP P
        r2, rmse, me = compute_stats(NN_test_true_P.numpy(), NN_test_inputs_P.numpy())
        axes[1, 0].errorbar(NN_test_inputs_P[test_elem], NN_test_true_P[test_elem],
                            yerr= NN_test_inputs_Perr[test_elem], fmt='o', alpha=plot_alpha, 
                            color='darkorange', zorder=2, label=f"R$^2$={r2:.4f}\n RMSE={rmse:.4f}\n ME = {me:.4f}" if test_elem == 0 else None)

        #GP+NN T
        r2, rmse, me = compute_stats(NN_test_true_T.numpy(), NN_pred_T)
        axes[0, 1].plot(NN_pred_T[test_elem], NN_test_true_T[test_elem],
                        'o', alpha=plot_alpha, 
                        color='skyblue', zorder=2, label=f"R$^2$={r2:.4f}\n RMSE={rmse:.4f}\n ME = {me:.4f}" if test_elem == 0 else None)

        #GP+NN P
        r2, rmse, me = compute_stats(NN_test_true_P.numpy(), NN_pred_P)
        axes[1, 1].plot(NN_pred_P[test_elem], NN_test_true_P[test_elem], 
                        'o', alpha=plot_alpha, 
                        color='deepskyblue', zorder=2, label=f"R$^2$={r2:.4f}\n RMSE={rmse:.4f}\n ME = {me:.4f}" if test_elem == 0 else None)

    axes[0,0].plot(np.linspace(-3000, 3000, 100), np.linspace(-3000, 3000, 100), 'k--', zorder=3)
    axes[0,0].set_xlim(min(NN_test_inputs_T.flatten()), max(NN_test_inputs_T.flatten()))
    axes[0,0].set_ylim(min(NN_test_true_T.flatten()), max(NN_test_true_T.flatten()))
    axes[0,0].set_ylabel('True Temperature (K)', fontsize=FS)
    axes[0,0].set_xlabel('Predicted Temperature (K)', fontsize=FS)
    axes[0,0].set_title('Ens-CGP', fontsize=FS)
    axes[0,0].tick_params(axis='both', labelsize=FS)
    axes[0,0].legend(fontsize=10, loc='upper left')


    axes[1,0].plot(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), 'k--', zorder=3)
    axes[1,0].set_xlim(min(NN_test_inputs_P.flatten()), max(NN_test_inputs_P.flatten()))
    axes[1,0].set_ylim(min(NN_test_true_P.flatten()), max(NN_test_true_P.flatten()))
    axes[1,0].set_ylabel(r'True Pressure ($\log_{10}$ bar)', fontsize=FS)
    axes[1,0].set_xlabel(r'Predicted Pressure ($\log_{10}$ bar)', fontsize=FS)
    axes[1,0].tick_params(axis='both', labelsize=FS)
    axes[1,0].legend(fontsize=10, loc='lower right')


    axes[0,1].plot(np.linspace(-3000, 3000, 100), np.linspace(-3000, 3000, 100), 'k--', zorder=3)
    axes[0,1].set_xlim(min(NN_pred_T.flatten()), max(NN_pred_T.flatten()))
    axes[0,1].set_ylim(min(NN_test_true_T.flatten()), max(NN_test_true_T.flatten()))
    axes[0,1].set_title('Ens-CGP + NN', fontsize=FS)
    axes[0,1].set_xlabel('Predicted Temperature (K)', fontsize=FS)
    axes[0,1].tick_params(axis='both', labelsize=FS)
    axes[0,1].legend(fontsize=10, loc='upper left')


    axes[1,1].plot(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), 'k--', zorder=3)
    axes[1,1].set_xlim(min(NN_pred_P.flatten()), max(NN_pred_P.flatten()))
    axes[1,1].set_ylim(min(NN_test_true_P.flatten()), max(NN_test_true_P.flatten()))
    axes[1,1].set_xlabel(r'Predicted Pressure ($\log_{10}$ bar)', fontsize=FS)
    axes[1,1].tick_params(axis='both', labelsize=FS)
    axes[1,1].legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.savefig(plot_save_path + '/truthvspred_scatter.pdf', bbox_inches='tight')
    plt.close()

    # ── Figure 4: corner plot of median RMSE across parameter space ────────────────────────────────────
    param_names = [
        r'H$_2$ (bar)',
        r'CO$_2$ (bar)',
        r'LoD (days)',
        r'Obliquity ($\circ$)',
        r'$T_{\rm eff}$ (K)',
    ]

    log_params  = {0, 1, 2}
    N_PARAMS    = 5
    FS=12

    # Per-parameter bin count — Teff only has 2 unique values
    param_nbins = {
        0: 10,   # H2
        1: 10,   # CO2
        2: 10,   # LoD
        3: 10,   # Obliquity
        4: 2,    # Teff
    }

    # ── Helper: bin edges ─────────────────────────────────────────────────────────
    def make_edges(vals, n_bins, log=False):
        unique_vals = np.unique(vals)
        if len(unique_vals) <= n_bins:
            # Fewer unique values than bins — use exact unique values as bin edges
            # Add a small padding on each side so all values fall inside
            edges = np.concatenate([
                [unique_vals[0] * 0.99],
                0.5 * (unique_vals[:-1] + unique_vals[1:]),
                [unique_vals[-1] * 1.01],
            ])
            return edges
        if log:
            v_min = vals.min() if vals.min() > 0 else 1e-10
            return np.geomspace(v_min, vals.max(), n_bins + 1)
        return np.linspace(vals.min(), vals.max(), n_bins + 1)

    # ── Precompute all grids ──────────────────────────────────────────────────────
    grids_T = {}
    grids_P = {}

    for i in range(N_PARAMS):
        for j in range(N_PARAMS):
            if i == j:
                continue

            x = test_inputs_np[:, j]
            y = test_inputs_np[:, i]

            n_bins_x = param_nbins[j]
            n_bins_y = param_nbins[i]

            x_edges = make_edges(x, n_bins_x, log=(j in log_params))
            y_edges = make_edges(y, n_bins_y, log=(i in log_params))

            n_x = len(x_edges) - 1
            n_y = len(y_edges) - 1

            x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
            y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, n_y - 1)

            gt     = np.full((n_y, n_x), np.nan)
            gp     = np.full((n_y, n_x), np.nan)
            counts = np.zeros((n_y, n_x), dtype=int)

            for xi in range(n_x):
                for yi in range(n_y):
                    mask = (x_bin == xi) & (y_bin == yi)
                    if mask.any():
                        gt[yi, xi]     = np.median(NN_rmse_T[mask])
                        gp[yi, xi]     = np.median(NN_rmse_P[mask])
                        counts[yi, xi] = mask.sum()

            grids_T[(i, j)] = (gt, x_edges, y_edges, counts)
            grids_P[(i, j)] = (gp, x_edges, y_edges, counts)

    # ── Global normalisations ─────────────────────────────────────────────────────
    def get_norm_range(grid_dict, log=False):
        all_vals = np.concatenate([v[0][~np.isnan(v[0])] for v in grid_dict.values()])
        if log:
            return mcolors.LogNorm(vmin=max(all_vals.min(), 1e-6), vmax=all_vals.max())
        return mcolors.Normalize(vmin=all_vals.min(), vmax=all_vals.max())

    norm_T = get_norm_range(grids_T, log=True)
    norm_P = get_norm_range(grids_P, log=True)

    # ── Build figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(N_PARAMS, N_PARAMS - 1,
                            figsize=(2.5 * (N_PARAMS - 1), 2 * N_PARAMS))

    for i in range(N_PARAMS):
        for k in range(N_PARAMS - 1):
            ax = axes[i, k]

            # Recover actual parameter index j from column index k,
            # skipping the diagonal
            is_lower = (k < i)
            j = k if is_lower else k + 1

            if is_lower:
                grid, x_edges, y_edges, counts = grids_T[(i, j)]
                cmap_used, norm_used = 'spring_r', norm_T
            else:
                grid, x_edges, y_edges, counts = grids_P[(i, j)]
                cmap_used, norm_used = 'winter_r', norm_P

            im = ax.pcolormesh(x_edges, y_edges, grid,
                            cmap=cmap_used, norm=norm_used,
                            shading='flat', edgecolors='black', linewidth=0.2)

            from scipy.interpolate import RectBivariateSpline
            from scipy.ndimage import gaussian_filter

            # ── Smooth interpolation and contours on top ──────────────────────────────
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

            # Work in log space for log-scaled axes so interpolation is uniform
            x_eval = np.log10(x_centers) if j in log_params else x_centers
            y_eval = np.log10(y_centers) if i in log_params else y_centers

            # compute full edge extent in the same space as x_eval/y_eval
            x_edge_lo = np.log10(x_edges[0])  if j in log_params else x_edges[0]
            x_edge_hi = np.log10(x_edges[-1]) if j in log_params else x_edges[-1]
            y_edge_lo = np.log10(y_edges[0])  if i in log_params else y_edges[0]
            y_edge_hi = np.log10(y_edges[-1]) if i in log_params else y_edges[-1]

            # Replace NaNs with local mean so the spline doesn't blow up
            grid_filled = grid.copy()
            nan_mask = np.isnan(grid_filled)
            if nan_mask.any():
                grid_filled[nan_mask] = np.nanmean(grid_filled)

            # Fit a bivariate spline on the bin-center grid
            # RectBivariateSpline expects (x, y) where x is the row axis — so pass
            # y_eval as the first argument (rows) and x_eval as second (columns)
            spline = RectBivariateSpline(y_eval, x_eval, grid_filled, kx=min(3, len(y_eval)-1), ky=min(3, len(x_eval)-1))

            # Evaluate on a fine grid
            N_FINE = 100
            x_fine = np.linspace(x_edge_lo, x_edge_hi, N_FINE)
            y_fine = np.linspace(y_edge_lo, y_edge_hi, N_FINE)
            grid_fine = spline(y_fine, x_fine)

            # Optional light Gaussian smoothing on top of the spline
            grid_fine = gaussian_filter(grid_fine, sigma=1.5)

            # Convert back from log space for plotting
            x_plot = 10**x_fine if j in log_params else x_fine
            y_plot = 10**y_fine if i in log_params else y_fine

            # Draw contour lines — choose ~5 levels spanning the data range
            finite_vals = grid_filled[~np.isnan(grid)]
            n_levels = 5
            levels = np.linspace(np.nanpercentile(finite_vals, 5),
                                np.nanpercentile(finite_vals, 95),
                                n_levels)

            ax.contour(x_plot, y_plot, grid_fine,
                    levels=levels,
                    colors='black',
                    linewidths=0.8,
                    alpha=0.5)

            ax.contourf(x_plot, y_plot, grid_fine,
                        levels=levels,
                        cmap=cmap_used,
                        norm=norm_used,
                        alpha=0.4)

            # ── Log scales ────────────────────────────────────────────────────────
            if j in log_params:
                ax.set_xscale('log')
                ax.xaxis.set_major_locator(ticker.LogLocator(numticks=3))
                ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=True))
            else:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune=None))
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

            if i in log_params:
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(ticker.LogLocator(numticks=3))
                ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=True))
            else:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune=None))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

            # ── Teff: force exact tick positions ──────────────────────────────────
            if j == 4:
                x_cents = 0.5 * (x_edges[:-1] + x_edges[1:])
                ax.set_xticks(x_cents)
                ax.set_xticklabels(['3000', '4500'])
            if i == 4:
                y_cents = 0.5 * (y_edges[:-1] + y_edges[1:])
                ax.set_yticks(y_cents)
                ax.set_yticklabels(['3000', '4500'])

            # ── Hide tick labels on non-edge axes ─────────────────────────────────
            if i != N_PARAMS - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param_names[j], fontsize=FS)
                ax.tick_params(axis='x', labelsize=FS-2, rotation=45)

            if k != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(param_names[i], fontsize=FS)
                ax.tick_params(axis='y', labelsize=FS-2)

    # ── Global colorbars ──────────────────────────────────────────────────────────
    cbar_ax_T = fig.add_axes([0.89, 0.11, 0.02, 0.35])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_T, cmap='spring_r'),
                cax=cbar_ax_T, orientation='vertical')
    cbar_ax_T.set_ylabel('Temp. RMSE (K)', fontsize=FS)
    cbar_ax_T.tick_params(labelsize=FS-2)

    cbar_ax_P = fig.add_axes([0.89, 0.53, 0.02, 0.35])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_P, cmap='winter_r'),
                cax=cbar_ax_P, orientation='vertical')
    cbar_ax_P.set_ylabel(r'Pressure RMSE ($\log_{10}$ bar)', fontsize=FS)
    cbar_ax_P.tick_params(labelsize=FS-2)

    plt.subplots_adjust(right=0.88, hspace=0.087, wspace=0.063)
    plt.savefig(plot_save_path + '/compact_rmse_corner.pdf', bbox_inches='tight')
    plt.close()

    # ── Figure 5: Plot of the best, worst, and median predictions ───────────────────
    # Get indices that would sort the RMSE list
    sorted_indices = np.argsort(NN_rmse_T)

    # 1. Index of Lowest RMSE (Best)
    best_idx = sorted_indices[0]

    # 2. Index of Highest RMSE (Worst)
    worst_idx = sorted_indices[-1]

    # 3. Index of Median RMSE (Middle/Average)
    median_idx = sorted_indices[len(sorted_indices) // 2]

    FS=12
    fig = plt.figure(figsize=(16, 6))

    # Outer GridSpec: 3 groups, with wspace between them
    outer_gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3,
                                left=0.06, right=0.98, top=0.82, bottom=0.1)

    group_axes = {}
    for g, col in enumerate(['best', 'median', 'worst']):
        # Inner GridSpec per group: 2 rows × 2 cols, zero internal spacing
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer_gs[g],
            width_ratios=(3, 1),
            height_ratios=(3, 1),
            wspace=0.0,
            hspace=0.0,
        )
        group_axes[f'{col}_results']         = fig.add_subplot(inner_gs[0, 0])
        group_axes[f'{col}_res_temperature'] = fig.add_subplot(inner_gs[0, 1])
        group_axes[f'{col}_res_pressure']    = fig.add_subplot(inner_gs[1, 0])
        # inner_gs[1, 1] intentionally empty

    axs = group_axes

    for col in ['best', 'median', 'worst']:
        idx = {'best': best_idx, 'median': median_idx, 'worst': worst_idx}[col]

        results_ax  = axs[f'{col}_results']
        res_temp_ax = axs[f'{col}_res_temperature']
        res_pres_ax = axs[f'{col}_res_pressure']

        # ── Main results plot ──────────────────────────────────────────────────────
        results_ax.plot(NN_test_true_T[idx,:],  NN_test_true_P[idx,:],  '.', linestyle='-', color='darkseagreen',  linewidth=2, label='Truth')
        results_ax.errorbar(NN_test_inputs_T[idx,:], NN_test_inputs_P[idx,:], xerr=NN_test_inputs_Terr[idx,:], yerr=NN_test_inputs_Perr[idx,:], alpha=0.4, fmt='-', color=GP_plot_color, linewidth=2, label='Ens-CGP')
        results_ax.plot(NN_test_inputs_T[idx,:], NN_test_inputs_P[idx,:], '-', color=GP_plot_color, linewidth=2)
        results_ax.plot(NN_pred_T[idx,:],        NN_pred_P[idx,:],                           color=GPNN_plot_color, linewidth=2, label='Ens-CGP+NN')
        results_ax.invert_yaxis()
        results_ax.set_ylabel(r'log$_{10}$ Pressure (bar)', fontsize=FS)
        results_ax.xaxis.set_label_position('top')
        results_ax.xaxis.tick_top()
        results_ax.tick_params(axis='x', bottom=False, labelbottom=False, top=True, labeltop=True, labelsize=FS)
        results_ax.set_xlabel('Temperature (K)', fontsize=FS)
        if col=='best':
            results_ax.legend(fontsize=FS, loc='upper right')
        results_ax.grid()

        # ── Temperature residuals ──────────────────────────────────────────────────
        res_temp_ax.plot(NN_res_T[idx,:], NN_test_true_P[idx,:], '.', linestyle='-', color=GPNN_plot_color, linewidth=2)
        res_temp_ax.errorbar(GP_res_T[idx,:], NN_test_true_P[idx,:], xerr=NN_test_inputs_Terr[idx,:],  alpha=0.4, fmt='-', color=GP_plot_color,   linewidth=2)
        res_temp_ax.plot(GP_res_T[idx,:], NN_test_true_P[idx,:], '-', color=GP_plot_color, linewidth=2)
        res_temp_ax.xaxis.set_label_position('top')
        res_temp_ax.xaxis.tick_top()
        res_temp_ax.tick_params(axis='x', bottom=False, labelbottom=False, top=True, labeltop=True, labelsize=FS)
        res_temp_ax.set_xlabel('Residuals (K)', fontsize=FS)
        res_temp_ax.sharey(results_ax)
        res_temp_ax.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)
        res_temp_ax.grid()
        res_temp_ax.axvline(0, color='darkseagreen', linestyle='dashed', zorder=2)

        # ── Pressure residuals ─────────────────────────────────────────────────────
        multiplier = 1e4 if col in ['best', 'median'] else 1e2
        multiplier_str = '$10^{-4}$' if col in ['best', 'median'] else '$10^{-2}$'

        res_pres_ax.plot(NN_test_true_T[idx,:], NN_res_P[idx,:]*multiplier, '.', linestyle='-', color=GPNN_plot_color, linewidth=2)
        res_pres_ax.errorbar(NN_test_true_T[idx,:], GP_res_P[idx,:]*multiplier, yerr=NN_test_inputs_Perr[idx,:]*multiplier,  alpha=0.4, fmt='-', color=GP_plot_color, linewidth=2)
        res_pres_ax.plot(NN_test_true_T[idx,:], GP_res_P[idx,:]*multiplier, '-', color=GP_plot_color, linewidth=2)
        res_pres_ax.set_ylabel(f'Residuals \nx{multiplier_str} (bar)', fontsize=FS)
        res_pres_ax.sharex(results_ax)
        res_pres_ax.tick_params(axis='x', bottom=False, labelbottom=False, top=False, labeltop=False, labelsize=FS)
        res_pres_ax.grid()
        res_pres_ax.axhline(0, color='darkseagreen', linestyle='dashed', zorder=2)

    # After all plotting is done, remove the rightmost x-tick label
    # on each results axis to prevent overlap with the adjacent residuals panel
    fig.canvas.draw()  # force rendering so tick positions are populated

    for col in ['worst']:
        results_ax = axs[f'{col}_results']
        # Suppress the last (rightmost) tick label on the top x-axis
        ticks = results_ax.xaxis.get_major_ticks()
        if ticks:
            for tick in ticks[-2:]:
                tick.label2.set_visible(False)

    for col, idx in [('best', best_idx), ('median', median_idx), ('worst', worst_idx)]:
        results_ax = axs[f'{col}_results']
        res_temp_ax = axs[f'{col}_res_temperature']
        
        # Get the bounding boxes of both top axes in figure coordinates
        fig.canvas.draw()
        bbox_left  = results_ax.get_position()
        bbox_right = res_temp_ax.get_position()
        
        # Centered x between left edge of results and right edge of temp residuals
        x_center = (bbox_left.x0 + bbox_right.x1) / 2
        y_top    = bbox_left.y1  # top of the results axis
        
        rmse_val_T = NN_rmse_T[idx]
        rmse_val_P = NN_rmse_P[idx]
        if col == 'best':
            multiplier = 1e5
            multiplier_str = r'$\mathbf{\times 10^{-5}}$'
        elif col == 'median':
            multiplier = 1e4
            multiplier_str = r'$\mathbf{\times 10^{-4}}$'
        else:
            multiplier = 1e2
            multiplier_str = r'$\mathbf{\times 10^{-2}}$'
        title_str = f'{col.capitalize()} (RMSE = {rmse_val_T:.2f} K, {rmse_val_P*multiplier:.2f}{multiplier_str} bar)'
        
        fig.text(x_center, y_top + 0.095, title_str,
                ha='center', va='bottom', fontsize=FS, fontweight='bold',
                transform=fig.transFigure)

    plt.savefig(plot_save_path + '/example_predictions.pdf', bbox_inches='tight')
    plt.close()

    print('Evaluation complete. Plots saved to:', plot_save_path)