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
from sklearn.preprocessing import StandardScaler
import seaborn as sns
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


##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
def check_and_make_dir(dir):
    if not os.path.isdir(dir): os.mkdir(dir)

base_dir = '/home/merci228/WORK/2G_ML/'

raw_data3000 = np.loadtxt(base_dir + 'Data/bt-3000k/training_data_ST2D.csv', delimiter=',')
raw_data4500 = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')

model_save_path = base_dir + 'Model_Storage/ST_Hyperparam_tuning_LRinit_CNNdepth_CNNchannels_L2_BS_SC/'
check_and_make_dir(model_save_path)
plot_save_path = base_dir + 'Plots/ST_Hyperparam_tuning_LRinit_CNNdepth_CNNchannels_L2_BS_SC/'
check_and_make_dir(plot_save_path)

inputs_3000 = np.hstack([raw_data3000[:, :4], np.full((len(raw_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_data4500[:, :4], np.full((len(raw_data4500), 1), 4500.0)])

raw_inputs  = np.vstack([inputs_3000,          inputs_4500         ])
raw_outputs = np.vstack([raw_data3000[:, 5:],  raw_data4500[:, 5:]])

N = raw_inputs.shape[0]
D = raw_inputs.shape[1]
O = raw_outputs.shape[1]

IMG_H, IMG_W = 46, 72
assert O == IMG_H * IMG_W, f"Output dim {O} != {IMG_H}x{IMG_W}"

shuffle_seed = 3
np.random.seed(shuffle_seed)
rp = np.random.permutation(N)
raw_inputs  = raw_inputs[rp, :]
raw_outputs = raw_outputs[rp, :]

N_neighbor     = 4
data_partition = [0.7, 0.1, 0.2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 96
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")

show_plot = False

#############################
#### run_mode selection  ####
#############################
run_mode = 'train'   # 'search1' | 'search2' | 'train' | 'evaluate'

## ── Parameters for 'train' and 'evaluate' modes ──────────────────────────────
## After Stage 2 completes, paste the best params here and switch run_mode
## to 'train', then to 'evaluate'.
FINAL_PARAMS = {
    'lr_init'          : 0.00010175170875090105,
    'cnn_depth'        : 10,
    'cnn_channels'     : 128,
    'reg_l2'           : 4.231195487398746e-06,
    'smoothness_coeff' : 0.01,
    'batch_size'       : 64,
}
FINAL_PARTITION_SEED = 4
FINAL_BATCH_SEED     = 5
FINAL_NN_SEED        = 6
FINAL_LR_PATIENCE    = 50
FINAL_LR_FACTOR      = 0.7
FINAL_LR_MIN         = 1e-7
FINAL_N_EPOCHS       = 1000
FINAL_ES_PATIENCE    = 100


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
#### Build CNN ####
###################
class ResidualConvBlock(nn.Module):
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
        return self.activation(x + self.block(x))


class ResidualCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels,
                 depth, img_height, img_width, generator=None):
        super().__init__()
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        self.img_height = img_height
        self.img_width  = img_width
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(hidden_channels) for _ in range(depth)]
        )
        self.output_proj = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class CNNDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs,
                 test_inputs, test_outputs, batch_size, rng,
                 n_phys, img_height, img_width):
        super().__init__()
        self.batch_size = batch_size
        self.rng        = rng
        self.n_phys     = n_phys
        self.img_height = img_height
        self.img_width  = img_width
        O_flat = img_height * img_width

        i0, i1, i2 = 0, n_phys, n_phys + O_flat

        # Output scaler
        out_scaler = StandardScaler()
        out_scaler.fit(train_outputs.cpu().numpy())
        self.out_scaler = out_scaler

        def scale_outputs(t):
            return torch.tensor(
                out_scaler.transform(t.cpu().numpy()), dtype=torch.float32
            )

        # Input scalers
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

        def to_cnn_input(X_flat):
            """(N, D+2*O) → (N, D+2, H, W)"""
            N_s  = X_flat.shape[0]
            phys = X_flat[:, i0:i1]
            pred = X_flat[:, i1:i2].reshape(N_s, 1, img_height, img_width)
            err  = X_flat[:, i2:  ].reshape(N_s, 1, img_height, img_width)
            phys_maps = phys[:, :, None, None].expand(N_s, n_phys, img_height, img_width)
            return torch.cat([phys_maps, pred, err], dim=1)

        def to_cnn_output(Y_flat):
            return Y_flat.reshape(-1, 1, img_height, img_width)

        self.train_inputs  = to_cnn_input(scale_flat(train_inputs))
        self.valid_inputs  = to_cnn_input(scale_flat(valid_inputs))
        self.test_inputs   = to_cnn_input(scale_flat(test_inputs))
        self.train_outputs = to_cnn_output(scale_outputs(train_outputs))
        self.valid_outputs = to_cnn_output(scale_outputs(valid_outputs))
        self.test_outputs  = to_cnn_output(scale_outputs(test_outputs))

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.train_inputs, self.train_outputs),
                          batch_size=self.batch_size, shuffle=True,
                          generator=self.rng, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.valid_inputs, self.valid_outputs),
                          batch_size=self.batch_size, generator=self.rng, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.test_inputs, self.test_outputs),
                          batch_size=self.batch_size, generator=self.rng, pin_memory=True)


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
            grad_mse = torch.autograd.grad(
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

gp_cache_path  = base_dir + f'Model_Storage/gp_ST_cache_Nn{N_neighbor}_seed{shuffle_seed}.npz'
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
            jnp.array(XTr), jnp.array(YTr), query_input,
            jnp.linalg.inv(jnp.cov(XTr)), N_neighbor,
        )
        GP_outputs[query_idx, :]     = Yh
        GP_outputs_err[query_idx, :] = err_Yh

    np.savez(gp_cache_path, GP_outputs=GP_outputs, GP_outputs_err=GP_outputs_err)
    print(f'  GP outputs cached to:\n  {gp_cache_path}')

# Residuals: what the CNN needs to learn
residuals = raw_outputs - GP_outputs


###############################
#### Shared helper function ####
###############################
def build_data_module(partition_seed, batch_size, batch_seed=5):
    """Build a CNNDataModule for a given partition seed and batch size."""
    _partition_rng = torch.Generator()
    _partition_rng.manual_seed(partition_seed)
    _batch_rng = torch.Generator()
    _batch_rng.manual_seed(batch_seed)

    _train_idx, _valid_idx, _test_idx = torch.utils.data.random_split(
        range(N), data_partition, generator=_partition_rng
    )

    def make_inputs(idx):
        return torch.cat([
            torch.tensor(raw_inputs[idx],       dtype=torch.float32),
            torch.tensor(GP_outputs[idx],       dtype=torch.float32),
            torch.tensor(GP_outputs_err[idx],   dtype=torch.float32),
        ], dim=1)

    def make_outputs(idx):
        return torch.tensor(residuals[idx], dtype=torch.float32)

    return (
        CNNDataModule(
            make_inputs(_train_idx),  make_outputs(_train_idx),
            make_inputs(_valid_idx),  make_outputs(_valid_idx),
            make_inputs(_test_idx),   make_outputs(_test_idx),
            batch_size, _batch_rng,
            n_phys=D, img_height=IMG_H, img_width=IMG_W,
        ),
        _test_idx,
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
    SEARCH_N_EPOCHS    = 1000
    SEARCH_ES_PATIENCE = 50   # tighter patience for faster search

    def print_trial_summary(study, trial):
        if trial.value is None:
            return
        print(f'\n--- Trial {trial.number} finished ---')
        print(f'  Value (mean val loss): {trial.value:.6f}')
        print(f'  Params: {trial.params}')
        print(f'  Best so far: {study.best_value:.6f} (trial {study.best_trial.number})')
        print(f'  Trials completed: {len([t for t in study.trials if t.value is not None])}')

    def objective(trial):
        lr_init          = trial.suggest_float('lr_init',    1e-4, 1e-2, log=True)
        cnn_depth        = trial.suggest_categorical('cnn_depth',    [4, 7, 10, 14])
        cnn_channels     = trial.suggest_categorical('cnn_channels', [32, 64, 128])
        reg_l2           = trial.suggest_float('reg_l2',     1e-6, 1e-3, log=True)
        smoothness_coeff = trial.suggest_categorical('smoothness_coeff', [0.0, 1e-2, 1e-1, 1.0])
        batch_size       = trial.suggest_categorical('batch_size', [16, 32, 64])

        val_losses = []

        for p_seed in PARTITION_SEEDS:

            _data_module, _ = build_data_module(p_seed, batch_size, SEARCH_BATCH_SEED)

            pl.seed_everything(SEARCH_NN_SEED, workers=True)
            _model = ResidualCNN(
                input_channels=D + 2,
                hidden_channels=cnn_channels,
                output_channels=1,
                depth=cnn_depth,
                img_height=IMG_H,
                img_width=IMG_W,
            )

            _lightning_module = RegressionModule(
                model=_model,
                optimizer=Adam,
                learning_rate=lr_init,
                reg_coeff_l1=0.0,
                reg_coeff_l2=reg_l2,
                weight_decay=0.0,
                smoothness_coeff=smoothness_coeff,
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
        n_warmup_steps=30,   # shorter warmup than MLP — CNN epochs are heavier
        interval_steps=10,
    )

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        storage=f'sqlite:///{model_save_path}optuna_study_stage1.db',
        study_name='st_map_cnn_stage1',
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
    top10 = results_df[results_df['value'].notna()].sort_values('value').head(10)[[
        'number', 'value', 'params_lr_init', 'params_cnn_depth',
        'params_cnn_channels', 'params_reg_l2', 'params_smoothness_coeff',
        'params_batch_size',
    ]]
    print(top10.to_string(index=False))

    # ── Clean up checkpoints from non-best trials ─────────────────────────────
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

    STAGE2_PARTITION_SEEDS = [4, 7, 13, 42, 99]
    STAGE2_BATCH_SEED      = 5
    STAGE2_NN_SEED         = 6
    STAGE2_LR_PATIENCE     = 50
    STAGE2_LR_FACTOR       = 0.7
    STAGE2_LR_MIN          = 1e-7
    STAGE2_N_EPOCHS        = 1000
    STAGE2_ES_PATIENCE     = 100   # relaxed vs Stage 1 for fairer comparison
    STAGE2_TOP_N           = 10

    # ── Load Stage 1 results ──────────────────────────────────────────────────
    stage1_csv = model_save_path + 'optuna_results_stage1.csv'
    assert os.path.exists(stage1_csv), f'Stage 1 results not found at {stage1_csv}'

    results_df = pd.read_csv(stage1_csv)

    # Keep only completed (non-pruned) trials
    completed = results_df[results_df['value'].notna()].copy()
    top_configs = completed.sort_values('value').head(STAGE2_TOP_N).reset_index(drop=True)

    print(f'\n=== Stage 2: Robustness evaluation of top {STAGE2_TOP_N} configs ===')
    print(f'Partition seeds: {STAGE2_PARTITION_SEEDS}')
    print(top_configs[[
        'number', 'value', 'params_lr_init', 'params_cnn_depth',
        'params_cnn_channels', 'params_reg_l2', 'params_smoothness_coeff',
        'params_batch_size',
    ]].to_string(index=False))

    stage2_results = []

    #Optuna logging
    stage2_study = optuna.create_study(
        direction='minimize',
        storage=f'sqlite:///{model_save_path}optuna_study_stage2.db',
        study_name='st_map_cnn_stage2',
        load_if_exists=True,
    )

    for rank, row in top_configs.iterrows():
        trial_num        = int(row['number'])
        lr_init          = float(row['params_lr_init'])
        cnn_depth        = int(row['params_cnn_depth'])
        cnn_channels     = int(row['params_cnn_channels'])
        reg_l2           = float(row['params_reg_l2'])
        smoothness_coeff = float(row['params_smoothness_coeff'])
        batch_size       = int(row['params_batch_size'])

        print(f'\n--- Stage 2 | Rank {rank+1} | Stage-1 trial {trial_num} ---')
        print(f'  lr={lr_init:.2e}, depth={cnn_depth}, channels={cnn_channels}, '
              f'l2={reg_l2:.2e}, smooth={smoothness_coeff:.2e}, bs={batch_size}')

        seed_val_losses = []

        for p_seed in STAGE2_PARTITION_SEEDS:

            data_module, _ = build_data_module(p_seed, batch_size, STAGE2_BATCH_SEED)

            pl.seed_everything(STAGE2_NN_SEED, workers=True)
            _model = ResidualCNN(
                input_channels=D + 2,
                hidden_channels=cnn_channels,
                output_channels=1,
                depth=cnn_depth,
                img_height=IMG_H,
                img_width=IMG_W,
            )

            _lightning_module = RegressionModule(
                model=_model,
                optimizer=Adam,
                learning_rate=lr_init,
                reg_coeff_l1=0.0,
                reg_coeff_l2=reg_l2,
                weight_decay=0.0,
                smoothness_coeff=smoothness_coeff,
                lr_patience=STAGE2_LR_PATIENCE,
                lr_factor=STAGE2_LR_FACTOR,
                lr_min=STAGE2_LR_MIN,
            )

            ckpt_dir = model_save_path + f'stage2_trial{trial_num}_seed{p_seed}/'
            checkpoint_cb = ModelCheckpoint(
                dirpath=ckpt_dir, monitor='valid_loss', mode='min', save_top_k=1,
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

        mean_loss  = float(np.nanmean(seed_val_losses))
        std_loss   = float(np.nanstd(seed_val_losses))
        worst_loss = float(np.nanmax(seed_val_losses))

        # After the per-seed loop for each config:
        trial = stage2_study.ask()
        trial.suggest_float('lr_init',  lr_init,    lr_init)
        trial.suggest_int(  'nn_depth', nn_depth,   nn_depth)
        trial.suggest_int(  'nn_width', nn_width,   nn_width)
        trial.suggest_float('reg_l2',   reg_l2,     reg_l2)
        trial.suggest_float('smoothness_coeff',   smoothness_coeff,     smoothness_coeff)
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
            'cnn_depth'       : cnn_depth,
            'cnn_channels'    : cnn_channels,
            'reg_l2'          : reg_l2,
            'smoothness_coeff': smoothness_coeff,
            'batch_size'      : batch_size,
            'per_seed_losses' : seed_val_losses,
        })

    # ── Rank and save ─────────────────────────────────────────────────────────
    stage2_df = pd.DataFrame(stage2_results).sort_values('mean_val_loss').reset_index(drop=True)
    stage2_df.to_csv(model_save_path + 'stage2_results.csv', index=False)

    print('\n=== Stage 2 Complete ===')
    print('Rankings by mean val loss across seeds:')
    print(stage2_df[[
        'stage1_trial', 'stage1_val_loss', 'mean_val_loss', 'std_val_loss',
        'worst_val_loss', 'lr_init', 'cnn_depth', 'cnn_channels',
        'reg_l2', 'smoothness_coeff', 'batch_size',
    ]].to_string(index=False))

    best = stage2_df.iloc[0]
    print(f'\nBest Stage-2 config (Stage-1 trial {int(best["stage1_trial"])}) :')
    print(f'  mean_val_loss    = {best["mean_val_loss"]:.6f}  (std={best["std_val_loss"]:.6f})')
    print(f'  lr_init          : {best["lr_init"]}')
    print(f'  cnn_depth        : {int(best["cnn_depth"])}')
    print(f'  cnn_channels     : {int(best["cnn_channels"])}')
    print(f'  reg_l2           : {best["reg_l2"]}')
    print(f'  smoothness_coeff : {best["smoothness_coeff"]}')
    print(f'  batch_size       : {int(best["batch_size"])}')
    print(f'\nPaste these into FINAL_PARAMS and set run_mode = "train"')
    print(f'Full results saved to {model_save_path}stage2_results.csv')

    # ── Clean up non-best checkpoints ────────────────────────────────────────
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
    model = ResidualCNN(
        input_channels=D + 2,
        hidden_channels=p['cnn_channels'],
        output_channels=1,
        depth=p['cnn_depth'],
        img_height=IMG_H,
        img_width=IMG_W,
        generator=_nn_rng,
    )
    summary(model, input_size=(1, D + 2, IMG_H, IMG_W))

    lightning_module = RegressionModule(
        model=model,
        optimizer=Adam,
        learning_rate=p['lr_init'],
        reg_coeff_l1=0.0,
        reg_coeff_l2=p['reg_l2'],
        weight_decay=0.0,
        smoothness_coeff=p['smoothness_coeff'],
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
        save_top_k=1, monitor='valid_loss', mode='min',
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
    _model = ResidualCNN(
        input_channels=D + 2,
        hidden_channels=p['cnn_channels'],
        output_channels=1,
        depth=p['cnn_depth'],
        img_height=IMG_H,
        img_width=IMG_W,
        generator=_nn_rng,
    )

    lightning_module = RegressionModule.load_from_checkpoint(
        best_ckpt_path,
        model=_model,
        optimizer=Adam,
        learning_rate=p['lr_init'],
        reg_coeff_l1=0.0,
        reg_coeff_l2=p['reg_l2'],
        weight_decay=0.0,
        smoothness_coeff=p['smoothness_coeff'],
        lr_patience=FINAL_LR_PATIENCE,
        lr_factor=FINAL_LR_FACTOR,
        lr_min=FINAL_LR_MIN,
    )

    model = lightning_module.model
    model.cpu()
    model.eval()

    # ── Rebuild data module with same partition seed ───────────────────────────
    data_module, test_idx = build_data_module(
        FINAL_PARTITION_SEED, p['batch_size'], FINAL_BATCH_SEED
    )

    saved_test_idx = np.load(model_save_path + 'test_idx.npy')
    assert np.array_equal(np.array(test_idx), saved_test_idx), \
        "Test indices don't match saved indices — check partition seed!"

    # ── Recover scalers ───────────────────────────────────────────────────────
    out_scaler     = data_module.out_scaler
    in_scaler_phys = data_module.in_scaler_phys
    in_scaler_pred = data_module.in_scaler_pred
    in_scaler_err  = data_module.in_scaler_err

    # ── Reconstruct test arrays ───────────────────────────────────────────────
    NN_test_inputs_phys = torch.tensor(raw_inputs[test_idx],       dtype=torch.float32)
    NN_test_inputs_pred = torch.tensor(GP_outputs[test_idx],       dtype=torch.float32)
    NN_test_inputs_err  = torch.tensor(GP_outputs_err[test_idx],   dtype=torch.float32)
    NN_test_true        = torch.tensor(raw_outputs[test_idx],      dtype=torch.float32)

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
    ax1.set_yscale('log');  ax2.set_yscale('log')
    ax2.set_xlabel('Epoch');  ax1.set_ylabel('MSE Loss');  ax2.set_ylabel('|Loss Diff.|')
    ax1.legend();  ax1.grid();  ax2.grid()
    plt.subplots_adjust(hspace=0)
    plt.savefig(plot_save_path + '/loss.pdf')
    plt.close()

    # ── Prediction loop ───────────────────────────────────────────────────────
    substep = 100
    n_test  = len(test_idx)
    GP_res  = np.zeros((n_test, O), dtype=float)
    NN_res  = np.zeros((n_test, O), dtype=float)

    for NN_test_idx, (test_input_phys, gp_pred, gp_err, true) in enumerate(zip(
            NN_test_inputs_phys, NN_test_inputs_pred, NN_test_inputs_err, NN_test_true)):

        phys_scaled = in_scaler_phys.transform(test_input_phys.numpy().reshape(1, -1))
        pred_scaled = in_scaler_pred.transform(gp_pred.numpy().reshape(1, -1))
        err_scaled  = in_scaler_err.transform( gp_err.numpy().reshape(1, -1))

        phys_maps = torch.tensor(phys_scaled, dtype=torch.float32
                    )[:, :, None, None].expand(1, D, IMG_H, IMG_W)
        pred_map  = torch.tensor(pred_scaled.reshape(1, 1, IMG_H, IMG_W), dtype=torch.float32)
        err_map   = torch.tensor(err_scaled.reshape( 1, 1, IMG_H, IMG_W), dtype=torch.float32)
        cnn_input = torch.cat([phys_maps, pred_map, err_map], dim=1)

        with torch.no_grad():
            cnn_pred = model(cnn_input)

        pred_resid    = out_scaler.inverse_transform(cnn_pred.numpy().reshape(1, -1)).flatten()
        cnn_pred_full = gp_pred.numpy() + pred_resid

        true_np  = true.numpy()
        phys_np  = test_input_phys.numpy()

        GP_res[NN_test_idx, :] = gp_pred.numpy() - true_np
        NN_res[NN_test_idx, :] = cnn_pred_full    - true_np

        if NN_test_idx % substep == 0:
            plot_true     = true_np.reshape((IMG_H, IMG_W))
            plot_gp_pred  = gp_pred.numpy().reshape((IMG_H, IMG_W))
            plot_cnn_pred = cnn_pred_full.reshape((IMG_H, IMG_W))
            plot_res_GP   = GP_res[NN_test_idx].reshape((IMG_H, IMG_W))
            plot_res_CNN  = NN_res[NN_test_idx].reshape((IMG_H, IMG_W))

            fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True, layout='constrained')
            for ax, data, title, label in zip(
                axs,
                [plot_true, plot_gp_pred, plot_cnn_pred, plot_res_GP, plot_res_CNN],
                ['Data', 'GP Model', 'CNN Model', 'GP Residuals', 'CNN Residuals'],
                ['Temperature (K)'] * 3 + ['Residual (K)'] * 2,
            ):
                ax.set_title(title)
                hm = sns.heatmap(data, ax=ax)
                hm.collections[0].colorbar.set_label(label)
                ax.set_yticks(np.linspace(0, IMG_H, 5))
                ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
                ax.set_ylabel('Latitude (deg)')

            axs[-1].set_xticks(np.linspace(0, IMG_W, 5))
            axs[-1].set_xticklabels(np.linspace(-180, 180, 5).astype(int))
            axs[-1].set_xlabel('Longitude (deg)')

            plt.suptitle(
                rf'H$_2$ : {phys_np[0]} bar, CO$_2$ : {phys_np[1]} bar, '
                rf'LoD : {phys_np[2]:.0f} days, Obliquity : {phys_np[3]} deg, '
                rf'Teff : {phys_np[4]} K'
            )
            plt.savefig(plot_save_path + f'/pred_vs_actual_n.{NN_test_idx}.pdf')
            plt.close()

    # ── Residuals summary plot ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[12, 8])
    for qid in range(n_test):
        ax1.plot(GP_res[qid, :], alpha=0.1, color='green')
        ax2.plot(NN_res[qid, :], alpha=0.1, color='blue')
    for ax in [ax1, ax2]:
        ax.axhline(0, color='black', linestyle='dashed')
        ax.grid()
    ax1.set_xlabel('Pixel Index');  ax2.set_xlabel('Pixel Index')
    ax1.set_ylabel('Temperature Residual (K)')
    ax1.set_title('GP Residuals');  ax2.set_title('CNN Residuals')

    plt.subplots_adjust(hspace=0.1, bottom=0.25)
    stats_text = (
        f"--- GP Residuals ---\n"
        f"Median = {np.median(GP_res):.2f} K,  "
        f"Std = {np.std(GP_res):.2f} K,  "
        f"RMSE = {np.sqrt(np.mean(GP_res**2)):.2f} K\n\n"
        f"--- CNN Residuals ---\n"
        f"Median = {np.median(NN_res):.2f} K,  "
        f"Std = {np.std(NN_res):.2f} K,  "
        f"RMSE = {np.sqrt(np.mean(NN_res**2)):.2f} K"
    )
    fig.text(0.1, 0.05, stats_text, fontsize=10, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig(plot_save_path + '/res_GP_CNN.pdf', bbox_inches='tight')
    plt.close()

    print('Evaluation complete. Plots saved to:', plot_save_path)