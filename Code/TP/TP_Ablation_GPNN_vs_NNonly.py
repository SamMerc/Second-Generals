#############################
#### Importing libraries ####
#############################
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import ticker
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
import torch
from torch import nn
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.math import convolve_and_sample_variable_resolution_breads
from petitRADTRANS.stellar_spectra.phoenix import PhoenixStarTable
import petitRADTRANS.physical_constants as cst
import warnings
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

# This script does not train anything: it reloads the two checkpoints produced
# by TP_GP_MLP_Tuning.py (run_mode='train', ens-CGP + NN) and TP_MLP.py
# (NN-only) and reproduces the "combined figure" from TP_GP_MLP_Tuning.py's
# evaluate mode with a third series (NN-only) added, so the two can be
# compared side by side as an ablation.

##########################################################
#### Paths ################################################
##########################################################
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'

gpnn_model_dir   = base_dir + 'Model_Storage/BO_Hyperparam_tuning_LRinit_NNdepth_NNwidth_L2_BS/'
nnonly_model_dir = base_dir + 'Model_Storage/nn_only/'

plot_save_path = base_dir + 'Plots/Ablation_GPNN_vs_NNonly/'
if not os.path.isdir(plot_save_path):
    os.mkdir(plot_save_path)

##########################################################
#### Raw data (identical preprocessing in both fits) ####
##########################################################
raw_T_data3000 = np.loadtxt(base_dir + 'Data/bt-3000k/training_data_T.csv', delimiter=',')
raw_T_data4500 = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_T.csv', delimiter=',')
raw_P_data3000 = np.loadtxt(base_dir + 'Data/bt-3000k/training_data_P.csv', delimiter=',')
raw_P_data4500 = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_P.csv', delimiter=',')

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

PARTITION_SEED = 4
BATCH_SEED     = 5
NN_SEED        = 6

# ── Recreate the exact same train/valid/test split used by both scripts ────
partition_rng = torch.Generator()
partition_rng.manual_seed(PARTITION_SEED)
train_idx, valid_idx, test_idx = torch.utils.data.random_split(
    range(N), data_partition, generator=partition_rng
)
train_idx = np.array(list(train_idx))
test_idx  = np.array(list(test_idx))

saved_test_idx = np.load(gpnn_model_dir + 'test_idx.npy')
assert np.array_equal(test_idx, saved_test_idx), \
    "Reconstructed test indices don't match the ens-CGP+NN saved test_idx — check the split seeds/data."

n_test = len(test_idx)

##########################################################
#### Load the ens-CGP cache (needed for the GP+NN fit) ##
##########################################################
gp_cache_path = base_dir + f'Model_Storage/gp_cache_Nn{N_neighbor}_seed{shuffle_seed}.npz'
if not os.path.exists(gp_cache_path):
    raise RuntimeError(
        f'No ens-CGP cache found at {gp_cache_path}. Run TP_GP_MLP_Tuning.py first '
        f'so the cache is built.'
    )
cache = np.load(gp_cache_path)
GP_outputs_T    = cache['GP_outputs_T']
GP_outputs_P    = cache['GP_outputs_P']
GP_outputs_Terr = cache['GP_outputs_Terr']
GP_outputs_Perr = cache['GP_outputs_Perr']

residuals_T = raw_outputs_T - GP_outputs_T
residuals_P = raw_outputs_P - GP_outputs_P


def resolve_ckpt_path(stored_path):
    """best_ckpt_path.txt holds the absolute path from whichever machine last
    ran training (e.g. a remote cluster) — rewrite it onto this machine's
    base_dir using the 'Model_Storage/...' suffix, which is identical on both."""
    marker = 'Model_Storage/'
    idx = stored_path.find(marker)
    if idx == -1:
        return stored_path
    local_path = base_dir + stored_path[idx:]
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f'Checkpoint not found locally either:\n  stored: {stored_path}\n  local:  {local_path}'
        )
    return local_path


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

    def forward(self, x):
        return self.model(x)

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


##########################################################
#### Load ens-CGP + NN checkpoint (TP_GP_MLP_Tuning.py) ##
##########################################################
GPNN_PARAMS = {
    'lr_init'    : 0.00012362412294491496,
    'nn_depth'   : 16,
    'nn_width'   : 209,
    'reg_l2'     : 8.584521938735355e-06,
    'smoothness_coeff' : 0.01,
}
GPNN_LR_PATIENCE = 50
GPNN_LR_FACTOR   = 0.7
GPNN_LR_MIN      = 1e-7

with open(gpnn_model_dir + 'best_ckpt_path.txt', 'r') as f:
    gpnn_ckpt_path = resolve_ckpt_path(f.read().strip())

_gpnn_nn_rng = torch.Generator()
_gpnn_nn_rng.manual_seed(NN_SEED)
gpnn_model = NeuralNetwork(D + 4 * O, GPNN_PARAMS['nn_width'], 2 * O,
                            GPNN_PARAMS['nn_depth'], generator=_gpnn_nn_rng)

gpnn_lightning = RegressionModule.load_from_checkpoint(
    gpnn_ckpt_path,
    model=gpnn_model,
    optimizer=Adam,
    learning_rate=GPNN_PARAMS['lr_init'],
    reg_coeff_l1=0.0,
    reg_coeff_l2=GPNN_PARAMS['reg_l2'],
    weight_decay=0.0,
    smoothness_coeff=GPNN_PARAMS['smoothness_coeff'],
    lr_patience=GPNN_LR_PATIENCE,
    lr_factor=GPNN_LR_FACTOR,
    lr_min=GPNN_LR_MIN,
)
gpnn_model = gpnn_lightning.model
gpnn_model.cpu()
gpnn_model.eval()

##########################################################
#### Load NN-only checkpoint (TP_MLP.py) ################
##########################################################
NNONLY_PARAMS = {
    'lr_init'    : 0.00012362412294491496,
    'nn_depth'   : 16,
    'nn_width'   : 209,
    'reg_l2'     : 8.584521938735355e-06,
    'smoothness_coeff' : 0.01,
}
NNONLY_LR_PATIENCE = 50
NNONLY_LR_FACTOR   = 0.7
NNONLY_LR_MIN      = 1e-7

with open(nnonly_model_dir + 'best_ckpt_path.txt', 'r') as f:
    nnonly_ckpt_path = resolve_ckpt_path(f.read().strip())

_nnonly_nn_rng = torch.Generator()
_nnonly_nn_rng.manual_seed(NN_SEED)
nnonly_model = NeuralNetwork(D, NNONLY_PARAMS['nn_width'], 2 * O,
                              NNONLY_PARAMS['nn_depth'], generator=_nnonly_nn_rng)

nnonly_lightning = RegressionModule.load_from_checkpoint(
    nnonly_ckpt_path,
    model=nnonly_model,
    optimizer=Adam,
    learning_rate=NNONLY_PARAMS['lr_init'],
    reg_coeff_l1=0.0,
    reg_coeff_l2=NNONLY_PARAMS['reg_l2'],
    weight_decay=0.0,
    smoothness_coeff=NNONLY_PARAMS['smoothness_coeff'],
    lr_patience=NNONLY_LR_PATIENCE,
    lr_factor=NNONLY_LR_FACTOR,
    lr_min=NNONLY_LR_MIN,
)
nnonly_model = nnonly_lightning.model
nnonly_model.cpu()
nnonly_model.eval()

##########################################################
#### Rebuild scalers (fit on the same train indices) ####
##########################################################
def _f32(x):
    return torch.tensor(x, dtype=torch.float32).cpu().numpy()

gpnn_in_scaler_phys = StandardScaler().fit(_f32(raw_inputs[train_idx]))
gpnn_in_scaler_T    = StandardScaler().fit(_f32(GP_outputs_T[train_idx]))
gpnn_in_scaler_P    = StandardScaler().fit(_f32(GP_outputs_P[train_idx]))
gpnn_in_scaler_Terr = StandardScaler().fit(_f32(GP_outputs_Terr[train_idx]))
gpnn_in_scaler_Perr = StandardScaler().fit(_f32(GP_outputs_Perr[train_idx]))
gpnn_out_scaler_T   = StandardScaler().fit(_f32(residuals_T[train_idx]))
gpnn_out_scaler_P   = StandardScaler().fit(_f32(residuals_P[train_idx]))

nnonly_in_scaler_phys = StandardScaler().fit(_f32(raw_inputs[train_idx]))
nnonly_out_scaler_T   = StandardScaler().fit(_f32(raw_outputs_T[train_idx]))
nnonly_out_scaler_P   = StandardScaler().fit(_f32(raw_outputs_P[train_idx]))

##########################################################
#### Run inference on the shared test set ################
##########################################################
test_inputs_phys = raw_inputs[test_idx]
test_true_T      = raw_outputs_T[test_idx]
test_true_P      = raw_outputs_P[test_idx]

GP_pred_T = GP_outputs_T[test_idx]
GP_pred_P = GP_outputs_P[test_idx]
GP_err_T  = GP_outputs_Terr[test_idx]
GP_err_P  = GP_outputs_Perr[test_idx]

# ── ens-CGP + NN ────────────────────────────────────────────────────────────
gpnn_scaled_input = np.hstack([
    gpnn_in_scaler_phys.transform(test_inputs_phys),
    gpnn_in_scaler_T.transform(GP_pred_T),
    gpnn_in_scaler_P.transform(GP_pred_P),
    gpnn_in_scaler_Terr.transform(GP_err_T),
    gpnn_in_scaler_Perr.transform(GP_err_P),
])
with torch.no_grad():
    gpnn_out = gpnn_model(torch.tensor(gpnn_scaled_input, dtype=torch.float32)).numpy()

GPNN_resid_T = gpnn_out_scaler_T.inverse_transform(gpnn_out[:, :O])
GPNN_resid_P = gpnn_out_scaler_P.inverse_transform(gpnn_out[:, O:])
GPNN_pred_T  = GP_pred_T + GPNN_resid_T
GPNN_pred_P  = GP_pred_P + GPNN_resid_P

# ── NN only ─────────────────────────────────────────────────────────────────
nnonly_scaled_input = nnonly_in_scaler_phys.transform(test_inputs_phys)
with torch.no_grad():
    nnonly_out = nnonly_model(torch.tensor(nnonly_scaled_input, dtype=torch.float32)).numpy()

NNonly_pred_T = nnonly_out_scaler_T.inverse_transform(nnonly_out[:, :O])
NNonly_pred_P = nnonly_out_scaler_P.inverse_transform(nnonly_out[:, O:])

# ── Residuals ─────────────────────────────────────────────────────────────
GP_res_T     = GP_pred_T     - test_true_T
GP_res_P     = GP_pred_P     - test_true_P
GPNN_res_T   = GPNN_pred_T   - test_true_T
GPNN_res_P   = GPNN_pred_P   - test_true_P
NNonly_res_T = NNonly_pred_T - test_true_T
NNonly_res_P = NNonly_pred_P - test_true_P

GPNN_rmse_T   = np.sqrt(np.mean(GPNN_res_T**2,   axis=1))
GPNN_rmse_P   = np.sqrt(np.mean(GPNN_res_P**2,   axis=1))
NNonly_rmse_T = np.sqrt(np.mean(NNonly_res_T**2, axis=1))
NNonly_rmse_P = np.sqrt(np.mean(NNonly_res_P**2, axis=1))

##########################################################
#### Combined ablation figure ############################
##########################################################
GP_plot_color     = 'sandybrown'
GPNN_plot_color   = 'skyblue'
NNonly_plot_color = 'mediumseagreen'

FS = 12
plot_alpha = 0.15


def compute_stats(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    r2   = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat)**2))
    me   = np.mean(y_pred_flat - y_true_flat)
    return r2, rmse, me


# ── Pick representative cases (median and median-1sigma) using ens-CGP+NN RMSE ──
sorted_indices           = np.argsort(GPNN_rmse_T)
median_idx               = sorted_indices[len(sorted_indices) // 2]
median_minus_1sigma      = np.median(GPNN_rmse_T) - np.std(GPNN_rmse_T)
median_minus_1sigma_idx  = np.argmin(np.abs(GPNN_rmse_T - median_minus_1sigma))

# ── Master figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 16))

master_gs = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=(6, 8),
    hspace=0.1,
    left=0.06, right=0.98, top=0.93, bottom=0.05
)

# ─────────────────────────────────────────────────────────────────────────────
# TOP SECTION — representative profile comparisons
# ─────────────────────────────────────────────────────────────────────────────
top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=master_gs[0], wspace=0.2)

group_axes = {}
for g, col in enumerate(['median', 'sigma']):
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=top_gs[g],
        width_ratios=(3, 1), height_ratios=(3, 1),
        wspace=0.0, hspace=0.0,
    )
    group_axes[f'{col}_results']         = fig.add_subplot(inner_gs[0, 0])
    group_axes[f'{col}_res_temperature'] = fig.add_subplot(inner_gs[0, 1])
    group_axes[f'{col}_res_pressure']    = fig.add_subplot(inner_gs[1, 0])

axs = group_axes

for col in ['median', 'sigma']:
    idx = {'median': median_idx, 'sigma': median_minus_1sigma_idx}[col]

    results_ax  = axs[f'{col}_results']
    res_temp_ax = axs[f'{col}_res_temperature']
    res_pres_ax = axs[f'{col}_res_pressure']

    results_ax.plot(test_true_T[idx, :], test_true_P[idx, :],
                     '.', linestyle='--', color='black', linewidth=2, label='Truth')
    results_ax.errorbar(GP_pred_T[idx, :], GP_pred_P[idx, :],
                         xerr=GP_err_T[idx, :], yerr=GP_err_P[idx, :],
                         alpha=0.4, fmt='-', color=GP_plot_color, linewidth=2, label='Ens-CGP')
    results_ax.plot(GP_pred_T[idx, :], GP_pred_P[idx, :], '-', color=GP_plot_color, linewidth=2)
    results_ax.plot(GPNN_pred_T[idx, :], GPNN_pred_P[idx, :],
                     color=GPNN_plot_color, linewidth=2, label='Ens-CGP+NN')
    results_ax.plot(NNonly_pred_T[idx, :], NNonly_pred_P[idx, :],
                     color=NNonly_plot_color, linewidth=2, label='NN only')
    results_ax.invert_yaxis()
    results_ax.set_ylabel(r'log$_{10}$ Pressure (bar)', fontsize=FS)
    results_ax.xaxis.set_label_position('top')
    results_ax.xaxis.tick_top()
    results_ax.tick_params(axis='x', bottom=False, labelbottom=False, top=True, labeltop=True, labelsize=FS)
    results_ax.set_xlabel('Temperature (K)', fontsize=FS)
    if col == 'median':
        results_ax.legend(fontsize=FS - 2, loc='upper right')
    results_ax.grid()

    res_temp_ax.plot(GPNN_res_T[idx, :], test_true_P[idx, :],
                      '.', linestyle='-', color=GPNN_plot_color, linewidth=2)
    res_temp_ax.plot(NNonly_res_T[idx, :], test_true_P[idx, :],
                      '.', linestyle='-', color=NNonly_plot_color, linewidth=2)
    res_temp_ax.errorbar(GP_res_T[idx, :], test_true_P[idx, :], xerr=GP_err_T[idx, :],
                          alpha=0.4, fmt='-', color=GP_plot_color, linewidth=2)
    res_temp_ax.plot(GP_res_T[idx, :], test_true_P[idx, :], '-', color=GP_plot_color, linewidth=2)
    res_temp_ax.xaxis.set_label_position('top')
    res_temp_ax.xaxis.tick_top()
    res_temp_ax.tick_params(axis='x', bottom=False, labelbottom=False, top=True, labeltop=True, labelsize=FS)
    res_temp_ax.set_xlabel('Residuals (K)', fontsize=FS)
    res_temp_ax.sharey(results_ax)
    res_temp_ax.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)
    res_temp_ax.grid()
    res_temp_ax.axvline(0, color='black', linestyle='dashed', zorder=2)

    multiplier     = 1e4
    multiplier_str = '$10^{-4}$'

    res_pres_ax.plot(test_true_T[idx, :], GPNN_res_P[idx, :] * multiplier,
                      '.', linestyle='-', color=GPNN_plot_color, linewidth=2)
    res_pres_ax.plot(test_true_T[idx, :], NNonly_res_P[idx, :] * multiplier,
                      '.', linestyle='-', color=NNonly_plot_color, linewidth=2)
    res_pres_ax.errorbar(test_true_T[idx, :], GP_res_P[idx, :] * multiplier,
                          yerr=GP_err_P[idx, :] * multiplier,
                          alpha=0.4, fmt='-', color=GP_plot_color, linewidth=2)
    res_pres_ax.plot(test_true_T[idx, :], GP_res_P[idx, :] * multiplier, '-', color=GP_plot_color, linewidth=2)
    res_pres_ax.set_ylabel(f'Residuals \nx{multiplier_str} (bar)', fontsize=FS)
    res_pres_ax.sharex(results_ax)
    res_pres_ax.tick_params(axis='x', bottom=False, labelbottom=False, top=False, labeltop=False, labelsize=FS)
    res_pres_ax.grid()
    res_pres_ax.axhline(0, color='black', linestyle='dashed', zorder=2)

# ── Panel titles: report both models' RMSE for a direct ablation readout ────
fig.canvas.draw()
multiplier      = 1e4
multiplier_str  = r'$\mathbf{\times 10^{-4}}$'
for col, col_str, idx in [('median', 'median', median_idx),
                          ('sigma',  r'median-$\mathbf{1\sigma}$', median_minus_1sigma_idx)]:
    results_ax  = axs[f'{col}_results']
    res_temp_ax = axs[f'{col}_res_temperature']
    fig.canvas.draw()
    bbox_left  = results_ax.get_position()
    bbox_right = res_temp_ax.get_position()
    x_center = (bbox_left.x0 + bbox_right.x1) / 2
    y_top    = bbox_left.y1

    title_str = (
        f'{col_str.capitalize()}\n'
        f'Ens-CGP+NN: RMSE = {GPNN_rmse_T[idx]:.2f} K, {GPNN_rmse_P[idx]*multiplier:.2f}{multiplier_str} bar\n'
        f'NN only: RMSE = {NNonly_rmse_T[idx]:.2f} K, {NNonly_rmse_P[idx]*multiplier:.2f}{multiplier_str} bar'
    )
    fig.text(x_center, y_top + 0.035, title_str,
              ha='center', va='bottom', fontsize=FS - 1, fontweight='bold',
              transform=fig.transFigure)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM SECTION — pred-vs-truth scatter with R2/RMSE/ME (3-way)
# ─────────────────────────────────────────────────────────────────────────────
bot_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=master_gs[1], hspace=0.25)

ax_T = fig.add_subplot(bot_gs[0])
ax_P = fig.add_subplot(bot_gs[1])

ax_P.plot(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), 'k--', zorder=4)
ax_T.plot(np.linspace(-3000, 3000, 100), np.linspace(-3000, 3000, 100),
          'k--', zorder=4, label='Perfect prediction - 1:1 line')

r2_gp_T,     rmse_gp_T,     me_gp_T     = compute_stats(test_true_T, GP_pred_T)
r2_gpnn_T,   rmse_gpnn_T,   me_gpnn_T   = compute_stats(test_true_T, GPNN_pred_T)
r2_nnonly_T, rmse_nnonly_T, me_nnonly_T = compute_stats(test_true_T, NNonly_pred_T)

r2_gp_P,     rmse_gp_P,     me_gp_P     = compute_stats(test_true_P, GP_pred_P)
r2_gpnn_P,   rmse_gpnn_P,   me_gpnn_P   = compute_stats(test_true_P, GPNN_pred_P)
r2_nnonly_P, rmse_nnonly_P, me_nnonly_P = compute_stats(test_true_P, NNonly_pred_P)

# Temperature — plot ens-CGP first (zorder=1), NN-only next (zorder=2), ens-CGP+NN last (zorder=3)
ax_T.errorbar(GP_pred_T.flatten(), test_true_T.flatten(), yerr=GP_err_T.flatten(),
              fmt='o', alpha=plot_alpha, color='orange', zorder=1,
              label=f"Ens-CGP      R$^2$={r2_gp_T:.4f}  RMSE={rmse_gp_T:.4f}  ME={me_gp_T:.4f}")
ax_T.plot(NNonly_pred_T.flatten(), test_true_T.flatten(),
          'o', alpha=plot_alpha, color=NNonly_plot_color, zorder=2,
          label=f"NN only      R$^2$={r2_nnonly_T:.4f}  RMSE={rmse_nnonly_T:.4f}  ME={me_nnonly_T:.4f}")
ax_T.plot(GPNN_pred_T.flatten(), test_true_T.flatten(),
          'o', alpha=plot_alpha, color='skyblue', zorder=3,
          label=f"Ens-CGP+NN  R$^2$={r2_gpnn_T:.4f}  RMSE={rmse_gpnn_T:.4f}  ME={me_gpnn_T:.4f}")

# Pressure — same ordering
ax_P.errorbar(GP_pred_P.flatten(), test_true_P.flatten(), yerr=GP_err_P.flatten(),
              fmt='o', alpha=plot_alpha, color='darkorange', zorder=1,
              label=f"Ens-CGP      R$^2$={r2_gp_P:.4f}  RMSE={rmse_gp_P:.4f}  ME={me_gp_P:.4f}")
ax_P.plot(NNonly_pred_P.flatten(), test_true_P.flatten(),
          'o', alpha=plot_alpha, color=NNonly_plot_color, zorder=2,
          label=f"NN only      R$^2$={r2_nnonly_P:.4f}  RMSE={rmse_nnonly_P:.4f}  ME={me_nnonly_P:.4f}")
ax_P.plot(GPNN_pred_P.flatten(), test_true_P.flatten(),
          'o', alpha=plot_alpha, color='deepskyblue', zorder=3,
          label=f"Ens-CGP+NN  R$^2$={r2_gpnn_P:.4f}  RMSE={rmse_gpnn_P:.4f}  ME={me_gpnn_P:.4f}")

# Temperature axis formatting
ax_T.set_xlim(min(GP_pred_T.min(), GPNN_pred_T.min(), NNonly_pred_T.min()),
              max(GP_pred_T.max(), GPNN_pred_T.max(), NNonly_pred_T.max()))
ax_T.set_ylim(test_true_T.min(), test_true_T.max())
ax_T.set_ylabel('True Temperature (K)', fontsize=FS)
ax_T.set_xlabel('Predicted Temperature (K)', fontsize=FS)
ax_T.tick_params(axis='both', labelsize=FS)
ax_T.legend(fontsize=10, loc='upper left')

# Pressure axis formatting
ax_P.set_xlim(min(GP_pred_P.min(), GPNN_pred_P.min(), NNonly_pred_P.min()),
              max(GP_pred_P.max(), GPNN_pred_P.max(), NNonly_pred_P.max()))
ax_P.set_ylim(test_true_P.min(), test_true_P.max())
ax_P.set_ylabel(r'True Pressure ($\log_{10}$ bar)', fontsize=FS)
ax_P.set_xlabel(r'Predicted Pressure ($\log_{10}$ bar)', fontsize=FS)
ax_P.tick_params(axis='both', labelsize=FS)
ax_P.legend(fontsize=10, loc='lower right')

plt.savefig(plot_save_path + '/combined_figures_ablation.png', bbox_inches='tight', dpi=1000)
plt.close()

##########################################################
#### Corner plots of median RMSE across parameter space ##
##########################################################
# Same corner plots as TP_GP_MLP_Tuning.py (compact_rmse_corner_T/P.pdf), made
# once per model and quantity (T, P), each with its own colour normalisation.

# Convert LoD to days (same mapping as TP_GP_MLP_Tuning.py)
test_inputs_np = raw_inputs[test_idx].copy()   # shape (n_test, D)
current_lod = np.unique(test_inputs_np[:, 2])
new_lod = np.array([0.17, 0.26, 0.42, 0.66, 1.04, 1.64, 2.71, 4.17, 6.67, 10.42])

for lod_idx, lod in enumerate(current_lod):
    idx = np.where(test_inputs_np[:, 2] == lod)[0]
    test_inputs_np[idx, 2] = new_lod[lod_idx]

param_names = [
    r'H$_2$ (bar)',
    r'CO$_2$ (bar)',
    r'LoD (days)',
    r'Obliquity ($\circ$)',
    r'$T_{\rm eff}$ (K)',
]

log_params  = {0, 1, 2}
N_PARAMS    = 5

param_nbins = {
    0: 10,   # H2
    1: 10,   # CO2
    2: 10,   # LoD
    3: 10,   # Obliquity
    4: 2,    # Teff
}

show_x_tick_labels: set = {
    (4, 0), (4, 1), (4, 2), (4, 3),
}
show_y_tick_labels: set = {
    (1, 0), (2, 0), (3, 0), (4, 0),
}


# ── Helper: bin edges ─────────────────────────────────────────────────────────
def make_edges(vals, n_bins, log=False):
    unique_vals = np.unique(vals)
    if len(unique_vals) <= n_bins:
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


def compute_corner_grids(rmse):
    """Lower-triangle 2-D median-RMSE grids and diagonal marginal histograms."""
    grids = {}
    for i in range(N_PARAMS):
        for j in range(i):
            x = test_inputs_np[:, j]
            y = test_inputs_np[:, i]

            x_edges = make_edges(x, param_nbins[j], log=(j in log_params))
            y_edges = make_edges(y, param_nbins[i], log=(i in log_params))
            n_x, n_y = len(x_edges) - 1, len(y_edges) - 1

            x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
            y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, n_y - 1)

            gt     = np.full((n_y, n_x), np.nan)
            counts = np.zeros((n_y, n_x), dtype=int)

            for xi in range(n_x):
                for yi in range(n_y):
                    mask = (x_bin == xi) & (y_bin == yi)
                    if mask.any():
                        gt[yi, xi]     = np.median(rmse[mask])
                        counts[yi, xi] = mask.sum()

            grids[(i, j)] = (gt, x_edges, y_edges, counts)

    # For each parameter p, bin along that axis and compute median RMSE
    # marginalised over all other parameters.
    diag_hists = {}   # p -> (bin_centers, median_rmse_per_bin, edges)
    for p in range(N_PARAMS):
        vals    = test_inputs_np[:, p]
        edges   = make_edges(vals, param_nbins[p], log=(p in log_params))
        n_bins  = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])

        bin_idx  = np.clip(np.digitize(vals, edges) - 1, 0, n_bins - 1)
        med_rmse = np.array([
            np.median(rmse[bin_idx == b]) if (bin_idx == b).any() else np.nan
            for b in range(n_bins)
        ])
        diag_hists[p] = (centers, med_rmse, edges)

    return grids, diag_hists


def get_norm_range(grid_dict, log=False):
    # Robust limits (median +/- 5*IQR) so a few outlier bins don't drive the
    # colour scale; clamped to the data range since RMSE can't go negative.
    all_vals = np.concatenate([v[0][~np.isnan(v[0])] for v in grid_dict.values()])
    med      = np.median(all_vals)
    q25, q75 = np.percentile(all_vals, [25, 75])
    iqr      = q75 - q25
    vmin = max(med - 5 * iqr, all_vals.min())
    vmax = min(med + 5 * iqr, all_vals.max())
    if log:
        return mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot_rmse_corner(grids, diag_hists, norm, cmap_name, cbar_label, save_file, FS=12):
    cmap = plt.get_cmap(cmap_name)
    fig, axes = plt.subplots(N_PARAMS, N_PARAMS,
                             figsize=(2.5 * N_PARAMS, 2 * N_PARAMS))

    for i in range(N_PARAMS):
        for j in range(N_PARAMS):
            ax = axes[i, j]

            # ── Upper triangle: hide ───────────────────────────────────────────
            if j > i:
                ax.set_visible(False)
                continue

            # ── Diagonal: marginal RMSE histogram ─────────────────────────────
            if i == j:
                centers, med_rmse, edges = diag_hists[p := i]
                widths = np.diff(edges)

                # Colour each bar by its RMSE value using the same norm/cmap
                bar_colors = cmap(norm(med_rmse))

                ax.bar(centers, med_rmse,
                       width=widths * 0.85,          # slight gap between bars
                       color=bar_colors,
                       edgecolor='black', linewidth=0.5,
                       align='center')

                if p in log_params:
                    ax.set_xscale('log')
                    ax.xaxis.set_major_locator(ticker.LogLocator(numticks=3))
                    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=True))
                else:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune=None))

                # Teff: force exact tick positions
                if p == 4:
                    ax.set_xticks(centers)
                    ax.set_xticklabels(['3000', '4500'])

                ax.tick_params(axis='both', labelsize=0, length=2)
                ax.yaxis.set_visible(False)
                ax.set_title(param_names[p], fontsize=FS, fontstyle='italic', pad=3)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)

                continue

            # ── Lower triangle: 2-D RMSE heatmap + contours ───────────────────
            grid, x_edges, y_edges, counts = grids[(i, j)]

            ax.pcolormesh(x_edges, y_edges, grid,
                          cmap=cmap_name, norm=norm,
                          shading='flat', edgecolors='black', linewidth=0.2)

            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

            x_eval    = np.log10(x_centers) if j in log_params else x_centers
            y_eval    = np.log10(y_centers) if i in log_params else y_centers
            x_edge_lo = np.log10(x_edges[0])  if j in log_params else x_edges[0]
            x_edge_hi = np.log10(x_edges[-1]) if j in log_params else x_edges[-1]
            y_edge_lo = np.log10(y_edges[0])  if i in log_params else y_edges[0]
            y_edge_hi = np.log10(y_edges[-1]) if i in log_params else y_edges[-1]

            grid_filled = grid.copy()
            nan_mask = np.isnan(grid_filled)
            if nan_mask.any():
                grid_filled[nan_mask] = np.nanmean(grid_filled)

            spline    = RectBivariateSpline(y_eval, x_eval, grid_filled,
                                            kx=min(3, len(y_eval) - 1),
                                            ky=min(3, len(x_eval) - 1))
            N_FINE    = 100
            x_fine    = np.linspace(x_edge_lo, x_edge_hi, N_FINE)
            y_fine    = np.linspace(y_edge_lo, y_edge_hi, N_FINE)
            grid_fine = gaussian_filter(spline(y_fine, x_fine), sigma=1.5)

            x_plot = 10**x_fine if j in log_params else x_fine
            y_plot = 10**y_fine if i in log_params else y_fine

            finite_vals = grid_filled[~np.isnan(grid)]
            levels = np.linspace(np.nanpercentile(finite_vals, 5),
                                 np.nanpercentile(finite_vals, 95), 5)
            ax.contour(x_plot, y_plot, grid_fine,
                       levels=levels, colors='black', linewidths=0.8, alpha=0.5)
            ax.contourf(x_plot, y_plot, grid_fine,
                        levels=levels, cmap=cmap_name, norm=norm, alpha=0.4)

            # ── Axis scales ───────────────────────────────────────────────────
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

            if j == 4:
                ax.set_xticks(x_centers)
                ax.set_xticklabels(['3000', '4500'])
            if i == 4:
                ax.set_yticks(y_centers)
                ax.set_yticklabels(['3000', '4500'])

            # ── Tick label visibility ─────────────────────────────────────────
            if (i, j) in show_x_tick_labels:
                ax.tick_params(axis='x', labelsize=FS - 2, rotation=45)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', length=2)

            if (i, j) in show_y_tick_labels:
                ax.tick_params(axis='y', labelsize=FS - 2)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=2)

    # ── Single colorbar ───────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_name),
                 cax=cbar_ax, orientation='vertical')
    cbar_ax.set_ylabel(cbar_label, fontsize=FS)
    cbar_ax.tick_params(labelsize=FS - 2)

    plt.subplots_adjust(right=0.91, hspace=0.15, wspace=0.10)
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()


GP_rmse_T = np.sqrt(np.mean(GP_res_T**2, axis=1))
GP_rmse_P = np.sqrt(np.mean(GP_res_P**2, axis=1))

corner_quantities = [
    # (suffix, cmap, colorbar label, models [(label, tag, rmse)])
    ('T', 'spring_r', 'Temp. RMSE (K)', [
        ('Ens-CGP',      'ensCGP',    GP_rmse_T),
        ('Ens-CGP + NN', 'ensCGP_NN', GPNN_rmse_T),
        ('NN only',      'NNonly',    NNonly_rmse_T),
    ]),
    ('P', 'winter_r', r'Pressure RMSE ($\log_{10}$ bar)', [
        ('Ens-CGP',      'ensCGP',    GP_rmse_P),
        ('Ens-CGP + NN', 'ensCGP_NN', GPNN_rmse_P),
        ('NN only',      'NNonly',    NNonly_rmse_P),
    ]),
]

for suffix, cmap_name, cbar_label, models in corner_quantities:
    for label, tag, rmse in models:
        grids, diag_hists = compute_corner_grids(rmse)
        norm_corner = get_norm_range(grids, log=True)
        plot_rmse_corner(grids, diag_hists, norm_corner, cmap_name,
                         cbar_label,
                         plot_save_path + f'/compact_rmse_corner_{suffix}_{tag}.pdf')

##########################################################
#### ens-CGP uncertainty vs NN correction ################
##########################################################
# For every level of every test profile: ens-CGP errorbar (x) against the
# residual between ens-CGP+NN and ens-CGP (y), i.e. the correction the NN
# applies. Pearson / Spearman are computed over the pooled set of points.
FS = 12

fig, (ax_T, ax_P) = plt.subplots(1, 2, figsize=(14, 6))

for ax, gp_err, nn_corr, color, xlabel, ylabel in [
    (ax_T, GP_err_T, GPNN_pred_T - GP_pred_T, GPNN_plot_color,
     'Ens-CGP Temperature Uncertainty (K)',
     'Ens-CGP+NN $-$ Ens-CGP Temperature (K)'),
    (ax_P, GP_err_P, GPNN_pred_P - GP_pred_P, 'deepskyblue',
     r'Ens-CGP Pressure Uncertainty ($\log_{10}$ bar)',
     r'Ens-CGP+NN $-$ Ens-CGP Pressure ($\log_{10}$ bar)'),
]:
    x = gp_err.flatten()
    y = nn_corr.flatten()
    pearson_r,  pearson_p  = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    ax.plot(x, y, 'o', markersize=3, alpha=plot_alpha, color=color, rasterized=True,
            label=(f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.1e})\n'
                   f'Spearman $\\rho$ = {spearman_r:.3f} (p = {spearman_p:.1e})'))
    ax.axhline(0, color='black', linestyle='dashed', zorder=2)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(axis='both', labelsize=FS)
    ax.grid()
    leg = ax.legend(fontsize=FS - 1, loc='best', handlelength=0, handletextpad=0)
    for handle in leg.legend_handles:
        handle.set_visible(False)

    print(f'{xlabel}: Pearson r = {pearson_r:.4f}, Spearman rho = {spearman_r:.4f}')

plt.tight_layout()
plt.savefig(plot_save_path + '/gp_uncertainty_vs_nn_correction.pdf', bbox_inches='tight', dpi=300)
plt.close()

##########################################################
#### Transmission + emission spectra of the representative cases ####
##########################################################
# Feeds the ground-truth ExoCAM TP profile and each model's predicted TP
# profile through petitRADTRANS for the median and median-1sigma cases shown
# in the combined figure, to see how the TP discrepancy propagates into a
# transmission spectrum (transit depth) and an emission spectrum (secondary
# eclipse depth). Atmosphere composition for all spectra of a given case is
# identical and fixed by the test case's known inputs: H2 and CO2 partial
# pressures (bar) plus 1 bar of N2 (fixed in the ExoCAM setup).
#
# The eclipse depth needs a host star spectrum to divide the planet's emitted
# flux by; this uses a PHOENIX model spectrum at the case's own Teff (3000 or
# 4500 K), consistent with the fixed 1 R_sun used below for the host star
# (ExoCAM fixes instellation flux rather than an actual star-planet geometry,
# so there is no "real" stellar radius to use instead).

# ExoCAM fixes N2 partial pressure at 1 bar and defines the planet radius/gravity
# at the surface, where total pressure = p(H2) + p(CO2) + p(N2). The atmosphere is dry
# and assumed well-mixed, so volume/mass mixing ratios are constant with height.
N2_PARTIAL_PRESSURE_BAR = 1.0
MOLAR_MASS = {'H2': 2.01588, 'CO2': 44.0095, 'N2': 28.0134}  # g/mol

# Planet body
PLANET_RADIUS_CM   = 6.37e6 * 100    # 1 R_Earth, ExoCAM's fixed exoplanet radius
PLANET_GRAVITY_CGS = 9.81 * 100      # ExoCAM's fixed exoplanet surface gravity

# ExoCAM fixes instellation flux to 1360 W/m^2 regardless of stellar Teff,
# so no star-planet distance or stellar radius is implied by the GCM setup.
# 1 R_sun is used here as a fixed value for converting transit radius to transit depth
STELLAR_RADIUS_CM = 6.957e10  # 1 R_sun

N_PRESSURE_LEVELS_PRT = 100
PRT_MIN_PRESSURE_BAR  = 1e-6   # extends above the GCM's own ~0.01 bar model top; isothermally extrapolated

WAVELENGTH_RANGE_UM = (0.6, 5.3)  # JWST NIRSpec PRISM coverage

# The PRISM rebinning below convolves the pRT spectrum with a Gaussian LSF at
# each output wavelength; for output points near the edges of
# WAVELENGTH_RANGE_UM that kernel needs model data beyond the range itself
# (and the output grid can slightly overshoot WAVELENGTH_RANGE_UM[1], since
# it's built by stepping by 1/R until >= the upper bound). Without padding,
# those edge points are convolved against missing data and collapse toward
# ~0. Padding the pRT calculation beyond WAVELENGTH_RANGE_UM on both sides
# gives every output point real data to convolve against.
RADTRANS_WAVELENGTH_PAD_UM = 0.1
radtrans_wavelength_boundaries = [
    WAVELENGTH_RANGE_UM[0] - RADTRANS_WAVELENGTH_PAD_UM,
    WAVELENGTH_RANGE_UM[1] + RADTRANS_WAVELENGTH_PAD_UM,
]


# Approximate PRISM resolving power: R ~ 30 at 0.6um rising to R ~ 330 at
# 5.3um, linearly interpolated in wavelength.
# This is a simplification of the real PRISM R(lambda) curve.
def prism_resolving_power(wavelength_um):
    lam_min, lam_max = WAVELENGTH_RANGE_UM
    R_min, R_max = 30.0, 330.0
    frac = (wavelength_um - lam_min) / (lam_max - lam_min)
    return R_min + (R_max - R_min) * frac


def build_prism_wavelength_grid():
    lam_min, lam_max = WAVELENGTH_RANGE_UM
    wavelengths = [lam_min]
    while wavelengths[-1] < lam_max:
        lam = wavelengths[-1]
        R = prism_resolving_power(lam)
        wavelengths.append(lam * (1.0 + 1.0/R))
    return np.array(wavelengths)


prism_wavelengths_um = build_prism_wavelength_grid()
prism_resolutions    = prism_resolving_power(prism_wavelengths_um)

star_table = PhoenixStarTable()
star_table.load()


def case_spectra(case_idx):
    """Truth / ens-CGP (+/- 1 sigma) / ens-CGP+NN / NN-only transit depths and
    eclipse depths (ppm, PRISM-rebinned) for one test case."""
    H2_bar, CO2_bar, _, _, Teff = test_inputs_phys[case_idx]

    # ── Atmosphere composition: test-case H2/CO2 + 1 bar N2 ────────────────
    P0_bar = H2_bar + CO2_bar + N2_PARTIAL_PRESSURE_BAR  # total surface pressure (bar)
    vmr = {
        'H2':  H2_bar / P0_bar,
        'CO2': CO2_bar / P0_bar,
        'N2':  N2_PARTIAL_PRESSURE_BAR / P0_bar,
    }
    mean_molar_mass = sum(vmr[s] * MOLAR_MASS[s] for s in vmr)
    mmr = {s: vmr[s] * MOLAR_MASS[s] / mean_molar_mass for s in vmr}

    # ── Pressure grid + Radtrans object (depends on P0) ────────────────────
    pressures_bar = np.logspace(np.log10(PRT_MIN_PRESSURE_BAR), np.log10(P0_bar), N_PRESSURE_LEVELS_PRT)
    mass_fractions    = {s: np.full(N_PRESSURE_LEVELS_PRT, mmr[s]) for s in mmr}
    mean_molar_masses = np.full(N_PRESSURE_LEVELS_PRT, mean_molar_mass)

    # n68equiv (ExoCAM's radiative transfer scheme) only has line opacity for CO2;
    # H2O/CH4/C2H6 are absent from this dataset.
    # petitRADTRANS' only low-resolution correlated-k CO2 table is UCL-4000
    # (ExoMol-based), not HITRAN2020 -- noted here as a deviation from the GCM's
    # exact line list, since no HITRAN-sourced c-k CO2 table is hosted by pRT.
    # H2 contributes via collision-induced absorption + Rayleigh scattering
    # (pressure broadening of CO2 lines is handled implicitly by petitRADTRANS'
    # own line-shape treatment, not by an explicit H2 line list, consistent with
    # n68equiv where H2 has no opacity of its own either).
    radtrans = Radtrans(
        pressures=pressures_bar,
        line_species=['CO2'],
        rayleigh_species=['H2', 'CO2', 'N2'],
        gas_continuum_contributors=['H2--H2', 'CO2--CO2', 'N2--N2'],
        wavelength_boundaries=radtrans_wavelength_boundaries,
        line_opacity_mode='c-k',
    )

    def tp_to_prt_grid(T_profile, log10_P_profile_bar):
        # Interpolate the (T, log10 P[bar]) profile onto the pRT pressure grid
        # in log10(P) space, extrapolating flat (isothermal) beyond its range.
        order = np.argsort(log10_P_profile_bar)
        return np.interp(np.log10(pressures_bar), log10_P_profile_bar[order], T_profile[order])

    def transit_depth_ppm(temperatures):
        _, transit_radii_cm, _ = radtrans.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=PLANET_GRAVITY_CGS,
            reference_pressure=P0_bar,
            planet_radius=PLANET_RADIUS_CM,
        )
        return (transit_radii_cm / STELLAR_RADIUS_CM)**2 * 1e6

    wavelengths_cm, _, _ = radtrans.calculate_transit_radii(
        temperatures=tp_to_prt_grid(test_true_T[case_idx], test_true_P[case_idx]),
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=PLANET_GRAVITY_CGS,
        reference_pressure=P0_bar,
        planet_radius=PLANET_RADIUS_CM,
    )
    model_wavelengths_um = wavelengths_cm * 1e4

    # ── Host star spectrum (for eclipse depth), interpolated onto the same
    # wavelength grid calculate_flux will return (same Radtrans object, so
    # identical grid to calculate_transit_radii above) ────────────────────
    stellar_spectrum_cm, _ = star_table.compute_spectrum(Teff)
    _star_order = np.argsort(stellar_spectrum_cm[:, 0])
    stellar_wavelengths_cm = stellar_spectrum_cm[_star_order, 0]
    stellar_flux_nu        = stellar_spectrum_cm[_star_order, 1]  # erg/s/cm^2/Hz, at the stellar surface
    stellar_flux_lambda = stellar_flux_nu * cst.c / stellar_wavelengths_cm**2  # erg/s/cm^2/cm
    stellar_flux_lambda_grid = np.interp(wavelengths_cm, stellar_wavelengths_cm, stellar_flux_lambda)

    def eclipse_depth_ppm(temperatures):
        _, planet_flux_lambda, _ = radtrans.calculate_flux(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=PLANET_GRAVITY_CGS,
            planet_radius=PLANET_RADIUS_CM,
        )
        return (PLANET_RADIUS_CM / STELLAR_RADIUS_CM)**2 * (planet_flux_lambda / stellar_flux_lambda_grid) * 1e6

    def rebin_to_prism(spectrum):
        return convolve_and_sample_variable_resolution_breads(
            wavelengths=prism_wavelengths_um,
            resolutions=prism_resolutions,
            model_wavelengths=model_wavelengths_um,
            model_fluxes=spectrum,
        )

    def spectra_pair(T_profile, log10_P_profile_bar):
        """(transit depth, eclipse depth), both PRISM-rebinned, for one TP profile."""
        temperatures = tp_to_prt_grid(T_profile, log10_P_profile_bar)
        return rebin_to_prism(transit_depth_ppm(temperatures)), rebin_to_prism(eclipse_depth_ppm(temperatures))

    gp_T, gp_P, gp_Terr = GP_pred_T[case_idx], GP_pred_P[case_idx], GP_err_T[case_idx]

    truth_trans,    truth_ecl    = spectra_pair(test_true_T[case_idx], test_true_P[case_idx])
    gp_trans,       gp_ecl       = spectra_pair(gp_T, gp_P)
    # ens-CGP +/- 1 sigma band: shift the temperature profile by its uncertainty
    gp_plus_trans,  gp_plus_ecl  = spectra_pair(gp_T + gp_Terr, gp_P)
    gp_minus_trans, gp_minus_ecl = spectra_pair(gp_T - gp_Terr, gp_P)
    gpnn_trans,     gpnn_ecl     = spectra_pair(GPNN_pred_T[case_idx],   GPNN_pred_P[case_idx])
    nnonly_trans,   nnonly_ecl   = spectra_pair(NNonly_pred_T[case_idx], NNonly_pred_P[case_idx])

    return {
        'transmission': {
            'truth': truth_trans, 'gp': gp_trans, 'gp_plus': gp_plus_trans, 'gp_minus': gp_minus_trans,
            'gpnn': gpnn_trans, 'nnonly': nnonly_trans,
        },
        'emission': {
            'truth': truth_ecl, 'gp': gp_ecl, 'gp_plus': gp_plus_ecl, 'gp_minus': gp_minus_ecl,
            'gpnn': gpnn_ecl, 'nnonly': nnonly_ecl,
        },
    }


FS = 12

# Each representative case needs petitRADTRANS run once (case_spectra computes
# both transmission and emission together), then reused for both figures below.
cases = [
    ('Median',                     'median',        median_idx),
    (r'Median-$\mathbf{1\sigma}$', 'median-1sigma', median_minus_1sigma_idx),
]
case_specs = {case_name: case_spectra(case_idx) for _, case_name, case_idx in cases}


def plot_ablation_figure(spectrum_key, ylabel, spectrum_label, save_path):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(18, 8),
                             gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05, 'wspace': 0.35})

    for col, (case_str, case_name, case_idx) in enumerate(cases):
        spec = case_specs[case_name][spectrum_key]
        ax1, ax2 = axes[0, col], axes[1, col]

        ax1.fill_between(prism_wavelengths_um, spec['gp_minus'], spec['gp_plus'],
                         color=GP_plot_color, alpha=0.25, label=r'Ens-CGP ($\pm 1\sigma$)')
        ax1.plot(prism_wavelengths_um, spec['gp'],     color=GP_plot_color,     linewidth=1.5, label='Ens-CGP')
        ax1.plot(prism_wavelengths_um, spec['nnonly'], color=NNonly_plot_color, linewidth=1.5, label='NN only')
        ax1.plot(prism_wavelengths_um, spec['gpnn'],   color=GPNN_plot_color,   linewidth=1.5, label='Ens-CGP+NN')
        ax1.plot(prism_wavelengths_um, spec['truth'],  color='black', linewidth=1.5, linestyle='--',
                 zorder=5, label='Truth (ExoCAM)')
        ax1.set_ylabel(ylabel, fontsize=FS)
        ax1.tick_params(axis='both', labelsize=FS)
        ax1.grid(alpha=0.3)
        if col == 0:
            ax1.legend(fontsize=FS - 2)

        H2_bar, CO2_bar, _, _, Teff = test_inputs_phys[case_idx]
        ax1.set_title(
            f'{case_str}\n'
            rf'H$_2$: {H2_bar:.4g} bar, CO$_2$: {CO2_bar:.4g} bar, $+$1 bar N$_2$, '
            rf'$T_{{\rm eff}}$: {Teff:.0f} K',
            fontsize=FS, fontweight='bold'
        )

        # Residuals (true - pred) in ppm on the left spine, relative residuals
        # (true - pred)/true (%) on the right spine. The two aren't a fixed linear
        # rescaling of one another (true depth varies with wavelength), so the
        # right spine is independently scaled, with y-limits set explicitly so both
        # zero lines align.
        diffs = {
            'Ens-CGP':    (spec['truth'] - spec['gp'],     GP_plot_color),
            'NN only':    (spec['truth'] - spec['nnonly'], NNonly_plot_color),
            'Ens-CGP+NN': (spec['truth'] - spec['gpnn'],   GPNN_plot_color),
        }
        diff_gp_upper = spec['truth'] - spec['gp_minus']
        diff_gp_lower = spec['truth'] - spec['gp_plus']

        ax2.fill_between(prism_wavelengths_um, diff_gp_lower, diff_gp_upper, color=GP_plot_color, alpha=0.25)
        for diff, color in diffs.values():
            ax2.plot(prism_wavelengths_um, diff, color=color, linewidth=1.5)
        ax2.axhline(0, color='black', linestyle='dashed', linewidth=1)
        ax2.set_xlabel(r'Wavelength ($\mu$m)', fontsize=FS)
        ax2.set_ylabel('Residuals (ppm)', fontsize=FS)
        ax2.tick_params(axis='both', labelsize=FS)
        ax2.grid(alpha=0.3)

        ax2b = ax2.twinx()
        ax2b.set_ylabel('Relative residuals (%)', color='darkorange', fontsize=FS)
        ax2b.tick_params(axis='y', labelcolor='darkorange', labelsize=FS)
        ax2b.spines['right'].set_color('darkorange')

        all_diffs = [d for d, _ in diffs.values()] + [diff_gp_lower, diff_gp_upper]
        max_abs_ppm = np.nanmax(np.abs(np.concatenate(all_diffs)))
        max_abs_pct = np.nanmax(np.abs(np.concatenate([d / spec['truth'] * 100 for d in all_diffs])))
        ax2.set_ylim(-max_abs_ppm*1.1, max_abs_ppm*1.1)
        ax2b.set_ylim(-max_abs_pct*1.1, max_abs_pct*1.1)

        print(f'{spectrum_label} spectrum, {case_name} (test case {case_idx}):')
        for label, (diff, _) in diffs.items():
            print(f'  {label:<11s}: RMS spectrum residual = {np.sqrt(np.nanmean(diff**2)):.4f} ppm')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


plot_ablation_figure('transmission', 'Transit depth (ppm)', 'Transmission',
                      plot_save_path + '/transmission_spectrum_comparison_ablation.pdf')
plot_ablation_figure('emission', 'Eclipse depth (ppm)', 'Emission',
                      plot_save_path + '/emission_spectrum_comparison_ablation.pdf')

print('Ablation comparison complete. Plots saved to:', plot_save_path)
