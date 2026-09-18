#############################
#### Importing libraries ####
#############################
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
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

print('Ablation comparison complete. Plot saved to:', plot_save_path)
