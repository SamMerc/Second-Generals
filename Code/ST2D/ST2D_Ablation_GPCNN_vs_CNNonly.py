#############################
#### Importing libraries ####
#############################
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# This script does not train anything: it reloads the two checkpoints produced
# by ST2D_GP_CNN_Tuning.py (run_mode='train', ens-CGP + CNN) and ST2D_CNN.py
# (CNN-only) and builds the ST2D counterpart of TP_Ablation_GPNN_vs_NNonly.py:
# a combined figure with map comparisons for representative test cases on top
# and a predicted-vs-true scatter (R2/RMSE/ME) on the bottom, with a third
# series (CNN-only) added to the ens-CGP / ens-CGP+CNN pair.

##########################################################
#### Paths ################################################
##########################################################
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'

gpcnn_model_dir   = base_dir + 'Model_Storage/ST_Hyperparam_tuning_LRinit_CNNdepth_CNNchannels_L2_BS_WC/'
cnnonly_model_dir = base_dir + 'Model_Storage/cnn_only/'

plot_save_path = base_dir + 'Plots/Ablation_GPCNN_vs_CNNonly/'
if not os.path.isdir(plot_save_path):
    os.mkdir(plot_save_path)

##########################################################
#### Raw data (identical preprocessing in both fits) ####
##########################################################
raw_data3000 = np.loadtxt(base_dir + 'Data/bt-3000k/training_data_ST2D.csv', delimiter=',')
raw_data4500 = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')

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

PARTITION_SEED = 4

# ── Recreate the exact same train/valid/test split used by both scripts ────
partition_rng = torch.Generator()
partition_rng.manual_seed(PARTITION_SEED)
train_idx, valid_idx, test_idx = torch.utils.data.random_split(
    range(N), data_partition, generator=partition_rng
)
train_idx = np.array(list(train_idx))
test_idx  = np.array(list(test_idx))

saved_test_idx = np.load(gpcnn_model_dir + 'test_idx.npy')
assert np.array_equal(test_idx, saved_test_idx), \
    "Reconstructed test indices don't match the ens-CGP+CNN saved test_idx — check the split seeds/data."

n_test = len(test_idx)

##########################################################
#### Load the ens-CGP cache (needed for the GP+CNN fit) ##
##########################################################
gp_cache_path = base_dir + f'Model_Storage/gp_ST_cache_Nn{N_neighbor}_seed{shuffle_seed}.npz'
if not os.path.exists(gp_cache_path):
    raise RuntimeError(
        f'No ens-CGP cache found at {gp_cache_path}. Run ST2D_GP_CNN_Tuning.py first '
        f'so the cache is built.'
    )
cache = np.load(gp_cache_path)
GP_outputs     = cache['GP_outputs']
GP_outputs_err = cache['GP_outputs_err']

residuals = raw_outputs - GP_outputs


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


def load_model_weights(model, ckpt_path):
    """Load the plain nn.Module weights out of a Lightning checkpoint (keys are
    prefixed with 'model.'; the RegressionModule's scaler buffers are ignored)."""
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.cpu()
    model.eval()
    return model


###################
#### Build CNNs ####
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
    """ens-CGP + CNN model (ST2D_GP_CNN.py / ST2D_GP_CNN_Tuning.py)."""
    def __init__(self, input_channels, hidden_channels, output_channels, depth):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_channels) for _ in range(depth)])
        self.output_proj = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.output_proj(self.blocks(self.input_proj(x)))


class CNN(nn.Module):
    """CNN-only model (ST2D_CNN.py): single Linear projection to (C, H, W)."""
    def __init__(self, input_dim, hidden_channels, depth, img_height, img_width):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.img_height = img_height
        self.img_width = img_width
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_channels * img_height * img_width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_channels) for _ in range(depth)])
        self.output_proj = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.view(-1, self.hidden_channels, self.img_height, self.img_width)
        return self.output_proj(self.blocks(x))


##########################################################
#### Load ens-CGP + CNN checkpoint (ST2D_GP_CNN_Tuning.py)
##########################################################
GPCNN_PARAMS = {
    'cnn_depth'    : 10,
    'cnn_channels' : 256,
}

with open(gpcnn_model_dir + 'best_ckpt_path.txt', 'r') as f:
    gpcnn_ckpt_path = resolve_ckpt_path(f.read().strip())

gpcnn_model = load_model_weights(
    ResidualCNN(D + 2, GPCNN_PARAMS['cnn_channels'], 1, GPCNN_PARAMS['cnn_depth']),
    gpcnn_ckpt_path,
)

##########################################################
#### Load CNN-only checkpoint (ST2D_CNN.py) #############
##########################################################
CNNONLY_PARAMS = {
    'cnn_depth'    : 10,
    'cnn_channels' : 256,
}

with open(cnnonly_model_dir + 'best_ckpt_path.txt', 'r') as f:
    cnnonly_ckpt_path = resolve_ckpt_path(f.read().strip())

cnnonly_model = load_model_weights(
    CNN(D, CNNONLY_PARAMS['cnn_channels'], CNNONLY_PARAMS['cnn_depth'], IMG_H, IMG_W),
    cnnonly_ckpt_path,
)

##########################################################
#### Rebuild scalers (fit on the same train indices) ####
##########################################################
def _f32(x):
    return torch.tensor(x, dtype=torch.float32).cpu().numpy()

gpcnn_in_scaler_phys = StandardScaler().fit(_f32(raw_inputs[train_idx]))
gpcnn_in_scaler_pred = StandardScaler().fit(_f32(GP_outputs[train_idx]))
gpcnn_in_scaler_err  = StandardScaler().fit(_f32(GP_outputs_err[train_idx]))
gpcnn_out_scaler     = StandardScaler().fit(_f32(residuals[train_idx]))

cnnonly_in_scaler  = StandardScaler().fit(_f32(raw_inputs[train_idx]))
cnnonly_out_scaler = StandardScaler().fit(_f32(raw_outputs[train_idx]))

##########################################################
#### Run inference on the shared test set ################
##########################################################
test_inputs_phys = raw_inputs[test_idx]
test_true        = raw_outputs[test_idx]
GP_pred          = GP_outputs[test_idx]
GP_err           = GP_outputs_err[test_idx]

BATCH = 256

# ── ens-CGP + CNN ───────────────────────────────────────────────────────────
phys_s = gpcnn_in_scaler_phys.transform(test_inputs_phys)
pred_s = gpcnn_in_scaler_pred.transform(GP_pred)
err_s  = gpcnn_in_scaler_err.transform(GP_err)

gpcnn_resid_scaled = np.zeros((n_test, O))
with torch.no_grad():
    for s in range(0, n_test, BATCH):
        sl = slice(s, s + BATCH)
        b = len(phys_s[sl])
        phys_maps = torch.tensor(phys_s[sl], dtype=torch.float32)[:, :, None, None].expand(b, D, IMG_H, IMG_W)
        pred_map  = torch.tensor(pred_s[sl].reshape(b, 1, IMG_H, IMG_W), dtype=torch.float32)
        err_map   = torch.tensor(err_s[sl].reshape(b, 1, IMG_H, IMG_W),  dtype=torch.float32)
        out = gpcnn_model(torch.cat([phys_maps, pred_map, err_map], dim=1))
        gpcnn_resid_scaled[sl] = out.numpy().reshape(b, -1)

GPCNN_pred = GP_pred + gpcnn_out_scaler.inverse_transform(gpcnn_resid_scaled)

# ── CNN only ────────────────────────────────────────────────────────────────
cnnonly_in = cnnonly_in_scaler.transform(test_inputs_phys)
cnnonly_out_scaled = np.zeros((n_test, O))
with torch.no_grad():
    for s in range(0, n_test, BATCH):
        sl = slice(s, s + BATCH)
        out = cnnonly_model(torch.tensor(cnnonly_in[sl], dtype=torch.float32))
        cnnonly_out_scaled[sl] = out.numpy().reshape(len(cnnonly_in[sl]), -1)

CNNonly_pred = cnnonly_out_scaler.inverse_transform(cnnonly_out_scaled)

# ── Residuals ───────────────────────────────────────────────────────────────
GP_res       = GP_pred      - test_true
GPCNN_res    = GPCNN_pred   - test_true
CNNonly_res  = CNNonly_pred - test_true

GPCNN_rmse   = np.sqrt(np.mean(GPCNN_res**2,   axis=1))
CNNonly_rmse = np.sqrt(np.mean(CNNonly_res**2, axis=1))

##########################################################
#### Combined ablation figure ############################
##########################################################
GP_plot_color      = 'sandybrown'
GPCNN_plot_color   = 'skyblue'
CNNonly_plot_color = 'mediumseagreen'

FS = 14
plot_alpha = 0.1


def compute_stats(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    r2   = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat)**2))
    me   = np.mean(y_pred_flat - y_true_flat)
    return r2, rmse, me


# ── Pick representative cases (median and median-1sigma) using ens-CGP+CNN RMSE ──
sorted_indices          = np.argsort(GPCNN_rmse)
median_idx              = sorted_indices[len(sorted_indices) // 2]
median_minus_1sigma     = np.median(GPCNN_rmse) - np.std(GPCNN_rmse)
median_minus_1sigma_idx = np.argmin(np.abs(GPCNN_rmse - median_minus_1sigma))

idxs   = [median_idx, median_minus_1sigma_idx]
titles = ['Median', r'Median-$\mathbf{1\sigma}$']

cmap = sns.color_palette("rocket", as_cmap=True)

# Rows: four temperature maps sharing one scale, three residual maps sharing
# another, then the ens-CGP uncertainty map on its own scale (same grouping as
# ST2D_GP_CNN_Tuning.py's combined figure).
row_data = [test_true, GP_pred, GPCNN_pred, CNNonly_pred, GP_res, GPCNN_res, CNNonly_res, GP_err]
row_labels = ['Truth', 'Ens-CGP', 'Ens-CGP + CNN', 'CNN only',
              'Ens-CGP Res.', 'Ens-CGP + CNN Res.', 'CNN only Res.', 'Ens-CGP Unc.']
N_TEMP_ROWS = 4
N_RES_ROWS  = 3
UNC_ROW     = N_TEMP_ROWS + N_RES_ROWS
N_ROWS      = len(row_data)

fig = plt.figure(figsize=(16, 27))
master_gs = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=(24, 6),
    hspace=0.1,
    left=0.07, right=0.97, top=0.96, bottom=0.04
)

# ─────────────────────────────────────────────────────────────────────────────
# TOP SECTION — map comparisons, 8 rows x 2 columns
# ─────────────────────────────────────────────────────────────────────────────
top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=master_gs[0], wspace=0.20)

axs_heat       = np.empty((N_ROWS, 2), dtype=object)
cbar_axes_temp = {}
cbar_axes_res  = {}
cbar_axes_unc  = {}

for col in range(2):
    inner = gridspec.GridSpecFromSubplotSpec(
        N_ROWS, 3, subplot_spec=top_gs[col],
        width_ratios=[20, 1, 1], hspace=0.06, wspace=0.05,
    )
    for row in range(N_ROWS):
        axs_heat[row, col] = fig.add_subplot(inner[row, 0])
    cbar_axes_temp[col] = fig.add_subplot(inner[0:N_TEMP_ROWS, 1])
    cbar_axes_res[col]  = fig.add_subplot(inner[N_TEMP_ROWS:UNC_ROW, 1])
    cbar_axes_unc[col]  = fig.add_subplot(inner[UNC_ROW, 2])

for col, (idx, title) in enumerate(zip(idxs, titles)):
    axs_heat[0, col].set_title(
        f'{title}\n'
        f'Ens-CGP+CNN: RMSE = {GPCNN_rmse[idx]:.2f} K   '
        f'CNN only: RMSE = {CNNonly_rmse[idx]:.2f} K',
        fontsize=FS - 1, fontweight='bold'
    )

    temp_data = np.concatenate([d[idx, :] for d in row_data[:N_TEMP_ROWS]])
    res_data  = np.concatenate([d[idx, :] for d in row_data[N_TEMP_ROWS:UNC_ROW]])
    vmin_t, vmax_t = temp_data.min(), temp_data.max()
    vmin_r, vmax_r = res_data.min(),  res_data.max()
    vmin_u, vmax_u = row_data[UNC_ROW][idx, :].min(), row_data[UNC_ROW][idx, :].max()

    for row, data in enumerate(row_data):
        if row < N_TEMP_ROWS:
            vmin, vmax = vmin_t, vmax_t
        elif row < UNC_ROW:
            vmin, vmax = vmin_r, vmax_r
        else:
            vmin, vmax = vmin_u, vmax_u
        ax = axs_heat[row, col]
        sns.heatmap(data[idx, :].reshape((IMG_H, IMG_W)), ax=ax,
                    vmin=vmin, vmax=vmax, cbar=False, cmap=cmap)
        ax.set_yticks(np.linspace(0, IMG_H, 5))
        if col == 0:
            ax.set_yticklabels(np.linspace(90, -90, 5).astype(int), fontsize=FS - 2)
            ax.set_ylabel(r'Latitude ($\circ$)', fontsize=FS)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        if row == N_ROWS - 1:
            ax.set_xticks(np.linspace(0, IMG_W, 5))
            ax.set_xticklabels(np.linspace(-180, 180, 5).astype(int), rotation=0, fontsize=FS - 2)
            ax.set_xlabel(r'Longitude ($\circ$)', fontsize=FS)
        else:
            ax.set_xticks([])

    sm_t = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin_t, vmax=vmax_t))
    fig.colorbar(sm_t, cax=cbar_axes_temp[col]).set_label('Temperature (K)', fontsize=FS)
    sm_r = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin_r, vmax=vmax_r))
    fig.colorbar(sm_r, cax=cbar_axes_res[col]).set_label('Residual (K)', fontsize=FS)
    sm_u = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin_u, vmax=vmax_u))
    cb_u = fig.colorbar(sm_u, cax=cbar_axes_unc[col])
    cb_u.set_label('Uncertainty (K)', fontsize=FS - 2)
    cb_u.ax.tick_params(labelsize=FS - 4)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM SECTION — pred-vs-truth scatter with R2/RMSE/ME (3-way)
# ─────────────────────────────────────────────────────────────────────────────
bot_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=master_gs[1])
ax = fig.add_subplot(bot_gs[0])

ax.plot(np.linspace(-1000, 1000, 100), np.linspace(-1000, 1000, 100),
        'k--', zorder=4, label='Perfect prediction - 1:1 line')

r2_gp,     rmse_gp,     me_gp     = compute_stats(test_true, GP_pred)
r2_gpcnn,  rmse_gpcnn,  me_gpcnn  = compute_stats(test_true, GPCNN_pred)
r2_cnnonly, rmse_cnnonly, me_cnnonly = compute_stats(test_true, CNNonly_pred)

# Subsample maps (every 10th, shuffled) so the scatter stays renderable
np.random.seed(45)
sub = np.random.permutation(n_test)[::10]

# ens-CGP first (zorder=1), CNN-only next (zorder=2), ens-CGP+CNN last (zorder=3)
for k, t in enumerate(sub):
    first = (k == 0)
    ax.errorbar(GP_pred[t], test_true[t], yerr=GP_err[t], fmt='o', alpha=plot_alpha,
                color='orangered', zorder=1,
                label=f"Ens-CGP      R$^2$={r2_gp:.4f}  RMSE={rmse_gp:.4f}  ME={me_gp:.4f}" if first else None)
    ax.plot(CNNonly_pred[t], test_true[t], 'o', alpha=plot_alpha,
            color=CNNonly_plot_color, zorder=2,
            label=f"CNN only     R$^2$={r2_cnnonly:.4f}  RMSE={rmse_cnnonly:.4f}  ME={me_cnnonly:.4f}" if first else None)
    ax.plot(GPCNN_pred[t], test_true[t], 'o', alpha=plot_alpha,
            color='dodgerblue', zorder=3,
            label=f"Ens-CGP+CNN  R$^2$={r2_gpcnn:.4f}  RMSE={rmse_gpcnn:.4f}  ME={me_gpcnn:.4f}" if first else None)

all_pred = np.concatenate([GP_pred[sub].ravel(), GPCNN_pred[sub].ravel(), CNNonly_pred[sub].ravel()])
ax.set_xlim(all_pred.min(), all_pred.max())
ax.set_ylim(test_true[sub].min(), test_true[sub].max())
ax.set_ylabel('True Temperature (K)', fontsize=FS)
ax.set_xlabel('Predicted Temperature (K)', fontsize=FS)
ax.tick_params(axis='both', labelsize=FS)
ax.legend(fontsize=10, loc='lower left')

# ── Row labels ──────────────────────────────────────────────────────────────
fig.canvas.draw()
for row, label in enumerate(row_labels):
    bbox = axs_heat[row, 0].get_position()
    fig.text(0.01, (bbox.y0 + bbox.y1) / 2, label, ha='left', va='center',
             fontsize=FS, fontweight='bold', rotation=90, transform=fig.transFigure)

plt.savefig(plot_save_path + '/combined_figures_ablation.png', bbox_inches='tight', dpi=300)
plt.close()

print('Ablation comparison complete. Plot saved to:', plot_save_path)
print(f'Ens-CGP      : R2={r2_gp:.4f}  RMSE={rmse_gp:.3f} K  ME={me_gp:.3f} K')
print(f'Ens-CGP+CNN  : R2={r2_gpcnn:.4f}  RMSE={rmse_gpcnn:.3f} K  ME={me_gpcnn:.3f} K')
print(f'CNN only     : R2={r2_cnnonly:.4f}  RMSE={rmse_cnnonly:.3f} K  ME={me_cnnonly:.3f} K')
