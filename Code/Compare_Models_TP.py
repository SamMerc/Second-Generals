#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
from torch.optim import Adam
import pytorch_lightning as pl
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from itertools import combinations


##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################

def check_and_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
raw_T_data = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_T.csv', delimiter=',')
raw_P_data = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_P.csv', delimiter=',')
model_save_path = base_dir + 'Model_Storage/Server/Model_Storage/'
plot_save_path  = base_dir + 'Plots/Model_Compare_Plots/'
check_and_make_dir(plot_save_path)   # only the plot dir is ever written to

raw_inputs    = raw_T_data[:, :4]
raw_outputs_T = raw_T_data[:, 5:]
raw_outputs_P = np.log10(raw_P_data[:, 5:] / 1000)

N = raw_inputs.shape[0]
D = raw_inputs.shape[1]
O = raw_outputs_T.shape[1]

INPUT_LABELS = [
    r'H$_2$ Pressure (bar)',
    r'CO$_2$ Pressure (bar)',
    r'LoD (days)',
    r'Obliquity (deg)',
]

## DATA / PARTITION HYPER-PARAMETERS (must match training run exactly) ##
data_partitions = [0.7, 0.1, 0.2]
partition_seed  = 4
batch_seed      = 5
nn_width_default = 102    # used when a MODEL_CONFIG entry omits 'nn_width'
nn_depth_default = 5      # used when a MODEL_CONFIG entry omits 'nn_depth'
batch_size = 200

# -----------------------------------------------------------------------
# MULTI-MODEL COMPARISON CONFIG
# Required keys : 'label', 'color', 'ckpt'
# Optional keys : 'nn_width', 'nn_depth'  (fall back to defaults above)
# -----------------------------------------------------------------------
MODEL_CONFIGS = [
    {
        'label' : 'PNNA - Baseline',
        'color' : 'C0',
        'ckpt'  : 'NN_fixedstand_nosmooth_noreg/100000epochs_0.0WD_0.0RC_0.0SC_0.001LR_200BS.ckpt',
        'seed'  : 6,
    },
    {
        'label' : 'PNNA - L2 Reg',
        'color' : 'C1',
        'ckpt'  : 'NN_fixedstand_nosmooth_reg/100000epochs_0.0WD_0.01RC_0.0SC_0.001LR_200BS.ckpt',
        'seed'  : 6,
    },
    {
        'label' : 'PNNA - Smooth',
        'color' : 'C2',
        'ckpt'  : 'NN_smooth/100000epochs_0.0WD_0.0RC_0.001SC_0.005LR_200BS.ckpt',
        'seed'  : 6,
    },
]

# Sanity-check: all checkpoints must exist before we do any work
for cfg in MODEL_CONFIGS:
    full = model_save_path + cfg['ckpt']
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint not found:\n  {full}\n"
            "Check MODEL_CONFIGS and model_save_path."
        )
print("All checkpoints found.\n")


###########################################
#### Partition data and build datasets ####
###########################################
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs,
                 test_inputs, test_outputs, batch_size, rng):
        super().__init__()

        out_scaler_T = StandardScaler()
        out_scaler_P = StandardScaler()
        out_scaler_T.fit(train_outputs[:, :O].cpu().numpy())
        out_scaler_P.fit(train_outputs[:, O:].cpu().numpy())

        def scale_outputs(outputs):
            T_s = torch.tensor(out_scaler_T.transform(outputs[:, :O].cpu().numpy()), dtype=torch.float32)
            P_s = torch.tensor(out_scaler_P.transform(outputs[:, O:].cpu().numpy()), dtype=torch.float32)
            return torch.cat([T_s, P_s], dim=1)

        in_scaler = StandardScaler()
        in_scaler.fit(train_inputs.cpu().numpy())

        def scale_inputs(inputs):
            return torch.tensor(in_scaler.transform(inputs.cpu().numpy()), dtype=torch.float32)

        self.out_scaler_T = out_scaler_T
        self.out_scaler_P = out_scaler_P
        self.in_scaler    = in_scaler

        self.train_inputs  = scale_inputs(train_inputs)
        self.train_outputs = scale_outputs(train_outputs)
        self.valid_inputs  = scale_inputs(valid_inputs)
        self.valid_outputs = scale_outputs(valid_outputs)
        self.test_inputs   = scale_inputs(test_inputs)
        self.test_outputs  = scale_outputs(test_outputs)
        self.batch_size    = batch_size
        self.rng           = rng

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.train_inputs, self.train_outputs),
                          batch_size=self.batch_size, shuffle=True, generator=self.rng)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.valid_inputs, self.valid_outputs),
                          batch_size=self.batch_size, generator=self.rng)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.test_inputs, self.test_outputs),
                          batch_size=self.batch_size, generator=self.rng)


## Split data (seeds must match training) ##
partition_rng = torch.Generator()
partition_rng.manual_seed(partition_seed)
batch_rng = torch.Generator()
batch_rng.manual_seed(batch_seed)

train_idx, valid_idx, test_idx = torch.utils.data.random_split(
    range(N), data_partitions, generator=partition_rng
)

def to_tensor(arr, idx):
    return torch.tensor(arr[idx], dtype=torch.float32)

train_inputs    = to_tensor(raw_inputs,    train_idx)
valid_inputs    = to_tensor(raw_inputs,    valid_idx)
test_inputs     = to_tensor(raw_inputs,    test_idx)
train_outputs_T = to_tensor(raw_outputs_T, train_idx)
valid_outputs_T = to_tensor(raw_outputs_T, valid_idx)
test_outputs_T  = to_tensor(raw_outputs_T, test_idx)
train_outputs_P = to_tensor(raw_outputs_P, train_idx)
valid_outputs_P = to_tensor(raw_outputs_P, valid_idx)
test_outputs_P  = to_tensor(raw_outputs_P, test_idx)

train_outputs = torch.cat([train_outputs_T, train_outputs_P], dim=1)
valid_outputs = torch.cat([valid_outputs_T, valid_outputs_P], dim=1)
test_outputs  = torch.cat([test_outputs_T,  test_outputs_P],  dim=1)

data_module = CustomDataModule(
    train_inputs, train_outputs,
    valid_inputs, valid_outputs,
    test_inputs,  test_outputs,
    batch_size, batch_rng
)

# Raw (unscaled) test inputs/outputs used later for plotting
raw_test_inputs = test_inputs.cpu().numpy()
true_T = test_outputs_T.cpu().numpy()
true_P = test_outputs_P.cpu().numpy()


##################
#### Build NN ####
##################
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, generator=None):
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)


###################################
#### Define optimization block ####
###################################
class RegressionModule(pl.LightningModule):
    def __init__(self, model, optimizer, learning_rate, weight_decay=0.0,
                 reg_coeff_l1=0.0, reg_coeff_l2=0.0, smoothness_coeff=0.0):
        super().__init__()
        self.model          = model
        self.learning_rate  = learning_rate
        self.reg_coeff_l1   = reg_coeff_l1
        self.reg_coeff_l2   = reg_coeff_l2
        self.smoothness_coeff = smoothness_coeff
        self.weight_decay   = weight_decay
        self.loss_fn        = nn.MSELoss()
        self.optimizer_class = optimizer

    def forward(self, x):
        return self.model(x)

    # training/validation/test steps kept minimal — only needed so
    # load_from_checkpoint can reconstruct the module without errors
    def training_step(self, batch):
        X, y = batch
        return self.loss_fn(self(X), y)

    def validation_step(self, batch):
        X, y = batch
        return self.loss_fn(self(X), y)

    def test_step(self, batch):
        X, y = batch
        return self.loss_fn(self(X), y)

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)


##############################################
#### Helper: evaluate one model on test set ##
##############################################

def _parse_hparams_from_ckpt_name(ckpt_path):
    """
    Extract learning_rate, reg_coeff_l2, weight_decay, smoothness_coeff
    from the standardised checkpoint filename convention:
      {epochs}epochs_{wd}WD_{rc}RC_{sc}SC_{lr}LR_{bs}BS.ckpt
    The path may include subdirectory prefixes — we only parse the basename.
    """
    stem = os.path.basename(ckpt_path).replace('.ckpt', '')
    parts = stem.split('_')
    # Build a lookup: suffix → value
    lookup = {}
    for part in parts:
        for tag in ('epochs', 'WD', 'RC', 'SC', 'LR', 'BS'):
            if part.endswith(tag):
                try:
                    lookup[tag] = float(part[:-len(tag)])
                except ValueError:
                    pass
    return {
        'learning_rate'  : lookup.get('LR', 1e-3),
        'reg_coeff_l2'   : lookup.get('RC', 0.0),
        'weight_decay'   : lookup.get('WD', 0.0),
        'smoothness_coeff': lookup.get('SC', 0.0),
    }


def evaluate_model(cfg, data_module):
    """
    Load a checkpoint and return per-sample RMSE for T and P on the test set.

    Parameters
    ----------
    cfg         : one entry from MODEL_CONFIGS
    data_module : fitted CustomDataModule (provides scalers)

    Returns
    -------
    rmse_T : ndarray (n_test,)  — per-profile temperature RMSE in K
    rmse_P : ndarray (n_test,)  — per-profile pressure RMSE in log10 bar
    """
    width = cfg.get('nn_width', nn_width_default)
    depth = cfg.get('nn_depth', nn_depth_default)
    ckpt_path = model_save_path + cfg['ckpt']

    hparams = _parse_hparams_from_ckpt_name(ckpt_path)

    NN_rng = torch.Generator()
    NN_rng.manual_seed(cfg['seed'])

    mdl = NeuralNetwork(D, width, 2 * O, depth, generator=NN_rng)
    lm  = RegressionModule.load_from_checkpoint(
        ckpt_path,
        model=mdl,
        optimizer=Adam,
        **hparams,
        reg_coeff_l1=0.0,   # l1 was never used; RC in filename = l2
    )
    mdl = lm.model.cpu().eval()

    scaled_inputs = torch.tensor(
        data_module.in_scaler.transform(raw_test_inputs), dtype=torch.float32
    )

    with torch.no_grad():
        preds = mdl(scaled_inputs).numpy()   # (n_test, 2*O)

    pred_T = data_module.out_scaler_T.inverse_transform(preds[:, :O])
    pred_P = data_module.out_scaler_P.inverse_transform(preds[:, O:])

    rmse_T = np.sqrt(np.mean((pred_T - true_T) ** 2, axis=1))
    rmse_P = np.sqrt(np.mean((pred_P - true_P) ** 2, axis=1))
    return rmse_T, rmse_P


#######################################################
#### Corner plot: performance across 4-D input space ##
#######################################################

def make_corner_plot(results, input_data, input_labels, metric='T',
                     title='Corner Plot', save_path=None):
    """
    Corner plot comparing multiple models.

    Diagonal  : overlaid histograms of per-sample RMSE (density), dashed
                median lines
    Lower tri : scatter of test points in each (input_i, input_j) plane,
                coloured by RMSE; models distinguished by marker shape + jitter
    Upper tri : hidden

    Parameters
    ----------
    results      : list of dicts with keys 'label', 'color', 'rmse_T', 'rmse_P'
    input_data   : ndarray (n_test, 4) — raw (unscaled) test inputs
    input_labels : list of 4 axis-label strings
    metric       : 'T' or 'P'
    title        : figure suptitle
    save_path    : if given, figure is saved here (no overwrite risk — plots only)
    """
    n_dims   = input_data.shape[1]
    n_models = len(results)
    markers  = ['o', 's', '^', 'D', 'v', 'P']

    fig, axes = plt.subplots(n_dims, n_dims,
                             figsize=(3.5 * n_dims, 3.5 * n_dims))

    for row in range(n_dims):
        for col in range(n_dims):
            ax = axes[row, col]

            # ── upper-triangle: hide ──────────────────────────────────────
            if col > row:
                ax.set_visible(False)
                continue

            # ── diagonal: RMSE histograms ─────────────────────────────────
            if col == row:
                for res in results:
                    vals = res[f'rmse_{metric}']
                    ax.hist(vals, bins=30, color=res['color'], alpha=0.5,
                            label=res['label'], density=True,
                            histtype='stepfilled', edgecolor='none')
                    ax.axvline(np.median(vals), color=res['color'],
                               linestyle='--', linewidth=1.2)
                ax.set_xlabel(f'RMSE {metric}', fontsize=8)
                ax.set_ylabel('Density', fontsize=8)
                if row == 0:
                    ax.legend(fontsize=6, loc='upper right')

            # ── lower-triangle: scatter coloured by RMSE ──────────────────
            else:
                x_data = input_data[:, col]
                y_data = input_data[:, row]
                x_range = x_data.max() - x_data.min()

                # Shared colour scale across all models for this panel
                all_vals = np.concatenate([res[f'rmse_{metric}'] for res in results])
                vmin, vmax = np.percentile(all_vals, [5, 95])

                sc_handle = None
                for k, res in enumerate(results):
                    vals   = res[f'rmse_{metric}']
                    jitter = 0.01 * (k - (n_models - 1) / 2) * x_range
                    sc_handle = ax.scatter(
                        x_data + jitter, y_data,
                        c=vals, cmap='viridis', vmin=vmin, vmax=vmax,
                        marker=markers[k % len(markers)],
                        s=8, alpha=0.6, linewidths=0,
                    )

                cbar = fig.colorbar(sc_handle, ax=ax, pad=0.02, fraction=0.046)
                cbar.set_label(f'RMSE {metric}', fontsize=7)
                cbar.ax.tick_params(labelsize=6)

            # ── axis labels on outer edges only ───────────────────────────
            if row == n_dims - 1:
                ax.set_xlabel(input_labels[col], fontsize=9)
            else:
                ax.set_xticklabels([])

            if col == 0 and row != col:
                ax.set_ylabel(input_labels[row], fontsize=9)
            elif row != col:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.4)

    fig.suptitle(title, fontsize=13, y=1.01)

    # ── Legend: marker shape → model identity ─────────────────────────────
    handles = [
        mlines.Line2D([], [], marker=markers[k % len(markers)],
                      color='w', markerfacecolor=res['color'],
                      markersize=8, label=res['label'])
        for k, res in enumerate(results)
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9,
               bbox_to_anchor=(1.0, 1.0), title='Models')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


########################################################
#### Heatmap: binned mean RMSE across all 2-D slices ##
########################################################

def make_performance_heatmap(results, input_data, input_labels, metric='T',
                              n_bins=6, save_path=None):
    """
    For each pair of input dimensions, show a 2-D heatmap of mean RMSE
    (test points binned into an n_bins x n_bins grid), one column per model.
    Colour scale is shared across models for each pair so comparisons are fair.

    Parameters
    ----------
    results      : list of dicts with keys 'label', 'color', 'rmse_T', 'rmse_P'
    input_data   : ndarray (n_test, 4)
    input_labels : list of 4 strings
    metric       : 'T' or 'P'
    n_bins       : grid resolution per axis
    save_path    : optional save path
    """
    pairs    = list(combinations(range(input_data.shape[1]), 2))
    n_pairs  = len(pairs)
    n_models = len(results)

    fig, axes = plt.subplots(n_pairs, n_models,
                             figsize=(4.5 * n_models, 4.0 * n_pairs),
                             squeeze=False)

    for p_idx, (xi, xj) in enumerate(pairs):
        x = input_data[:, xi]
        y = input_data[:, xj]
        x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
        y_edges = np.linspace(y.min(), y.max(), n_bins + 1)

        # Build all heatmaps first so colour scale can be shared
        all_hmaps = []
        for res in results:
            vals = res[f'rmse_{metric}']
            hmap = np.full((n_bins, n_bins), np.nan)
            for ix in range(n_bins):
                for iy in range(n_bins):
                    mask = (
                        (x >= x_edges[ix]) & (x < x_edges[ix + 1]) &
                        (y >= y_edges[iy]) & (y < y_edges[iy + 1])
                    )
                    if mask.sum() > 0:
                        hmap[iy, ix] = np.mean(vals[mask])
            all_hmaps.append(hmap)

        for m_idx, (res, hmap) in enumerate(zip(results, all_hmaps)):
            ax = axes[p_idx, m_idx]
            im = ax.imshow(
                hmap, origin='lower', aspect='auto',
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                cmap='viridis')
            
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'Mean RMSE {metric}', fontsize=8)
            ax.set_xlabel(input_labels[xi], fontsize=9)
            ax.set_ylabel(input_labels[xj], fontsize=9)
            ax.set_title(res['label'], fontsize=10)
            ax.tick_params(labelsize=7)

    fig.suptitle(f'Mean RMSE {metric} across 4-D input space (binned)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


##########################
#### Main comparison  ####
##########################

# --- 1. Evaluate every model ---
print("--- Evaluating models ---")
results = []

for cfg in MODEL_CONFIGS:
    print(f"  Loading: {cfg['label']}  ({cfg['ckpt']})")
    rmse_T, rmse_P = evaluate_model(cfg, data_module)
    results.append({
        'label'  : cfg['label'],
        'color'  : cfg['color'],
        'rmse_T' : rmse_T,
        'rmse_P' : rmse_P,
    })
    print(f"    T  →  median = {np.median(rmse_T):.3f} K,           std = {np.std(rmse_T):.3f} K")
    print(f"    P  →  median = {np.median(rmse_P):.4f} log10 bar,   std = {np.std(rmse_P):.4f}")


# --- 2. Corner plots ---
print("\n--- Generating corner plots ---")

make_corner_plot(
    results, raw_test_inputs, INPUT_LABELS,
    metric='T',
    title='Corner Plot: Temperature RMSE across Input Space',
    save_path=plot_save_path + 'corner_T.pdf',
)

make_corner_plot(
    results, raw_test_inputs, INPUT_LABELS,
    metric='P',
    title=r'Corner Plot: Pressure RMSE (log$_{10}$ bar) across Input Space',
    save_path=plot_save_path + 'corner_P.pdf',
)


# --- 3. Heatmap comparison ---
print("\n--- Generating heatmap comparison plots ---")

make_performance_heatmap(
    results, raw_test_inputs, INPUT_LABELS,
    metric='T', n_bins=6,
    save_path=plot_save_path + 'heatmap_T.pdf',
)

make_performance_heatmap(
    results, raw_test_inputs, INPUT_LABELS,
    metric='P', n_bins=6,
    save_path=plot_save_path + 'heatmap_P.pdf',
)

plt.show()
print("\nDone.")