#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
from torch.optim import Adam
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from sklearn.preprocessing import StandardScaler
from itertools import combinations


##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################

def check_and_make_dir(dir):
    if not os.path.isdir(dir): os.mkdir(dir)

base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
raw_T_data = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_T.csv', delimiter=',')
raw_P_data = np.loadtxt(base_dir + 'Data/bt-4500k/training_data_P.csv', delimiter=',')
model_save_path = base_dir + 'Model_Storage/Server/Model_Storage/'
check_and_make_dir(model_save_path)
plot_save_path = base_dir + 'Plots/Model_Compare_Plots/'
check_and_make_dir(plot_save_path)

raw_inputs = raw_T_data[:, :4]
raw_outputs_T = raw_T_data[:, 5:]
raw_outputs_P = raw_P_data[:, 5:]
raw_outputs_P = np.log10(raw_outputs_P / 1000)

N = raw_inputs.shape[0]
D = raw_inputs.shape[1]
O = raw_outputs_T.shape[1]

# Input dimension labels and units (used in corner plot axis labels)
INPUT_LABELS = [
    r'H$_2$ Pressure (bar)',
    r'CO$_2$ Pressure (bar)',
    r'LoD (days)',
    r'Obliquity (deg)',
]

## HYPER-PARAMETERS ##
data_partitions = [0.7, 0.1, 0.2]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 1
torch.set_num_threads(num_threads)

partition_seed = 4
partition_rng = torch.Generator()
partition_rng.manual_seed(partition_seed)

batch_seed = 5
batch_rng = torch.Generator()
batch_rng.manual_seed(batch_seed)

NN_seed = 6
NN_rng = torch.Generator()
NN_rng.manual_seed(NN_seed)

nn_width = 102
nn_depth = 5
learning_rate = 1e-3
regularization_coeff_l1 = 0.0
regularization_coeff_l2 = 0.0
smoothness_coeff = 0.0
weight_decay = 0.0
batch_size = 200
n_epochs = 10000

# -----------------------------------------------------------------------
# MODE:  'train'  → train a single model and save it
#        'compare' → load multiple saved checkpoints and compare them
# -----------------------------------------------------------------------
run_mode = 'compare'

# -----------------------------------------------------------------------
# MULTI-MODEL COMPARISON CONFIG
# Each entry: (checkpoint_filename_stem, display_label, plot_color)
# The filename stem follows the convention used when saving:
#   f'{n_epochs}epochs_{wd}WD_{rc}RC_{sc}SC_{lr}LR_{bs}BS'
# -----------------------------------------------------------------------
MODEL_CONFIGS = [
    #PNNA
    {
        'label'    : 'PNNA - Baseline',
        'color'    : 'C0',
        'ckpt'     : 'NN_fixedstand_nosmooth_noreg/100000epochs_0.0WD_0.0RC_0.0SC_0.001LR_200BS.ckpt',
    },
    {
        'label'    : 'PNNA - L2 Reg',
        'color'    : 'C1',
        'ckpt'     : 'NN_fixedstand_nosmooth_reg/100000epochs_0.0WD_0.0RC_0.0SC_0.001LR_200BS.ckpt',
    },
    {
        'label'    : 'PNNA - Smooth',
        'color'    : 'C2',
        'ckpt'     : 'NN_smooth/100000epochs_0.0WD_0.0RC_0.001SC_0.005LR_200BS.ckpt',
    },
]


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

        train_T_scaled = torch.tensor(out_scaler_T.transform(train_outputs[:, :O].cpu().numpy()), dtype=torch.float32)
        train_P_scaled = torch.tensor(out_scaler_P.transform(train_outputs[:, O:].cpu().numpy()), dtype=torch.float32)
        valid_T_scaled = torch.tensor(out_scaler_T.transform(valid_outputs[:, :O].cpu().numpy()), dtype=torch.float32)
        valid_P_scaled = torch.tensor(out_scaler_P.transform(valid_outputs[:, O:].cpu().numpy()), dtype=torch.float32)
        test_T_scaled  = torch.tensor(out_scaler_T.transform(test_outputs[:, :O].cpu().numpy()),  dtype=torch.float32)
        test_P_scaled  = torch.tensor(out_scaler_P.transform(test_outputs[:, O:].cpu().numpy()),  dtype=torch.float32)

        train_outputs = torch.cat([train_T_scaled, train_P_scaled], dim=1)
        valid_outputs = torch.cat([valid_T_scaled, valid_P_scaled], dim=1)
        test_outputs  = torch.cat([test_T_scaled,  test_P_scaled],  dim=1)

        self.out_scaler_T = out_scaler_T
        self.out_scaler_P = out_scaler_P

        in_scaler = StandardScaler()
        in_scaler.fit(train_inputs.cpu().numpy())
        train_inputs = torch.tensor(in_scaler.transform(train_inputs.cpu().numpy()), dtype=torch.float32)
        valid_inputs = torch.tensor(in_scaler.transform(valid_inputs.cpu().numpy()), dtype=torch.float32)
        test_inputs  = torch.tensor(in_scaler.transform(test_inputs.cpu().numpy()),  dtype=torch.float32)
        self.in_scaler = in_scaler

        self.train_inputs  = train_inputs
        self.train_outputs = train_outputs
        self.valid_inputs  = valid_inputs
        self.valid_outputs = valid_outputs
        self.test_inputs   = test_inputs
        self.test_outputs  = test_outputs
        self.batch_size    = batch_size
        self.rng           = rng

    def train_dataloader(self):
        dataset = TensorDataset(self.train_inputs, self.train_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=self.rng)

    def val_dataloader(self):
        dataset = TensorDataset(self.valid_inputs, self.valid_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, generator=self.rng)

    def test_dataloader(self):
        dataset = TensorDataset(self.test_inputs, self.test_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, generator=self.rng)


## Split data ##
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(N), data_partitions, generator=partition_rng)

train_inputs    = torch.tensor(raw_inputs[train_idx],    dtype=torch.float32)
train_outputs_T = torch.tensor(raw_outputs_T[train_idx], dtype=torch.float32)
train_outputs_P = torch.tensor(raw_outputs_P[train_idx], dtype=torch.float32)
valid_inputs    = torch.tensor(raw_inputs[valid_idx],    dtype=torch.float32)
valid_outputs_T = torch.tensor(raw_outputs_T[valid_idx], dtype=torch.float32)
valid_outputs_P = torch.tensor(raw_outputs_P[valid_idx], dtype=torch.float32)
test_inputs     = torch.tensor(raw_inputs[test_idx],     dtype=torch.float32)
test_outputs_T  = torch.tensor(raw_outputs_T[test_idx],  dtype=torch.float32)
test_outputs_P  = torch.tensor(raw_outputs_P[test_idx],  dtype=torch.float32)

train_outputs = torch.cat([train_outputs_T, train_outputs_P], dim=1)
valid_outputs = torch.cat([valid_outputs_T, valid_outputs_P], dim=1)
test_outputs  = torch.cat([test_outputs_T,  test_outputs_P],  dim=1)

data_module = CustomDataModule(
    train_inputs, train_outputs,
    valid_inputs, valid_outputs,
    test_inputs, test_outputs,
    batch_size, batch_rng
)


##################
#### Build NN ####
##################
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, generator=None):
        super().__init__()
        layers = []
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
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
        self.model = model
        self.learning_rate = learning_rate
        self.reg_coeff_l1 = reg_coeff_l1
        self.reg_coeff_l2 = reg_coeff_l2
        self.smoothness_coeff = smoothness_coeff
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()
        self.optimizer_class = optimizer

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

    def compute_smoothness_constraint(self, X, output):
        if self.smoothness_coeff == 0:
            return torch.tensor(0., device=self.device)
        X_grad = X.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            output_grad = self.model(X_grad)
            grad_outputs = torch.ones_like(output_grad)
            gradients = torch.autograd.grad(
                outputs=output_grad, inputs=X_grad,
                grad_outputs=grad_outputs, create_graph=True,
                retain_graph=True, only_inputs=True)[0]
            smoothness_penalty = torch.mean(torch.norm(gradients, p=2, dim=1))
        return self.smoothness_coeff * smoothness_penalty

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        l1_penalty, l2_penalty = self.compute_weight_regularization()
        loss += l1_penalty + l2_penalty
        loss += self.compute_smoothness_constraint(X, pred)
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
        return self.optimizer_class(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)


##############################################
#### Helper: evaluate one model on test set ##
##############################################

def evaluate_model(ckpt_path, width, depth, data_module, return_raw_inputs=False):
    """
    Load a checkpoint, run inference on the full test set, and return
    per-sample RMSE arrays for T and P (shape: [n_test]).
    """
    mdl = NeuralNetwork(D, width, 2 * O, depth)

    lr = float(ckpt_path.split('LR_')[0].split('_')[-1])
    rl2 = float(ckpt_path.split('RC_')[0].split('_')[-1])
    wd = float(ckpt_path.split('WD_')[0].split('_')[-1])
    sc = float(ckpt_path.split('SC_')[0].split('_')[-1])

    lm  = RegressionModule.load_from_checkpoint(
        ckpt_path, model=mdl, optimizer=Adam, learning_rate=lr,
        reg_coeff_l1=0., reg_coeff_l2=rl2, weight_decay=wd, smoothness_coeff=sc,
    )
    mdl = lm.model.cpu().eval()

    out_scaler_T = data_module.out_scaler_T
    out_scaler_P = data_module.out_scaler_P
    in_scaler    = data_module.in_scaler

    # Raw (unscaled) test inputs and outputs
    raw_test_inputs = test_inputs.cpu().numpy()          # (n_test, 4)
    true_T = test_outputs_T.cpu().numpy()                # (n_test, O)
    true_P = test_outputs_P.cpu().numpy()                # (n_test, O)

    scaled_inputs = torch.tensor(in_scaler.transform(raw_test_inputs), dtype=torch.float32)

    with torch.no_grad():
        preds = mdl(scaled_inputs).numpy()               # (n_test, 2*O)

    pred_T = out_scaler_T.inverse_transform(preds[:, :O])
    pred_P = out_scaler_P.inverse_transform(preds[:, O:])

    rmse_T = np.sqrt(np.mean((pred_T - true_T) ** 2, axis=1))   # (n_test,)
    rmse_P = np.sqrt(np.mean((pred_P - true_P) ** 2, axis=1))   # (n_test,)

    if return_raw_inputs:
        return rmse_T, rmse_P, raw_test_inputs
    return rmse_T, rmse_P


#######################################################
#### Corner plot: performance across 4-D input space ##
#######################################################

def make_corner_plot(results, input_data, input_labels, metric='T',
                     title='Corner Plot', save_path=None):
    """
    Corner plot comparing multiple models.

    Parameters
    ----------
    results : list of dicts, each with keys 'label', 'color', 'rmse_T', 'rmse_P'
    input_data : ndarray (n_test, 4)  — raw (unscaled) test inputs
    input_labels : list of 4 strings
    metric : 'T' or 'P' — which RMSE to visualise
    title : overall figure title
    save_path : if given, figure is saved here
    """
    n_dims   = input_data.shape[1]
    n_models = len(results)
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(3.5 * n_dims, 3.5 * n_dims))

    scatter_kw = dict(s=8, alpha=0.6, linewidths=0)

    for row in range(n_dims):
        for col in range(n_dims):
            ax = axes[row, col]

            # ── upper-triangle: hide ──────────────────────────────────────
            if col > row:
                ax.set_visible(False)
                continue

            # ── diagonal: 1-D histogram of RMSE per input dimension ───────
            if col == row:
                for res in results:
                    vals = res[f'rmse_{metric}']
                    ax.hist(vals, bins=30, color=res['color'], alpha=0.5,
                            label=res['label'], density=True, histtype='stepfilled',
                            edgecolor='none')
                    ax.axvline(np.median(vals), color=res['color'],
                               linestyle='--', linewidth=1.2)
                ax.set_xlabel(f'RMSE {metric}')
                ax.set_ylabel('Density')
                if row == 0:
                    ax.legend(fontsize=7, loc='upper right')

            # ── lower-triangle: scatter coloured by RMSE ──────────────────
            else:
                x_data = input_data[:, col]
                y_data = input_data[:, row]

                # Determine shared colour scale across all models for this panel
                all_vals = np.concatenate([res[f'rmse_{metric}'] for res in results])
                vmin, vmax = np.percentile(all_vals, [5, 95])

                for k, res in enumerate(results):
                    vals = res[f'rmse_{metric}']
                    # Offset points slightly per model so overlapping models are visible
                    jitter = 0.01 * (k - (n_models - 1) / 2) * (x_data.max() - x_data.min())
                    sc = ax.scatter(
                        x_data + jitter, y_data,
                        c=vals, cmap='viridis',
                        vmin=vmin, vmax=vmax,
                        marker=['o', 's', '^', 'D'][k % 4],
                        **scatter_kw,
                        label=res['label']
                    )

                # One colourbar per panel
                cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
                cbar.set_label(f'RMSE {metric}', fontsize=7)
                cbar.ax.tick_params(labelsize=6)

            # Axis labels on edges only
            if row == n_dims - 1:
                ax.set_xlabel(input_labels[col], fontsize=9)
            else:
                ax.set_xticklabels([])

            if col == 0 and row != col:
                ax.set_ylabel(input_labels[row], fontsize=9)
            elif col == 0 and row == 0:
                pass  # diagonal top-left keeps its own label
            elif row != col:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.4)

    fig.suptitle(title, fontsize=13, y=1.01)

    # ── Legend patch for marker shapes (model identity) ───────────────────────
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=['o', 's', '^', 'D'][k % 4],
               color='w', markerfacecolor=res['color'],
               markersize=8, label=res['label'])
        for k, res in enumerate(results)
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9,
               bbox_to_anchor=(1.0, 1.0), title='Models')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


########################################################
#### Quantitative summary table across the 4-D space ##
########################################################

def make_performance_heatmap(results, input_data, input_labels, metric='T',
                              n_bins=6, save_path=None):
    """
    For each pair of input dimensions produce a 2-D heatmap of mean RMSE
    (binned), with one subplot column per model.

    results      : same list of dicts as corner plot
    input_data   : (n_test, 4) raw inputs
    input_labels : list of 4 strings
    metric       : 'T' or 'P'
    n_bins       : number of bins per axis
    save_path    : optional save path
    """
    pairs    = list(combinations(range(input_data.shape[1]), 2))
    n_pairs  = len(pairs)
    n_models = len(results)

    fig, axes = plt.subplots(n_pairs, n_models,
                             figsize=(4.5 * n_models, 4.0 * n_pairs),
                             squeeze=False)

    for p_idx, (xi, xj) in enumerate(pairs):
        x  = input_data[:, xi]
        y  = input_data[:, xj]
        x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
        y_edges = np.linspace(y.min(), y.max(), n_bins + 1)

        # Colour scale: consistent across all models for this pair
        all_hmaps = []
        for res in results:
            vals = res[f'rmse_{metric}']
            hmap = np.full((n_bins, n_bins), np.nan)
            for ix in range(n_bins):
                for iy in range(n_bins):
                    mask = (
                        (x >= x_edges[ix])  & (x < x_edges[ix + 1]) &
                        (y >= y_edges[iy])  & (y < y_edges[iy + 1])
                    )
                    if mask.sum() > 0:
                        hmap[iy, ix] = np.mean(vals[mask])
            all_hmaps.append(hmap)

        vmin = np.nanpercentile(np.concatenate([h.ravel() for h in all_hmaps]), 5)
        vmax = np.nanpercentile(np.concatenate([h.ravel() for h in all_hmaps]), 95)

        for m_idx, (res, hmap) in enumerate(zip(results, all_hmaps)):
            ax = axes[p_idx, m_idx]
            im = ax.imshow(
                hmap, origin='lower', aspect='auto',
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                cmap='viridis', vmin=vmin, vmax=vmax
            )
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
        print(f"Saved: {save_path}")
    return fig


##########################
#### Main comparison  ####
##########################

# --- 1. Evaluate every model ---
print("\n--- Evaluating models ---")
results = []
raw_test_inputs = None

for cfg in MODEL_CONFIGS:
    ckpt_path = model_save_path + cfg['ckpt']
    print(f"  Loading: {cfg['label']}  ({cfg['ckpt']})")
    rmse_T, rmse_P, raw_test_inputs = evaluate_model(
        ckpt_path, cfg['nn_width'], cfg['nn_depth'],
        data_module, return_raw_inputs=True
    )
    results.append({
        'label'  : cfg['label'],
        'color'  : cfg['color'],
        'rmse_T' : rmse_T,
        'rmse_P' : rmse_P,
    })
    print(f"    T: median RMSE = {np.median(rmse_T):.3f} K,  std = {np.std(rmse_T):.3f} K")
    print(f"    P: median RMSE = {np.median(rmse_P):.4f} log10 bar,  std = {np.std(rmse_P):.4f}")


# --- 2. Corner plots ---
print("\n--- Generating corner plots ---")

fig_corner_T = make_corner_plot(
    results, raw_test_inputs, INPUT_LABELS,
    metric='T',
    title='Corner Plot: Temperature RMSE across Input Space',
    save_path=plot_save_path + 'corner_T.pdf'
)

fig_corner_P = make_corner_plot(
    results, raw_test_inputs, INPUT_LABELS,
    metric='P',
    title=r'Corner Plot: Pressure RMSE (log$_{10}$ bar) across Input Space',
    save_path=plot_save_path + 'corner_P.pdf'
)


# --- 3. Heatmap comparison ---
print("\n--- Generating heatmap comparison plots ---")

fig_heat_T = make_performance_heatmap(
    results, raw_test_inputs, INPUT_LABELS,
    metric='T', n_bins=6,
    save_path=plot_save_path + 'heatmap_T.pdf'
)

fig_heat_P = make_performance_heatmap(
    results, raw_test_inputs, INPUT_LABELS,
    metric='P', n_bins=6,
    save_path=plot_save_path + 'heatmap_P.pdf'
)

plt.show()
print("\nAll plots saved.")