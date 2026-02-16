#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
import pytorch_lightning as pl
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.stats import pearsonr, spearmanr


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
plot_save_path  = base_dir + 'Plots/Model_Compare_Plots/TP/'
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
data_partitions  = [0.7, 0.1, 0.2]
partition_seed   = 4
batch_seed       = 5
nn_width_default = 102
nn_depth_default = 5
batch_size       = 200

# -----------------------------------------------------------------------
# MULTI-MODEL COMPARISON CONFIG
# Required keys : 'label', 'ckpt', 'seed', 'model_type'
# PNNA: model_type='pnna' (standard NN on raw 4-D inputs)
# GPNNA: model_type='gpnna', 'n_neighbors': int (GP → NN two-stage)
# Optional keys : 'nn_width', 'nn_depth'
# -----------------------------------------------------------------------
MODEL_CONFIGS = [
    #PNNA
    {
        'label'      : 'PNNA - Baseline',
        'ckpt'       : 'NN_fixedstand_nosmooth_noreg/100000epochs_0.0WD_0.0RC_0.0SC_0.001LR_200BS.ckpt',
        'seed'       : 6,
        'model_type' : 'pnna',
    },
    {
        'label'      : 'PNNA - L2 Reg',
        'ckpt'       : 'NN_fixedstand_nosmooth_reg/100000epochs_0.0WD_0.01RC_0.0SC_0.001LR_200BS.ckpt',
        'seed'       : 6,
        'model_type' : 'pnna',
    },
    {
        'label'      : 'PNNA - Smooth',
        'ckpt'       : 'NN_smooth/100000epochs_0.0WD_0.0RC_0.001SC_0.005LR_200BS.ckpt',
        'seed'       : 6,
        'model_type' : 'pnna',
    },
    # GPNNA
    {
        'label'       : 'GPNNA - Baseline',
        'ckpt'        : 'GP_fixedstand_nosmooth_noreg/100000epochs_0.0WD_0.0RC_0.0SC_0.001LR_200BS.ckpt',
        'seed'        : 6,
        'model_type'  : 'gpnna',
        'n_neighbors' : 10,
    },
    {
        'label'       : 'GPNNA - Smooth',
        'ckpt'        : 'GP_fixedstand_smooth_noreg/100000epochs_0.0WD_0.0RC_0.001SC_0.001LR_200BS.ckpt',
        'seed'        : 6,
        'model_type'  : 'gpnna',
        'n_neighbors' : 10,
    },
]

for cfg in MODEL_CONFIGS:
    full = model_save_path + cfg['ckpt']
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint not found:\n  {full}\n"
            "Check MODEL_CONFIGS and model_save_path."
        )
print("All checkpoints found.\n")


##############################################
#### Conditional Gaussian Process (for GPNNA)
##############################################

def Sai_CGP(obs_features, obs_labels, query_features):
    """
    Conditional Gaussian Process

    Inputs
    ------
    obs_features   : ndarray (D, N) — D-dim features of N ensemble points
    obs_labels     : ndarray (K, N) — K-dim labels of N ensemble points
    query_features : ndarray (D, 1) — D-dim features of query point

    Outputs
    -------
    query_labels     : ndarray (K, N) — updated K-dim labels
    query_cov_labels : ndarray (K, K) — covariance of ensemble labels
    """
    n = obs_features.shape[1]
    Cyx = (obs_labels @ obs_features.T) / (n - 1)
    Cxy = (obs_features @ obs_labels.T) / (n - 1)
    Cxx = (obs_features @ obs_features.T) / (n - 1)
    Cyy = (obs_labels @ obs_labels.T) / (n - 1)
    Cxx += 1e-8 * np.eye(Cxx.shape[0])

    query_labels     = obs_labels + (Cyx @ scipy.linalg.pinv(Cxx) @ (query_features - obs_features))
    query_cov_labels = Cyy - Cyx @ scipy.linalg.pinv(Cxx) @ Cxy

    return query_labels, query_cov_labels


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

# Raw (unscaled) arrays for evaluation
raw_test_inputs  = test_inputs.cpu().numpy()
raw_train_inputs = train_inputs.cpu().numpy()   # needed for GPNNA k-NN
raw_train_outputs_T = train_outputs_T.cpu().numpy()
raw_train_outputs_P = train_outputs_P.cpu().numpy()
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
        self.model            = model
        self.learning_rate    = learning_rate
        self.reg_coeff_l1     = reg_coeff_l1
        self.reg_coeff_l2     = reg_coeff_l2
        self.smoothness_coeff = smoothness_coeff
        self.weight_decay     = weight_decay
        self.loss_fn          = nn.MSELoss()
        self.optimizer_class  = optimizer

    def forward(self, x):
        return self.model(x)

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
#### Helper: parse hparams from checkpoint name
##############################################

def _parse_hparams_from_ckpt_name(ckpt_path):
    """
    Extract learning_rate, reg_coeff_l2, weight_decay, smoothness_coeff
    from the checkpoint filename convention:
      {epochs}epochs_{wd}WD_{rc}RC_{sc}SC_{lr}LR_{bs}BS.ckpt
    Operates on the basename only so subdirectory prefixes are ignored.
    """
    stem   = os.path.basename(ckpt_path).replace('.ckpt', '')
    lookup = {}
    for part in stem.split('_'):
        for tag in ('epochs', 'WD', 'RC', 'SC', 'LR', 'BS'):
            if part.endswith(tag):
                try:
                    lookup[tag] = float(part[:-len(tag)])
                except ValueError:
                    pass
    return {
        'learning_rate'   : lookup.get('LR', 1e-3),
        'reg_coeff_l2'    : lookup.get('RC', 0.0),
        'weight_decay'    : lookup.get('WD', 0.0),
        'smoothness_coeff': lookup.get('SC', 0.0),
    }


##############################################
#### Evaluate PNNA model (standard NN)    ##
##############################################

def evaluate_pnna(cfg, data_module):
    """
    Load a PNNA checkpoint and return signed residuals on the test set.
    PNNA: raw 4-D input → NN → T-P profile

    Returns
    -------
    res_T : ndarray (n_test, O)
    res_P : ndarray (n_test, O)
    """
    width     = cfg.get('nn_width', nn_width_default)
    depth     = cfg.get('nn_depth', nn_depth_default)
    ckpt_path = model_save_path + cfg['ckpt']
    hparams   = _parse_hparams_from_ckpt_name(ckpt_path)

    NN_rng = torch.Generator()
    NN_rng.manual_seed(cfg['seed'])

    mdl = NeuralNetwork(D, width, 2 * O, depth, generator=NN_rng)
    lm  = RegressionModule.load_from_checkpoint(
        ckpt_path, model=mdl, optimizer=Adam, **hparams, reg_coeff_l1=0.0,
    )
    mdl = lm.model.cpu().eval()

    scaled_inputs = torch.tensor(
        data_module.in_scaler.transform(raw_test_inputs), dtype=torch.float32
    )
    with torch.no_grad():
        preds = mdl(scaled_inputs).numpy()

    pred_T = data_module.out_scaler_T.inverse_transform(preds[:, :O])
    pred_P = data_module.out_scaler_P.inverse_transform(preds[:, O:])

    res_T = pred_T - true_T
    res_P = pred_P - true_P
    return res_T, res_P


##############################################
#### Evaluate GPNNA model (GP → NN)       ##
##############################################

class GPNNADataModule(pl.LightningDataModule):
    """
    Like CustomDataModule but uses separate T and P scalers for *both*
    inputs and outputs (GPNNA needs this because inputs are TP profiles).
    """
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs,
                 test_inputs, test_outputs, batch_size, rng):
        super().__init__()

        # Output scalers (same as before)
        out_scaler_T = StandardScaler()
        out_scaler_P = StandardScaler()
        out_scaler_T.fit(train_outputs[:, :O].cpu().numpy())
        out_scaler_P.fit(train_outputs[:, O:].cpu().numpy())

        # Input scalers — GPNNA inputs are also TP profiles (from GP)
        in_scaler_T = StandardScaler()
        in_scaler_P = StandardScaler()
        in_scaler_T.fit(train_inputs[:, :O].cpu().numpy())
        in_scaler_P.fit(train_inputs[:, O:].cpu().numpy())

        def scale_profile(profiles, scaler_T, scaler_P):
            T_s = torch.tensor(scaler_T.transform(profiles[:, :O].cpu().numpy()), dtype=torch.float32)
            P_s = torch.tensor(scaler_P.transform(profiles[:, O:].cpu().numpy()), dtype=torch.float32)
            return torch.cat([T_s, P_s], dim=1)

        self.out_scaler_T = out_scaler_T
        self.out_scaler_P = out_scaler_P
        self.in_scaler_T  = in_scaler_T
        self.in_scaler_P  = in_scaler_P

        self.train_inputs  = scale_profile(train_inputs,  in_scaler_T, in_scaler_P)
        self.train_outputs = scale_profile(train_outputs, out_scaler_T, out_scaler_P)
        self.valid_inputs  = scale_profile(valid_inputs,  in_scaler_T, in_scaler_P)
        self.valid_outputs = scale_profile(valid_outputs, out_scaler_T, out_scaler_P)
        self.test_inputs   = scale_profile(test_inputs,   in_scaler_T, in_scaler_P)
        self.test_outputs  = scale_profile(test_outputs,  out_scaler_T, out_scaler_P)
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


def evaluate_gpnna(cfg):
    """
    Load a GPNNA checkpoint and return signed residuals on the test set.
    GPNNA: raw 4-D input → GP (k-NN) → NN → T-P profile

    Returns
    -------
    res_T : ndarray (n_test, O)
    res_P : ndarray (n_test, O)
    """
    width       = cfg.get('nn_width', nn_width_default)
    depth       = cfg.get('nn_depth', nn_depth_default)
    n_neighbors = cfg.get('n_neighbors', 10)
    ckpt_path   = model_save_path + cfg['ckpt']
    hparams     = _parse_hparams_from_ckpt_name(ckpt_path)

    NN_rng = torch.Generator()
    NN_rng.manual_seed(cfg['seed'])

    # GPNNA model: input_dim = 2*O (T and P profiles), output_dim = 2*O
    mdl = NeuralNetwork(2 * O, width, 2 * O, depth, generator=NN_rng)
    lm  = RegressionModule.load_from_checkpoint(
        ckpt_path, model=mdl, optimizer=Adam, **hparams, reg_coeff_l1=0.0,
    )
    mdl = lm.model.cpu().eval()

    # Build dummy GPNNA scalers (we only need them for inverse_transform)
    # Use the *training* GP outputs to fit the input scalers.
    # For simplicity we'll recompute GP predictions on training set once.
    print("    [GPNNA] Generating GP predictions on training set to fit scalers...")
    gp_train_T = np.zeros((len(raw_train_inputs), O), dtype=float)
    gp_train_P = np.zeros((len(raw_train_inputs), O), dtype=float)
    for i, q_input in enumerate(raw_train_inputs):
        distances = np.linalg.norm(raw_train_inputs - q_input, axis=1)
        k_idx = np.argsort(distances)[:n_neighbors]
        mean_labels, _ = Sai_CGP(
            raw_train_inputs[k_idx].T,
            np.concatenate([raw_train_outputs_T[k_idx], raw_train_outputs_P[k_idx]], axis=1).T,
            q_input.reshape((D, 1))
        )
        gp_train_T[i] = np.mean(mean_labels[:O], axis=1)
        gp_train_P[i] = np.mean(mean_labels[O:], axis=1)

    in_scaler_T = StandardScaler()
    in_scaler_P = StandardScaler()
    in_scaler_T.fit(gp_train_T)
    in_scaler_P.fit(gp_train_P)

    out_scaler_T = StandardScaler()
    out_scaler_P = StandardScaler()
    out_scaler_T.fit(raw_train_outputs_T)
    out_scaler_P.fit(raw_train_outputs_P)

    # --- Run two-stage inference on test set ---
    print("    [GPNNA] Running two-stage inference (GP → NN) on test set...")
    res_T = np.zeros((len(raw_test_inputs), O), dtype=float)
    res_P = np.zeros((len(raw_test_inputs), O), dtype=float)

    for i, q_input in enumerate(raw_test_inputs):
        # Stage 1: GP prediction from k-NN in training set
        distances = np.linalg.norm(raw_train_inputs - q_input, axis=1)
        k_idx = np.argsort(distances)[:n_neighbors]
        mean_labels, _ = Sai_CGP(
            raw_train_inputs[k_idx].T,
            np.concatenate([raw_train_outputs_T[k_idx], raw_train_outputs_P[k_idx]], axis=1).T,
            q_input.reshape((D, 1))
        )
        gp_T = np.mean(mean_labels[:O], axis=1)
        gp_P = np.mean(mean_labels[O:], axis=1)

        # Stage 2: NN refinement
        gp_T_scaled = in_scaler_T.transform(gp_T.reshape(1, -1))
        gp_P_scaled = in_scaler_P.transform(gp_P.reshape(1, -1))
        nn_input    = torch.tensor(np.concatenate([gp_T_scaled, gp_P_scaled], axis=1), dtype=torch.float32)

        with torch.no_grad():
            nn_output = mdl(nn_input).numpy()

        pred_T = out_scaler_T.inverse_transform(nn_output[:, :O]).flatten()
        pred_P = out_scaler_P.inverse_transform(nn_output[:, O:]).flatten()

        res_T[i] = pred_T - true_T[i]
        res_P[i] = pred_P - true_P[i]

    return res_T, res_P


##########################
#### Main comparison  ####
##########################

# # --- 1. Evaluate every model ---
print("--- Evaluating models ---")

#Loop over model configurations and retrieve the residuals on T and P
for cfg in MODEL_CONFIGS:
    print(f"  Loading: {cfg['label']}  ({cfg['ckpt']})")

    model_type = cfg.get('model_type', 'pnna')  # default to pnna if not specified

    if model_type == 'pnna':
        res_T, res_P = evaluate_pnna(cfg, data_module)
    elif model_type == 'gpnna':
        res_T, res_P = evaluate_gpnna(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute per-sample RMSE
    rmse_T = np.sqrt(np.mean(res_T ** 2, axis=1))  # (n_test,)
    rmse_P = np.sqrt(np.mean(res_P ** 2, axis=1))

    # --- Generate diagnostic plot: RMSE vs input dimensions ---
    fig, axes = plt.subplots(2, 4, figsize=(12, 10))

    for i, test_input_dim in enumerate(raw_test_inputs.T):
        for j, (quantity, label) in enumerate([(rmse_T, 'Temperature RMSE'),
                                                 (rmse_P, 'Pressure RMSE')]):
            ax = axes[j, i]

            pearson_r  = pearsonr(test_input_dim, quantity).statistic
            spearman_r = spearmanr(test_input_dim, quantity).statistic

            ax.plot(test_input_dim, quantity, '.',
                    label=f'Spearman: {spearman_r:.2f}\nPearson: {pearson_r:.2f}')

            if i == 0:
                ax.set_ylabel(label)
            if j == 1:
                ax.set_xlabel(INPUT_LABELS[i])
            if i != 0:
                ax.sharey(axes[j, 0])

            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    fig.tight_layout()
    safe_label = cfg['label'].replace(' ', '_').replace('-', '').replace('/', '')
    plt.savefig(plot_save_path + f'RMSE_correlation_model_{safe_label}.pdf')
    plt.close()