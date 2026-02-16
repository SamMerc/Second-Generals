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
# Required keys : 'label', 'ckpt', 'seed'
# Optional keys : 'nn_width', 'nn_depth'  (fall back to defaults above)
# -----------------------------------------------------------------------
MODEL_CONFIGS = [
    {
        'label' : 'PNNA - Baseline',
        'ckpt'  : 'NN_fixedstand_nosmooth_noreg/100000epochs_0.0WD_0.0RC_0.0SC_0.001LR_200BS.ckpt',
        'seed'  : 6,
    },
    {
        'label' : 'PNNA - L2 Reg',
        'ckpt'  : 'NN_fixedstand_nosmooth_reg/100000epochs_0.0WD_0.01RC_0.0SC_0.001LR_200BS.ckpt',
        'seed'  : 6,
    },
    {
        'label' : 'PNNA - Smooth',
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
#### Helper: evaluate one model on test set ##
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


def evaluate_model(cfg, data_module):
    """
    Load a checkpoint and return full signed residual profiles on the test set.

    Parameters
    ----------
    cfg         : one entry from MODEL_CONFIGS
    data_module : fitted CustomDataModule (provides scalers)

    Returns
    -------
    res_T : ndarray (n_test, O)  — residuals pred - true, temperature in K
    res_P : ndarray (n_test, O)  — residuals pred - true, pressure in log10 bar
    """
    width     = cfg.get('nn_width', nn_width_default)
    depth     = cfg.get('nn_depth', nn_depth_default)
    ckpt_path = model_save_path + cfg['ckpt']
    hparams   = _parse_hparams_from_ckpt_name(ckpt_path)

    NN_rng = torch.Generator()
    NN_rng.manual_seed(cfg['seed'])

    mdl = NeuralNetwork(D, width, 2 * O, depth, generator=NN_rng)
    lm  = RegressionModule.load_from_checkpoint(
        ckpt_path,
        model=mdl,
        optimizer=Adam,
        **hparams,
        reg_coeff_l1=0.0,
    )
    mdl = lm.model.cpu().eval()

    scaled_inputs = torch.tensor(
        data_module.in_scaler.transform(raw_test_inputs), dtype=torch.float32
    )
    with torch.no_grad():
        preds = mdl(scaled_inputs).numpy()   # (n_test, 2*O)

    pred_T = data_module.out_scaler_T.inverse_transform(preds[:, :O])
    pred_P = data_module.out_scaler_P.inverse_transform(preds[:, O:])

    res_T = pred_T - true_T   # (n_test, O) signed residuals
    res_P = pred_P - true_P
    return res_T, res_P


##########################
#### Main comparison  ####
##########################

# # --- 1. Evaluate every model ---
print("--- Evaluating models ---")

#Loop over model configurations and retrieve the residuals on T and P
for cfg in MODEL_CONFIGS:
    
    print(f"  Loading: {cfg['label']}  ({cfg['ckpt']})")
    res_T, res_P = evaluate_model(cfg, data_module)

    # Compute RMSE from residuals for the printed summary
    rmse_T = np.sqrt(np.mean(res_T ** 2, axis=1)) # (n_test,)
    rmse_P = np.sqrt(np.mean(res_P ** 2, axis=1)) # (n_test,)

    fig, axes = plt.subplots(2, 4, figsize=(12, 10))

    for i, test_input in enumerate(test_inputs.T):
        for j, quantity in enumerate([rmse_T, rmse_P]):
            
            ax = axes[j, i]

            pearson=pearsonr(test_input, quantity).statistic
            spearman=spearmanr(test_input, quantity).statistic

            ax.plot(test_input, quantity, '.', label=f'Spearman:{spearman:.2f}, \n Pearson:{pearson:.2f}')

            if j==1:
                ax.set_xlabel(INPUT_LABELS[i])

            if i==0:
                if j==0:ax.set_ylabel('Temperature RMSE')
                else:ax.set_ylabel('Pressure RMSE')

            else:
                ax.sharey(axes[j,0])

            ax.legend()
    fig.tight_layout()
    plt.savefig(plot_save_path + f'RMSE_correlation_model_{cfg['label']}.pdf')
    plt.close()

