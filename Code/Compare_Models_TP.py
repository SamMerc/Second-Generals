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


#------------#
#--- PNNA ---#
#------------#

###########################################
#### Partition data and build datasets ####
###########################################
class PNNA_CustomDataModule(pl.LightningDataModule):
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
        
        # Normalizing the input
        ## Create scaler
        in_scaler = StandardScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        in_scaler.fit(train_inputs.cpu().numpy())
        
        ## Transform all datasets and convert back to tensors
        train_inputs = torch.tensor(in_scaler.transform(train_inputs.cpu().numpy()), dtype=torch.float32)
        valid_inputs = torch.tensor(in_scaler.transform(valid_inputs.cpu().numpy()), dtype=torch.float32)
        test_inputs = torch.tensor(in_scaler.transform(test_inputs.cpu().numpy()), dtype=torch.float32)
        
        # Store the scaler if you need to inverse transform later
        self.in_scaler = in_scaler

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




##################
#### Build NN ####
##################
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, generator=None):
        super().__init__()
        layers = []
        # Set seed if generator provided
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
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


##############################################
#### Evaluate PNNA model (standard NN)    ##
##############################################

def evaluate_pnna(cfg, data_module, raw_test_inputs, true_T, true_P):
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
        preds = mdl(scaled_inputs).detach().numpy()

    pred_T = data_module.out_scaler_T.inverse_transform(preds[:, :O])
    pred_P = data_module.out_scaler_P.inverse_transform(preds[:, O:])

    res_T = pred_T - true_T
    res_P = pred_P - true_P
    return res_T, res_P


#-------------#
#--- GPNNA ---#
#-------------#

######################################
#### Conditional Gaussian Process ####
######################################

def Sai_CGP(obs_features, obs_labels, query_features):
    """
    Conditional Gaussian Process
    Inputs: 
        obs_features : ndarray (D, N)
            D-dimensional features of the N ensemble data points.
        obs_labels : ndarray (K, N)
            K-dimensional labels of the N ensemble data points.
        query_features : ndarray (D, 1)
            D-dimensional features of the query data point.
    Outputs:
        query_labels : ndarray (K, N)
            K-dimensional labels of the ensemble updated from the query point.
        query_cov_labels : ndarray (K, K)
            K-by-K covariance of the ensemble labels.
    """
    
    # Defining relevant covariance matrices
    ## Between feature and label of observation data
    Cyx = (obs_labels @ obs_features.T) / (obs_features.shape[0] - 1)
    ## Between label and feature of observation data
    Cxy = (obs_features @ obs_labels.T) / (obs_features.shape[0] - 1)
    ## Between feature and feature of observation data
    Cxx = (obs_features @ obs_features.T) / (obs_features.shape[0] - 1)
    ## Between label and label of observation data
    Cyy = (obs_labels @ obs_labels.T) / (obs_features.shape[0] - 1)
    ## Adding regularizer to avoid singularities
    Cxx += 1e-8 * np.eye(Cxx.shape[0]) 

    query_labels = obs_labels + (Cyx @ scipy.linalg.pinv(Cxx) @ (query_features - obs_features))

    query_cov_labels = Cyy - Cyx @ scipy.linalg.pinv(Cxx) @ Cxy

    return query_labels, query_cov_labels


###########################################
#### Partition data and build datasets ####
###########################################

class GPNNADataModule(pl.LightningDataModule):
    """
    Like CustomDataModule but uses separate T and P scalers for *both*
    inputs and outputs (GPNNA needs this because inputs are TP profiles).
    """
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


##############################################
#### Evaluate GPNNA model (GP → NN)       ##
##############################################

def evaluate_gpnna(cfg, data_module, raw_test_inputs_T, raw_test_inputs_P, true_T, true_P):
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

    scaled_inputs_T = torch.tensor(
        data_module.in_scaler_T.transform(raw_test_inputs_T), dtype=torch.float32
    )
    scaled_inputs_P = torch.tensor(
        data_module.in_scaler_P.transform(raw_test_inputs_P), dtype=torch.float32
    )
    scaled_inputs = torch.cat([scaled_inputs_T, scaled_inputs_P], dim=1)

    with torch.no_grad():
        preds = mdl(scaled_inputs).detach().numpy()

    pred_T = data_module.out_scaler_T.inverse_transform(preds[:, :O])
    pred_P = data_module.out_scaler_P.inverse_transform(preds[:, O:])

    res_T = pred_T - true_T
    res_P = pred_P - true_P

    return res_T, res_P


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



##########################
#### Main comparison  ####
##########################

# # --- 1. Evaluate every model ---
print("--- Evaluating models ---")

#Loop over model configurations and retrieve the residuals on T and P
for cfg in MODEL_CONFIGS:
    print(f"  Loading: {cfg['label']}  ({cfg['ckpt']})")

    model_type = cfg.get('model_type', 'pnna')  # default to pnna if not specified

    #Define RNGs
    partition_rng = torch.Generator()
    partition_rng.manual_seed(partition_seed)
    batch_rng = torch.Generator()
    batch_rng.manual_seed(batch_seed)
    
    if model_type == 'pnna':
        ## Split data (seeds must match training) ##
        train_idx, valid_idx, test_idx = torch.utils.data.random_split(
            range(N), data_partitions, generator=partition_rng
        )

        ## Generate the data partitions
        ### Training
        train_inputs = torch.tensor(raw_inputs[train_idx], dtype=torch.float32)
        train_outputs_T = torch.tensor(raw_outputs_T[train_idx], dtype=torch.float32)
        train_outputs_P = torch.tensor(raw_outputs_P[train_idx], dtype=torch.float32)
        ### Validation
        valid_inputs = torch.tensor(raw_inputs[valid_idx], dtype=torch.float32)
        valid_outputs_T = torch.tensor(raw_outputs_T[valid_idx], dtype=torch.float32)
        valid_outputs_P = torch.tensor(raw_outputs_P[valid_idx], dtype=torch.float32)
        ### Testing
        test_inputs = torch.tensor(raw_inputs[test_idx], dtype=torch.float32)
        test_outputs_T = torch.tensor(raw_outputs_T[test_idx], dtype=torch.float32)
        test_outputs_P = torch.tensor(raw_outputs_P[test_idx], dtype=torch.float32)

        ## Concatenating outputs
        train_outputs = torch.cat([
            train_outputs_T,
            train_outputs_P
        ], dim=1)

        valid_outputs = torch.cat([
            valid_outputs_T,
            valid_outputs_P
        ], dim=1)

        test_outputs = torch.cat([
            test_outputs_T,
            test_outputs_P
        ], dim=1)

        # Create DataModule
        data_module = PNNA_CustomDataModule(
            train_inputs, train_outputs,
            valid_inputs, valid_outputs,
            test_inputs, test_outputs,
            batch_size, batch_rng
        )

        # Raw (unscaled) arrays for evaluation
        raw_test_inputs  = test_inputs.cpu().numpy()
        true_T = test_outputs_T.cpu().numpy()
        true_P = test_outputs_P.cpu().numpy()
    
        res_T, res_P = evaluate_pnna(cfg, data_module, raw_test_inputs, true_T, true_P)

    elif model_type == 'gpnna':
        
        #######################################################
        #### Partition data into training and testing sets ####
        #######################################################
        ## Retrieving indices of data partitions
        train_idx, test_idx = torch.utils.data.random_split(range(N), [0.8, 0.2], generator=partition_rng)
        ## Generate the data partitions
        ### Training
        train_inputs = raw_inputs[train_idx]
        train_outputs_T = raw_outputs_T[train_idx]
        train_outputs_P = raw_outputs_P[train_idx]

        ### Testing
        test_inputs = raw_inputs[test_idx]
        test_outputs_T = raw_outputs_T[test_idx]
        test_outputs_P = raw_outputs_P[test_idx]


        ############################
        #### Build training set ####
        ############################
        #Initialize array to store inputs
        train_NN_inputs_T = np.zeros(train_outputs_T.shape, dtype=float)
        train_NN_inputs_P = np.zeros(train_outputs_P.shape, dtype=float)

        for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(train_inputs, train_outputs_T, train_outputs_P)):

            #Calculate proximity of query point to observations
            distances = np.sqrt( (query_input[0] - train_inputs[:,0])**2 + (query_input[1] - train_inputs[:,1])**2 + (query_input[2] - train_inputs[:,2])**2 + (query_input[3] - train_inputs[:,3])**2 )

            #Choose the N closest points
            N_closest_idx = np.argsort(distances)[:cfg['n_neighbors']]
            prox_train_inputs = train_inputs[N_closest_idx, :]
            prox_train_outputs_T = train_outputs_T[N_closest_idx, :]
            prox_train_outputs_P = train_outputs_P[N_closest_idx, :]
            
            #Find the query labels from nearest neigbours
            mean_test_output, cov_test_output = Sai_CGP(prox_train_inputs.T, np.concat((prox_train_outputs_T, prox_train_outputs_P), axis=1).T, query_input.reshape((1, 4)).T)
            model_test_output_T = np.mean(mean_test_output[:O],axis=1)
            model_test_output_P = np.mean(mean_test_output[O:],axis=1)
            model_test_output_Terr = np.sqrt(np.diag(cov_test_output))[:O]
            model_test_output_Perr = np.sqrt(np.diag(cov_test_output))[O:]
            train_NN_inputs_T[query_idx, :] = model_test_output_T
            train_NN_inputs_P[query_idx, :] = model_test_output_P


        # Split training dataset into training, validation, and testing, and format it correctly
        ## Retrieving indices of data partitions
        train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(train_inputs.shape[0]), data_partitions, generator=partition_rng)

        ## Generate the data partitions
        ### Training
        NN_train_inputs_T = torch.tensor(train_NN_inputs_T[train_idx], dtype=torch.float32)
        NN_train_inputs_P = torch.tensor(train_NN_inputs_P[train_idx], dtype=torch.float32)
        NN_train_outputs_T = torch.tensor(train_outputs_T[train_idx], dtype=torch.float32)
        NN_train_outputs_P = torch.tensor(train_outputs_P[train_idx], dtype=torch.float32)
        ### Validation
        NN_valid_inputs_T = torch.tensor(train_NN_inputs_T[valid_idx], dtype=torch.float32)
        NN_valid_inputs_P = torch.tensor(train_NN_inputs_P[valid_idx], dtype=torch.float32)
        NN_valid_outputs_T = torch.tensor(train_outputs_T[valid_idx], dtype=torch.float32)
        NN_valid_outputs_P = torch.tensor(train_outputs_P[valid_idx], dtype=torch.float32)
        ### Testing
        NN_test_og_inputs = torch.tensor(train_inputs[test_idx], dtype=torch.float32) 
        NN_test_inputs_T = torch.tensor(train_NN_inputs_T[test_idx], dtype=torch.float32)
        NN_test_inputs_P = torch.tensor(train_NN_inputs_P[test_idx], dtype=torch.float32)
        NN_test_outputs_T = torch.tensor(train_outputs_T[test_idx], dtype=torch.float32)
        NN_test_outputs_P = torch.tensor(train_outputs_P[test_idx], dtype=torch.float32)

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
        data_module = GPNNADataModule(
            NN_train_inputs, NN_train_outputs,
            NN_valid_inputs, NN_valid_outputs,
            NN_test_inputs, NN_test_outputs,
            batch_size, batch_rng
        )

        # Raw (unscaled) arrays for evaluation
        raw_test_inputs  = NN_test_og_inputs.cpu().numpy()
        raw_test_inputs_T  = NN_test_inputs_T.cpu().numpy()
        raw_test_inputs_P  = NN_test_inputs_P.cpu().numpy()
        true_T = NN_test_outputs_T.cpu().numpy()
        true_P = NN_test_outputs_P.cpu().numpy()

        res_T, res_P = evaluate_gpnna(cfg, data_module, raw_test_inputs_T, raw_test_inputs_P, true_T, true_P)

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