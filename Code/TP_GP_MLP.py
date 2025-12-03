#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import scipy
from torchinfo import summary
import pandas as pd





##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing temperature values
raw_T_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/GP/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/GP/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
raw_inputs = raw_T_data[:, :4]
raw_outputs_T = raw_T_data[:, 5:]
raw_outputs_P = raw_P_data[:, 5:]
#Convert raw outputs to log10 scale so we don't have to deal with it later
raw_outputs_P = np.log10(raw_outputs_P/1000)

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs_T.shape[1] #Number of outputs

## HYPER-PARAMETERS ##
#Defining partition of data used for 1. training and 2. testing
data_partition = [0.8, 0.2]

#Definine sub-partitiion for splitting NN dataset
sub_data_partitions = [0.7, 0.1, 0.2]

#Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 1
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")
torch.set_default_device(device)

#Defining the noise seed for the random partitioning of the training data
partition_seed = 4
rng = torch.Generator(device=device)
rng.manual_seed(partition_seed)

# Variable to show plots or not 
show_plot = False

#Number of nearest neighbors to choose
N_neigbors = 10

#Distance metric to use
distance_metric = 'euclidean' #options: 'euclidean', 'mahalanobis', 'logged_euclidean', 'logged_mahalanobis'

#Neural network width and depth
nn_width = 102
nn_depth = 5

#Optimizer learning rate
learning_rate = 1e-5

#Regularization coefficient
regularization_coeff = 1e-2

#Weight decay 
weight_decay = 0.0

#Batch size 
batch_size = 200

#Number of epochs 
n_epochs = 1000

#Mode for optimization
run_mode = 'use'

#Convert raw inputs for H2 and CO2 pressures to log10 scale so don't have to deal with it later
if 'logged' in distance_metric:
    raw_inputs[:, 0] = np.log10(raw_inputs[:, 0]) #H2
    raw_inputs[:, 1] = np.log10(raw_inputs[:, 1]) #CO2







#######################################################
#### Partition data into training and testing sets ####
#######################################################
## Retrieving indices of data partitions
train_idx, test_idx = torch.utils.data.random_split(range(N), data_partition, generator=rng)
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
#### Build ensemble CGP ####
############################
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

    query_labels = obs_labels + (Cyx @ scipy.linalg.inv(Cxx) @ (query_features - obs_features))

    query_cov_labels = Cyy - Cyx @ scipy.linalg.inv(Cxx) @ Cxy

    return query_labels, query_cov_labels







###################
#### Build MLP ####
###################
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super().__init__()
        layers = []
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
    
# PyTorch Lightning DataModule
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs, test_inputs, test_outputs, batch_size, rng):
        super().__init__()
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

model = NeuralNetwork(2*O, nn_width, 2*O, nn_depth).to(device)
summary(model)




############################
#### Build training set ####
############################
#Initialize array to store residuals
train_NN_inputs_T = np.zeros(train_outputs_T.shape, dtype=float)
train_NN_inputs_P = np.zeros(train_outputs_P.shape, dtype=float)

for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(train_inputs, train_outputs_T, train_outputs_P)):

    #Calculate proximity of query point to observations
    distances = np.sqrt( (query_input[0] - train_inputs[:,0])**2 + (query_input[1] - train_inputs[:,1])**2 + (query_input[2] - train_inputs[:,2])**2 + (query_input[3] - train_inputs[:,3])**2 )

    #Choose the N closest points
    N_closest_idx = np.argsort(distances)[:N_neigbors]
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

    #Diagnostic plot
    if show_plot:

        #Plot TP profiles
        fig, ax = plt.subplots(figsize=(8, 6))
        for prox_idx in range(N_neigbors):
            ax.plot(prox_train_outputs_T[prox_idx], prox_train_outputs_P[prox_idx], '.', linestyle='-', color='red', alpha=0.1, linewidth=2, zorder=1, label='Ensemble' if prox_idx==0 else None)
        ax.plot(model_test_output_T, model_test_output_P, '.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=2, label='Prediction')
        ax.plot(query_output_T, query_output_P, '.', linestyle='-', color='blue', linewidth=2, zorder=2, markersize=10, label='Truth')
        ax.invert_yaxis()
        ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
        ax.set_xlabel('Temperature (K)')
        ax.grid()
        ax.legend()        

        plt.suptitle(rf'H$_2$ : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()

# Split training dataset into training, validation, and testing, and format it correctly

## Retrieving indices of data partitions
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(train_inputs.shape[0]), sub_data_partitions, generator=rng)

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
data_module = CustomDataModule(
    NN_train_inputs, NN_train_outputs,
    NN_valid_inputs, NN_valid_outputs,
    NN_test_inputs, NN_test_outputs,
    batch_size, rng
)






###################################
#### Define optimization block ####
###################################
# PyTorch Lightning Module
class RegressionModule(pl.LightningModule):
    def __init__(self, model, optimizer, learning_rate, weight_decay=0.0, reg_coeff=0.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.reg_coeff = reg_coeff
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()
        self.optimizer_class = optimizer
    
    def compute_gradient_penalty(self, X):
        """
        Compute the gradient of model output with respect to input.
        Returns the L2 norm of the gradients as a regularization term.
        """
        if self.reg_coeff == 0:
            return torch.tensor(0., device=self.device)
        
        # Clone and enable gradient computation for inputs
        X_grad = X.clone().detach().requires_grad_(True)

        # Temporarily enable gradients (needed for validation/test steps)
        with torch.enable_grad():
            
            # Compute output (need to recompute to track gradients w.r.t. X)
            output = self.model(X_grad)
            
            # Compute gradients of output with respect to input
            grad_outputs = torch.ones_like(output)
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=X_grad,
                grad_outputs=grad_outputs,
                create_graph=True,  # Keep computation graph for backprop
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Compute L2 norm of gradients (squared)
            gradient_penalty = torch.mean(gradients ** 2)
        
        return self.reg_coeff * gradient_penalty
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        
        # Add gradient regularization
        grad_penalty = self.compute_gradient_penalty(X)
        loss += grad_penalty

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)

        # Log metrics
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)








######################
#### Run training ####
######################
# Create Lightning Module
lightning_module = RegressionModule(
    model=model,
    optimizer=SGD,
    learning_rate=learning_rate,
    reg_coeff=regularization_coeff,
    weight_decay=weight_decay
)

# Setup logger
logger = CSVLogger(model_save_path+'logs', name='NeuralNetwork')

# Create Trainer and train
trainer = Trainer(
    max_epochs=n_epochs,
    logger=logger,
    deterministic=True  # For reproducibility
)

if run_mode == 'use':
    
    trainer.fit(lightning_module, datamodule=data_module)
    
    # Save model (PyTorch Lightning style)
    trainer.save_checkpoint(model_save_path + f'{n_epochs}epochs_{regularization_coeff}WD_{regularization_coeff}RC_{learning_rate}LR_{batch_size}BS.ckpt')
    
    print("Done!")
    
else:
    # Load model
    lightning_module = RegressionModule.load_from_checkpoint(
        model_save_path + f'{n_epochs}epochs_{regularization_coeff}WD_{regularization_coeff}RC_{learning_rate}LR_{batch_size}BS.ckpt',
        model=model,
        optimizer=SGD,
    learning_rate=learning_rate,
    reg_coeff=regularization_coeff,
    weight_decay=weight_decay
    )
    print("Model loaded!")


#Testing model on test dataset
trainer.test(lightning_module, datamodule=data_module)

# --- Accessing Training History After Training ---
# Find the version directory (e.g., version_0, version_1, etc.)
log_dir = model_save_path+'logs/NeuralNetwork'
versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
latest_version = sorted(versions)[-1]  # Get the latest version
csv_path = os.path.join(log_dir, latest_version, 'metrics.csv')

# Read the metrics
metrics_df = pd.read_csv(csv_path)

# Extract losses per epoch
train_losses = metrics_df[metrics_df['train_loss_epoch'].notna()]['train_loss_epoch'].tolist()
eval_losses = metrics_df[metrics_df['valid_loss'].notna()]['valid_loss'].tolist()







##########################
#### Diagnostic plots ####
##########################
# Loss curves
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3, 1]}, figsize=(10, 6))

# Calculate number of batches per epoch
n_batches = len(train_losses) // n_epochs

# Create x-axis in terms of epochs (0 to n_epochs)
x_all = np.linspace(0, n_epochs, len(train_losses))
x_epoch = np.arange(n_epochs+1)

# Plot transparent background showing all batch losses
ax1.plot(x_all, train_losses, alpha=0.3, color='C0', linewidth=0.5)
ax1.plot(x_all, eval_losses, alpha=0.3, color='C1', linewidth=0.5)

# Plot solid lines showing epoch-level losses (every n_batches steps)
train_epoch = [train_losses[0]] + train_losses[n_batches-1::n_batches]  # Last batch of each epoch
eval_epoch = [eval_losses[0]] + eval_losses[n_batches-1::n_batches]
ax1.plot(x_epoch, train_epoch, label="Train", color='C0', linewidth=2, marker='o')
ax1.plot(x_epoch, eval_epoch, label="Validation", color='C1', linewidth=2, marker='o')

# Same for difference plot
diff_all = np.array(train_losses) - np.array(eval_losses)
diff_epoch = np.array(train_epoch) - np.array(eval_epoch)

ax2.plot(x_all, diff_all, alpha=0.3, color='C2', linewidth=0.5)
ax2.plot(x_epoch, diff_epoch, color='C2', linewidth=2, marker='o')

ax1.set_yscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax2.set_ylabel("Loss Diff.")
ax1.legend()
ax1.grid()
ax2.grid()
plt.subplots_adjust(hspace=0)
plt.savefig(plot_save_path+'/loss.pdf')

#Comparing GP predicted T-P profiles vs NN predicted T-P profiles vs true T-P profiles with residuals
substep = 1000

#Converting tensors to numpy arrays if this isn't already done
if (type(NN_test_outputs_T) != np.ndarray):
    NN_test_outputs_T = NN_test_outputs_T.numpy()
    NN_test_outputs_P = NN_test_outputs_P.numpy()

GP_res_T = np.zeros(NN_test_outputs_P.shape, dtype=float)
GP_res_P = np.zeros(NN_test_outputs_P.shape, dtype=float)
NN_res_T = np.zeros(NN_test_outputs_P.shape, dtype=float)
NN_res_P = np.zeros(NN_test_outputs_P.shape, dtype=float)

for NN_test_idx, (NN_test_input, GP_test_output_T, GP_test_output_P, NN_test_output_T, NN_test_output_P) in enumerate(zip(NN_test_og_inputs, NN_test_inputs_T, NN_test_inputs_P, NN_test_outputs_T, NN_test_outputs_P)):

    #Retrieve prediction
    NN_pred_output = model(torch.cat([GP_test_output_T,GP_test_output_P])).detach().numpy()
    NN_pred_output_T = NN_pred_output[:O]
    NN_pred_output_P = NN_pred_output[O:]

    #Convert to numpy
    NN_test_input = NN_test_input.numpy()

    #Storing residuals 
    GP_res_T[NN_test_idx, :] = GP_test_output_T.numpy() - NN_test_output_T
    GP_res_P[NN_test_idx, :] = GP_test_output_P.numpy() - NN_test_output_P
    NN_res_T[NN_test_idx, :] = NN_pred_output_T - NN_test_output_T
    NN_res_P[NN_test_idx, :] = NN_pred_output_P - NN_test_output_P

    #Plotting
    if (NN_test_idx % substep == 0):
        fig, axs = plt.subplot_mosaic([['res_pressure', '.'],
                                       ['results', 'res_temperature']],
                              figsize=(8, 6),
                              width_ratios=(3, 1), height_ratios=(1, 3),
                              layout='constrained')        
        axs['results'].plot(NN_test_output_T, NN_test_output_P, '.', linestyle='-', color='blue', linewidth=2, label='Truth')
        axs['results'].plot(NN_pred_output_T, NN_pred_output_P, color='green', linewidth=2, label='NN prediction')
        axs['results'].plot(GP_test_output_T, GP_test_output_P, color='red', linewidth=2, label='GP prediction')
        axs['results'].invert_yaxis()
        axs['results'].set_ylabel(r'log$_{10}$ Pressure (bar)')
        axs['results'].set_xlabel('Temperature (K)')
        axs['results'].legend()
        axs['results'].grid()

        axs['res_temperature'].plot(NN_res_T[NN_test_idx, :], NN_test_output_P, '.', linestyle='-', color='green', linewidth=2)
        axs['res_temperature'].plot(GP_res_T[NN_test_idx, :], NN_test_output_P, '.', linestyle='-', color='red', linewidth=2)
        axs['res_temperature'].set_xlabel('Residuals (K)')
        axs['res_temperature'].invert_yaxis()
        axs['res_temperature'].grid()
        axs['res_temperature'].axvline(0, color='black', linestyle='dashed', zorder=2)
        axs['res_temperature'].yaxis.tick_right()
        axs['res_temperature'].yaxis.set_label_position("right")
        axs['res_temperature'].sharey(axs['results'])

        axs['res_pressure'].plot(NN_test_output_T, NN_res_P[NN_test_idx, :], '.', linestyle='-', color='green', linewidth=2)
        axs['res_pressure'].plot(NN_test_output_T, GP_res_P[NN_test_idx, :], '.', linestyle='-', color='red', linewidth=2)
        axs['res_pressure'].set_ylabel('Residuals (bar)')
        axs['res_pressure'].invert_yaxis()
        axs['res_pressure'].grid()
        axs['res_pressure'].axhline(0, color='black', linestyle='dashed', zorder=2)
        axs['res_pressure'].xaxis.tick_top()
        axs['res_pressure'].xaxis.set_label_position("top")
        axs['res_pressure'].sharex(axs['results'])

        plt.suptitle(rf'H$_2$ : {NN_test_input[0]} bar, CO$_2$ : {NN_test_input[1]} bar, LoD : {NN_test_input[2]:.0f} days, Obliquity : {NN_test_input[3]} deg')
        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{NN_test_idx}.pdf')
    