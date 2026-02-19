#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, SGD
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import scipy
from torchinfo import summary
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm



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
model_save_path = base_dir+'Model_Storage/GP_TESTINGSTUFF/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/GP_TESTINGSTUFF/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
raw_inputs = raw_T_data[:, :4]
GP_raw_inputs = raw_inputs - np.mean(raw_inputs)
raw_outputs_T = raw_T_data[:, 5:]
GP_raw_outputs_T = raw_outputs_T - np.mean(raw_outputs_T)
raw_outputs_P = raw_P_data[:, 5:]
#Convert raw outputs to log10 scale so we don't have to deal with it later
raw_outputs_P = np.log10(raw_outputs_P/1000)
GP_raw_outputs_P = raw_outputs_P - np.mean(raw_outputs_P)

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs_T.shape[1] #Number of outputs

## HYPER-PARAMETERS ##
#Definine partitiion for splitting NN dataset
data_partition = [0.7, 0.1, 0.2]

#Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 96
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")

#Defining the noise seed for the random partitioning of the training data
partition_seed = 4
partition_rng = torch.Generator()
partition_rng.manual_seed(partition_seed)

#Defining the noise seed for the generating of batches from the partitioned data
batch_seed = 5
batch_rng = torch.Generator()
batch_rng.manual_seed(batch_seed)

#Defining the noise seed for the neural network initialization
NN_seed = 6
NN_rng = torch.Generator()
NN_rng.manual_seed(NN_seed)

# Variable to show plots or not 
show_plot = True

#Number of nearest neighbors to choose
N_neigbors = np.linspace(5, 200, 5, dtype=int).tolist()

#Distance metric to use
distance_metric = 'euclidean' #options: 'euclidean', 'mahalanobis', 'logged_euclidean', 'logged_mahalanobis'

#Neural network width and depth
nn_width = 102
nn_depth = 5

#Optimizer learning rate
learning_rate = 1e-3

#Regularization coefficient
regularization_coeff_l1 = 0.0
regularization_coeff_l2 = 0.0

#Smoothness constraint coefficient
smoothness_coeff = 0.001

#Weight decay 
weight_decay = 0.0

#Batch size 
batch_size = 200

#Number of epochs 
n_epochs = 10000

#Mode for optimization
run_mode = 'use'

#Convert raw inputs for H2 and CO2 pressures to log10 scale so don't have to deal with it later
if 'logged' in distance_metric:
    raw_inputs[:, 0] = np.log10(raw_inputs[:, 0]) #H2
    raw_inputs[:, 1] = np.log10(raw_inputs[:, 1]) #CO2







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
    Cxx += 1e-6 * np.eye(Cxx.shape[0]) 

    query_labels = obs_labels + (Cyx @ scipy.linalg.pinv(Cxx) @ (query_features - obs_features))

    query_cov_labels = Cyy - Cyx @ scipy.linalg.pinv(Cxx) @ Cxy

    return query_labels, query_cov_labels







###################
#### Build MLP ####
###################
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, generator=None):
        super().__init__()
        # Set seed if generator provided
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
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

model = NeuralNetwork(2*O, nn_width, 2*O, nn_depth, generator=NN_rng)
summary(model)




############################
#### Build training set ####
############################

if type(N_neigbors)==int:
    print('BUILDING GP TRAINING SET')
    N_neighbor = N_neigbors
    # Initialize array to store NN inputs / GP outputs
    NN_inputs_T = np.zeros(raw_outputs_T.shape, dtype=float)
    NN_inputs_P = np.zeros(raw_outputs_P.shape, dtype=float)

    for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(GP_raw_inputs, GP_raw_outputs_T, GP_raw_outputs_P)):

        #Calculate proximity of query point to observations
        distances = np.sqrt( (query_input[0] - raw_inputs[:,0])**2 + (query_input[1] - raw_inputs[:,1])**2 + (query_input[2] - raw_inputs[:,2])**2 + (query_input[3] - raw_inputs[:,3])**2 )

        #Choose the N closest points
        # skip the first point since it corresponds to the query point itself
        N_closest_idx = np.argsort(distances)[1:N_neighbor+1]
        prox_train_inputs = raw_inputs[N_closest_idx, :]
        prox_train_outputs_T = raw_outputs_T[N_closest_idx, :]
        prox_train_outputs_P = raw_outputs_P[N_closest_idx, :]

        #Find the query labels from nearest neigbours
        mean_test_output, cov_test_output = Sai_CGP(prox_train_inputs.T, np.concat((prox_train_outputs_T, prox_train_outputs_P), axis=1).T, query_input.reshape((1, 4)).T)
        model_test_output_T = np.mean(mean_test_output[:O],axis=1)
        model_test_output_P = np.mean(mean_test_output[O:],axis=1)
        model_test_output_Terr = np.sqrt(np.diag(cov_test_output))[:O]
        model_test_output_Perr = np.sqrt(np.diag(cov_test_output))[O:]
        NN_inputs_T[query_idx, :] = model_test_output_T
        NN_inputs_P[query_idx, :] = model_test_output_P

        #Diagnostic plot
        if show_plot:

            #Plot TP profiles
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
            
            #ax1 : prediction, truth and the neighbors
            for prox_idx in range(N_neighbor):
                ax1.plot(prox_train_outputs_T[prox_idx], prox_train_outputs_P[prox_idx], '.', linestyle='-', color='red', alpha=0.1, linewidth=2, zorder=1, label='Ensemble' if prox_idx==0 else None)
            
            #ax2 : prediction, truth and prediction errorbars in T
            ax2.errorbar(model_test_output_T, model_test_output_P, xerr=model_test_output_Terr, fmt='.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=2, alpha=0.4)
            ax2.fill_betweenx(model_test_output_P, model_test_output_T-model_test_output_Terr, model_test_output_T+model_test_output_Terr, color='green', zorder=2, alpha=0.2)
            #ax3 : prediction, truth and prediction errorbars in P
            ax3.errorbar(model_test_output_T, model_test_output_P, yerr=model_test_output_Perr, fmt='.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=2, alpha=0.4)
            ax3.fill_between(model_test_output_T, model_test_output_P-model_test_output_Perr, model_test_output_P+model_test_output_Perr, color='green', zorder=2, alpha=0.2)

            for ax in [ax1, ax2, ax3]:
                ax.plot(model_test_output_T, model_test_output_P, '.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=3, label='Prediction')

                ax.plot(query_output_T, query_output_P, '.', linestyle='-', color='blue', linewidth=2, zorder=3, markersize=10, label='Truth')

                ax.invert_yaxis()
                
                if ax == ax1 : ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
                ax.set_xlabel('Temperature (K)')
                
                ax.grid()
                ax.legend()        

            plt.suptitle(rf'H$_2$ : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg')
            plt.subplots_adjust(wspace=0.2)
            plt.show()

elif type(N_neigbors) == list:
    print('DIAGNOSTIC PLOTTING')
    #bias estimator
    bias_estim_T = np.zeros(len(N_neigbors), dtype=float)
    bias_estim_P = np.zeros(len(N_neigbors), dtype=float)
    
    #variance estimator
    var_estim_T = np.zeros(len(N_neigbors), dtype=float)
    var_estim_P = np.zeros(len(N_neigbors), dtype=float)

    #Loop over possible N_Neighbors values
    for n_idx, N_neighbor in enumerate(tqdm(N_neigbors)):
        
        # Initialize array to store NN inputs / GP outputs
        NN_inputs_T = np.zeros(raw_outputs_T.shape, dtype=float)
        NN_inputs_P = np.zeros(raw_outputs_P.shape, dtype=float)
        NN_inputs_Terr = np.zeros(raw_outputs_T.shape, dtype=float)
        NN_inputs_Perr = np.zeros(raw_outputs_P.shape, dtype=float)

        for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(GP_raw_inputs, GP_raw_outputs_T, GP_raw_outputs_P)):

            #Calculate proximity of query point to observations
            distances = np.sqrt( (query_input[0] - raw_inputs[:,0])**2 + (query_input[1] - raw_inputs[:,1])**2 + (query_input[2] - raw_inputs[:,2])**2 + (query_input[3] - raw_inputs[:,3])**2 )

            #Choose the N closest points
            # skip the first point since it corresponds to the query point itself
            N_closest_idx = np.argsort(distances)[1:N_neighbor+1]
            prox_train_inputs = raw_inputs[N_closest_idx, :]
            prox_train_outputs_T = raw_outputs_T[N_closest_idx, :]
            prox_train_outputs_P = raw_outputs_P[N_closest_idx, :]

            #Find the query labels from nearest neigbours
            mean_test_output, cov_test_output = Sai_CGP(prox_train_inputs.T, np.concat((prox_train_outputs_T, prox_train_outputs_P), axis=1).T, query_input.reshape((1, 4)).T)
            
            model_test_output_T = np.mean(mean_test_output[:O],axis=1)
            model_test_output_P = np.mean(mean_test_output[O:],axis=1)
            model_test_output_Terr = np.sqrt(np.diag(cov_test_output))[:O]
            model_test_output_Perr = np.sqrt(np.diag(cov_test_output))[O:]
            
            NN_inputs_T[query_idx, :] = model_test_output_T
            NN_inputs_P[query_idx, :] = model_test_output_P
            NN_inputs_Terr[query_idx, :] = model_test_output_Terr
            NN_inputs_Perr[query_idx, :] = model_test_output_Perr

        # Calculate bias estimator (for T and P separately)
        bias_estim_T[n_idx] = np.sqrt(np.mean((NN_inputs_T - raw_outputs_T)**2))  # RMSE over all points & levels
        bias_estim_P[n_idx] = np.sqrt(np.mean((NN_inputs_P - raw_outputs_P)**2))

        # Calculate variance estimator (for T and P separately)
        var_estim_T[N_neigbors.index(N_neighbor)] = np.mean(NN_inputs_Terr**2)
        var_estim_P[n_idx] = np.mean(NN_inputs_Perr**2)

    #Combine bias estimators to get a bias for the full prediction
    # Normalize T and P bias estimators with the mean values to remove scales that could lead to one of the estimators biasing the total
    bias_estim = ((bias_estim_T / np.mean(raw_outputs_T)) + (bias_estim_P / np.mean(raw_outputs_P))) / 2.
    var_estim = (var_estim_T + var_estim_P) / 2.
    print(bias_estim_T, bias_estim_P, bias_estim, np.mean(raw_outputs_T), np.mean(raw_outputs_P))
    print(var_estim_T, var_estim_P, var_estim)
    
    #Plot the bias and variance estimators as a function of the number of neighbors 
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(12, 8))
    
    axes[0,0].plot(N_neigbors, bias_estim_T, markersize=10, linestyle='-', color='blue')
    axes[0,1].plot(N_neigbors, bias_estim_P, markersize=10, linestyle='-', color='blue')
    axes[0,2].plot(N_neigbors, bias_estim, markersize=10, linestyle='-', color='blue')
    
    axes[1,0].plot(N_neigbors, var_estim_T, markersize=10, linestyle='-', color='blue')
    axes[1,1].plot(N_neigbors, var_estim_P, markersize=10, linestyle='-', color='blue')
    axes[1,2].plot(N_neigbors, var_estim, markersize=10, linestyle='-', color='blue')
    
    titles_row0 = ['Bias T', 'Bias P', 'Bias Combined']
    titles_row1 = ['Var T', 'Var P', 'Var Combined']
    for j in range(3):
        axes[0,j].set_title(titles_row0[j])
        axes[1,j].set_title(titles_row1[j])
        axes[0,j].grid(True)
        axes[1,j].grid(True)
        axes[1,j].set_xlabel('N neighbors')
    
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()

    raise KeyboardInterrupt('Done diagnostic plotting. Re-run with fixed N_neighbors for model training.')

else:
    raise KeyboardInterrupt('Invalid number of neighbors variable. Should be a list of int type.')
# Split training dataset into training, validation, and testing, and format it correctly

## Retrieving indices of data partitions
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(N), data_partition, generator=partition_rng)

## Generate the data partitions
### Training
NN_train_inputs_T = torch.tensor(NN_inputs_T[train_idx], dtype=torch.float32)
NN_train_inputs_P = torch.tensor(NN_inputs_P[train_idx], dtype=torch.float32)
NN_train_outputs_T = torch.tensor(raw_outputs_T[train_idx], dtype=torch.float32)
NN_train_outputs_P = torch.tensor(raw_outputs_P[train_idx], dtype=torch.float32)
### Validation
NN_valid_inputs_T = torch.tensor(NN_inputs_T[valid_idx], dtype=torch.float32)
NN_valid_inputs_P = torch.tensor(NN_inputs_P[valid_idx], dtype=torch.float32)
NN_valid_outputs_T = torch.tensor(raw_outputs_T[valid_idx], dtype=torch.float32)
NN_valid_outputs_P = torch.tensor(raw_outputs_P[valid_idx], dtype=torch.float32)
### Testing
NN_test_inputs_T = torch.tensor(NN_inputs_T[test_idx], dtype=torch.float32)
NN_test_inputs_P = torch.tensor(NN_inputs_P[test_idx], dtype=torch.float32)
NN_test_outputs_T = torch.tensor(raw_outputs_T[test_idx], dtype=torch.float32)
NN_test_outputs_P = torch.tensor(raw_outputs_P[test_idx], dtype=torch.float32)

NN_test_og_inputs = torch.tensor(raw_inputs[test_idx], dtype=torch.float32) 

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
    batch_size, batch_rng
)






###################################
#### Define optimization block ####
###################################
# PyTorch Lightning Module
class RegressionModule(pl.LightningModule):
    def __init__(self, model, optimizer, learning_rate, weight_decay=0.0, reg_coeff_l1=0.0, reg_coeff_l2=0.0, smoothness_coeff=0.0):
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
        """
        Compute L1 and L2 regularization on model weights (parameters).
        """
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
        """
        Compute smoothness constraint: ||∇s|| where s is the model output.
        This penalizes rapid changes in output with respect to input.
        """
        if self.smoothness_coeff == 0:
            return torch.tensor(0., device=self.device)
        
        # Clone and enable gradient computation for inputs
        X_grad = X.clone().detach().requires_grad_(True)

        # Temporarily enable gradients
        with torch.enable_grad():
            # Recompute output with gradient tracking
            output_grad = self.model(X_grad)
            
            # Compute gradients of output with respect to input: ∂s/∂x
            grad_outputs = torch.ones_like(output_grad)
            gradients = torch.autograd.grad(
                outputs=output_grad,
                inputs=X_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Smoothness: ||∇s|| = L2 norm of gradient vector for each sample
            # Shape: gradients is (batch_size, input_dim)
            # We want: mean over batch of ||∇s|| for each sample
            smoothness_penalty = torch.mean(torch.norm(gradients, p=2, dim=1))
        
        return self.smoothness_coeff * smoothness_penalty

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        X, y = batch
        pred = self(X)
        
        # Base loss: ||y - s||
        loss = self.loss_fn(pred, y)
        
        # Add weight regularization (L1/L2 on network parameters)
        l1_penalty, l2_penalty = self.compute_weight_regularization()
        loss += l1_penalty + l2_penalty
        
        # Add smoothness constraint (gradient of output w.r.t. input)
        smoothness_penalty = self.compute_smoothness_constraint(X, pred)
        loss += smoothness_penalty

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
    optimizer=Adam,
    learning_rate=learning_rate,
    reg_coeff_l1=regularization_coeff_l1,
    reg_coeff_l2=regularization_coeff_l2,
    weight_decay=weight_decay,
    smoothness_coeff=smoothness_coeff,
)

# Setup logger
logger = CSVLogger(model_save_path+'logs', name='NeuralNetwork')

# Set all seeds for complete reproducibility
pl.seed_everything(NN_seed, workers=True)

# Create Trainer and train
trainer = Trainer(
    max_epochs=n_epochs,
    logger=logger,
    deterministic=True  # For reproducibility
)

if run_mode == 'use':
    
    trainer.fit(lightning_module, datamodule=data_module)
    
    # Save model (PyTorch Lightning style)
    trainer.save_checkpoint(model_save_path + f'{n_epochs}epochs_{weight_decay}WD_{regularization_coeff_l1+regularization_coeff_l2}RC_{smoothness_coeff}SC_{learning_rate}LR_{batch_size}BS.ckpt')
    
    print("Done!")
    
else:
    # Load model
    lightning_module = RegressionModule.load_from_checkpoint(
        model_save_path + f'{n_epochs}epochs_{weight_decay}WD_{regularization_coeff_l1+regularization_coeff_l2}RC_{smoothness_coeff}SC_{learning_rate}LR_{batch_size}BS.ckpt',
        model=model,
        optimizer=Adam,
    learning_rate=learning_rate,
    reg_coeff_l1=regularization_coeff_l1,
    reg_coeff_l2=regularization_coeff_l2,
    weight_decay=weight_decay,
    smoothness_coeff=smoothness_coeff,
    )
    print("Model loaded!")


#Testing model on test dataset
if run_mode == 'use':trainer.test(lightning_module, datamodule=data_module)

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
substep = 100

# Get the scalers from data module
out_scaler_T = data_module.out_scaler_T
out_scaler_P = data_module.out_scaler_P
in_scaler_T = data_module.in_scaler_T
in_scaler_P = data_module.in_scaler_P

# Move model to CPU for inference to avoid GPU memory issues
model.cpu()
model.eval()

#Converting tensors to numpy arrays if this isn't already done
if (type(NN_test_outputs_T) != np.ndarray):
    NN_test_outputs_T = NN_test_outputs_T.cpu().numpy()
    NN_test_outputs_P = NN_test_outputs_P.cpu().numpy()

GP_res_T = np.zeros(NN_test_outputs_P.shape, dtype=float)
GP_res_P = np.zeros(NN_test_outputs_P.shape, dtype=float)
NN_res_T = np.zeros(NN_test_outputs_P.shape, dtype=float)
NN_res_P = np.zeros(NN_test_outputs_P.shape, dtype=float)

for NN_test_idx, (NN_test_input, GP_test_output_T, GP_test_output_P, NN_test_output_T, NN_test_output_P) in enumerate(zip(NN_test_og_inputs, NN_test_inputs_T, NN_test_inputs_P, NN_test_outputs_T, NN_test_outputs_P)):

    #Retrieve prediction
    scaled_GP_test_output_T = torch.tensor(in_scaler_T.transform(GP_test_output_T.reshape(1, -1)), dtype=torch.float32)
    scaled_GP_test_output_P = torch.tensor(in_scaler_P.transform(GP_test_output_P.reshape(1, -1)), dtype=torch.float32)
    scaled_input = torch.cat([scaled_GP_test_output_T, scaled_GP_test_output_P], dim=1)

    NN_pred_output = model(torch.tensor(scaled_input, dtype=torch.float32)).detach().numpy()
    
    #Inverse scaling
    pred_T_scaled = NN_pred_output[:, :O]
    pred_P_scaled = NN_pred_output[:, O:]
    NN_pred_output_T = out_scaler_T.inverse_transform(pred_T_scaled.reshape(1, -1)).flatten()
    NN_pred_output_P = out_scaler_P.inverse_transform(pred_P_scaled.reshape(1, -1)).flatten()

    #Convert to numpy
    NN_test_input = NN_test_input.cpu().numpy()

    #Storing residuals 
    GP_res_T[NN_test_idx, :] = GP_test_output_T - NN_test_output_T
    GP_res_P[NN_test_idx, :] = GP_test_output_P - NN_test_output_P
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


#Plot residuals
fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2, 2, sharex=True, figsize=[12, 8])
ax1.plot(GP_res_T.T, alpha=0.1, color='green')
ax2.plot(GP_res_P.T, alpha=0.1, color='green')
ax3.plot(NN_res_T.T, alpha=0.1, color='blue')
ax4.plot(NN_res_P.T, alpha=0.1, color='blue')
for ax in [ax1, ax2, ax3, ax4]:
    ax.axhline(0, color='black', linestyle='dashed')
    ax.grid()
ax2.set_xlabel('Index')
ax4.set_xlabel('Index')
ax1.set_ylabel('Temperature')
ax2.set_ylabel('log$_{10}$ Pressure (bar)')
ax3.set_ylabel('Temperature')
ax4.set_ylabel('log$_{10}$ Pressure (bar)')
plt.subplots_adjust(hspace=0.1, bottom=0.25)

# Add statistics text at the bottom
stats_text = (
    f"--- GP Residuals ---\n"
    f"Temperature Residuals : Median = {np.median(GP_res_T):.2f} K, Std = {np.std(GP_res_T):.2f} K\n"
    f"Pressure Residuals : Median = {np.median(GP_res_P):.2f} $log_{{10}}$ bar, Std = {np.std(GP_res_P):.2f} $log_{{10}}$ bar\n"
    f"\n"
    f"--- NN Residuals ---\n"
    f"Temperature Residuals : Median = {np.median(NN_res_T):.2f} K, Std = {np.std(NN_res_T):.2f} K\n"
    f"Pressure Residuals : Median = {np.median(NN_res_P):.3f} $log_{{10}}$ bar, Std = {np.std(NN_res_P):.2f} $log_{{10}}$ bar"
)

fig.text(0.1, 0.05, stats_text, fontsize=10, family='monospace',
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(plot_save_path+f'/res_GP_NN.pdf', bbox_inches='tight')
plt.show()