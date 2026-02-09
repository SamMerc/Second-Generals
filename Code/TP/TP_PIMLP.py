#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler





##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
##Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing temperature values
raw_T_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/PIMLP/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/PIMLP/'
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
#Defining partition of data used for 1. training, 2. validation and 3. testing
data_partitions = [0.7, 0.1, 0.2]

#Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 1
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")
torch.set_default_device(device)

#Defining the noise seed for the random partitioning of the training data
partition_seed = 4
partition_rng = torch.Generator(device=device)
partition_rng.manual_seed(partition_seed)

#Defining the noise seed for the generating of batches from the partitioned data
batch_seed = 5
batch_rng = torch.Generator(device=device)
batch_rng.manual_seed(batch_seed)

#Defining the noise seed for the neural network initialization
NN_seed = 6
NN_rng = torch.Generator(device=device)
NN_rng.manual_seed(NN_seed)

#Neural network width and depth
nn_width = 102
nn_depth = 5

#Optimizer learning rate
learning_rate = 1e-5

#Regularization coefficient
regularization_coeff_l1 = 0.0
regularization_coeff_l2 = 0.0

#Weight decay 
weight_decay = 0.0

#Batch size 
batch_size = 200

#Number of epochs 
n_epochs = 10000

#Mode for optimization
run_mode = 'use'



################################################################
#### Define parametric TP model (Madhusudhan & Seager 2009) ####
################################################################

def TP_model(params, pressures):
    """
    Parametric Temperature-Pressure profile model based on Madhusudhan & Seager (2009).
    
    Parameters:
    - params: Array of parameters.
    - pressures: Array of pressure values.
    
    Returns:
    - temperatures: Array of temperatues values given the input pressure values values following the specified model.
    """

    #Extract parameters
    T0, P0, P1, P2, P3, alpha1, alpha2 = params

    #Fix beta coefficients
    beta1 = 0.5
    beta2 = 0.5

    #Define T2 and T3 based on boundary conditions
    T2 = T0 + (np.log(P1/P0) / alpha1) ** (1/beta1) - (np.log(P1/P2) / alpha2) ** (1/beta2)
    T3 = T2 + (np.log(P3/P2) / alpha2) ** (1/beta2)

    # Initialize pressure tensor
    temperatures = np.zeros(pressures.shape, dtype=float)
    
    #Populate the layers in the pressure profile
    
    # Layer 1: P0 < P < P1
    P_layer_1 = (pressures > P0) & (pressures < P1)
    func_layer_1 = lambda P : T0 + (np.log(P/P0) / alpha1) ** (1/beta1)
    temperatures[P_layer_1] = func_layer_1(pressures[P_layer_1])

    # Layer 2: P1 < P < P3
    P_layer_2 = (pressures >= P1) & (pressures < P3)
    func_layer_2 = lambda P : T2 + (np.log(P/P2) / alpha2) ** (1/beta2)
    temperatures[P_layer_2] = func_layer_2(pressures[P_layer_2])

    # Layer 3: P > P3
    P_layer_3 = (pressures >= P3)
    func_layer_3 = lambda P : T3
    temperatures[P_layer_3] = func_layer_3(pressures[P_layer_3])

    return temperatures, ((P0, T0), (P1, func_layer_1(P1)), (P2, T2), (P3, T3))


###############################################
#### Example fit on one of the TP profiles ####
###############################################

from lmfit import minimize, Parameters

data_T = raw_outputs_T[6]
data_P = 10**raw_outputs_P[6]
params = Parameters()
params.add('T0', 
           value=data_T.min() + 10,
           min=data_T.min() - 50,
           max=data_T.max() + 50)

P0_init = data_P.min() * 2
params.add('P0', 
           value=P0_init,
           min=data_P.min() * 0.5,
           max=data_P.min() * 100)

P1_init = data_P.min() * 100
params.add('P1', 
           value=P1_init,
           min=P0_init * 10,        
           max=data_P.max() * 0.1)

P2_init = data_P.max() * 0.1
params.add('P2', 
           value=P2_init,
           min=P1_init * 10,        
           max=data_P.max() * 0.5)

params.add('P3', 
           value=data_P.max() * 2,   
           min=P2_init * 2,          
           max=data_P.max() * 100)

params.add('alpha1', value=0.2, min=0.01, max=2.0)
params.add('alpha2', value=0.2, min=0.01, max=2.0)

def residual(params, pressures, data_T):
    param_values = (params['T0'], params['P0'], params['P1'], params['P2'], params['P3'], params['alpha1'], params['alpha2'])
    model_T, _ = TP_model(param_values, pressures)
    return model_T - data_T

result = minimize(residual, params, args=(10**raw_outputs_P[6], raw_outputs_T[6]))
print(result.params)
bestfit = (result.params['T0'].value, result.params['P0'].value, result.params['P1'].value, result.params['P2'].value, result.params['P3'].value, result.params['alpha1'].value, result.params['alpha2'].value)

model_pressures = np.logspace(raw_outputs_P[6][0], raw_outputs_P[6][-1], 10000)
temperature_profile, points = TP_model(bestfit, model_pressures)

plt.figure(figsize=(6, 8))
plt.plot(raw_outputs_T[6], raw_outputs_P[6], label='Truth')
plt.plot(temperature_profile, np.log10(model_pressures), label='Model')
plt.gca().invert_yaxis()
plt.xlabel('Temperature (K)')
plt.ylabel(r'log$_{10}$ Pressure (bar)')
plt.title('Sample Temperature-Pressure Profile')
plt.legend()
plt.grid()
plt.show()