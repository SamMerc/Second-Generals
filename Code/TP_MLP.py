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
model_save_path = base_dir+'Model_Storage/NN/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/NN/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
raw_inputs = raw_T_data[:, :4]
raw_outputs_T = raw_T_data[:, 5:]
raw_outputs_P = raw_P_data[:, 5:]

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = 2*raw_outputs_T.shape[1] #Number of outputs

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

# Variable to show plots or not 
show_plot = False

#Neural network width and depth
nn_width = 102
nn_depth = 5

#Optimizer learning rate
learning_rate = 1e-5

#Regularization coefficient
regularization_coeff = 0.0

#Weight decay 
weight_decay = 0.0

#Batch size 
batch_size = 200

#Number of epochs 
n_epochs = 10000

#Mode for optimization
run_mode = 'use'







###########################################
#### Partition data and build datasets ####
###########################################
# PyTorch Lightning DataModule
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs, test_inputs, test_outputs, batch_size, rng):
        super().__init__()

        # Standardizing the output
        ## Create scaler
        out_scaler = StandardScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        out_scaler.fit(train_outputs.numpy())
        
        ## Transform all datasets and convert back to tensors
        train_outputs = torch.tensor(out_scaler.transform(train_outputs.numpy()), dtype=torch.float32)
        valid_outputs = torch.tensor(out_scaler.transform(valid_outputs.numpy()), dtype=torch.float32)
        test_outputs = torch.tensor(out_scaler.transform(test_outputs.numpy()), dtype=torch.float32)
        
        # Store the scaler if you need to inverse transform later
        self.out_scaler = out_scaler
        
        # Normalizing the input
        ## Create scaler
        in_scaler = MinMaxScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        in_scaler.fit(train_inputs.numpy())
        
        ## Transform all datasets and convert back to tensors
        train_inputs = torch.tensor(in_scaler.transform(train_inputs.numpy()), dtype=torch.float32)
        valid_inputs = torch.tensor(in_scaler.transform(valid_inputs.numpy()), dtype=torch.float32)
        test_inputs = torch.tensor(in_scaler.transform(test_inputs.numpy()), dtype=torch.float32)
        
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

#Splitting the data 

## Retrieving indices of data partitions
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(N), data_partitions, generator=partition_rng)

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


model = NeuralNetwork(D, nn_width, 2*O, nn_depth, generator=NN_rng).to(device)
summary(model)












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

#Comparing predicted T-P profiles vs true T-P profiles with residuals
substep = 100

# Get the scalers from data module
out_scaler = data_module.out_scaler
in_scaler = data_module.in_scaler

#Converting tensors to numpy arrays if this isn't already done
if (type(test_outputs_T) != np.ndarray):
    test_outputs_T = test_outputs_T.numpy()
    test_outputs_P = test_outputs_P.numpy()

res_T = np.zeros(test_outputs_P.shape, dtype=float)
res_P = np.zeros(test_outputs_P.shape, dtype=float)

for test_idx, (test_input, test_output_T, test_output_P) in enumerate(zip(test_inputs, test_outputs_T, test_outputs_P)):

    #Convert to numpy and reshape
    test_input = test_input.numpy()

    #Retrieve prediction
    pred_output = model(torch.tensor(in_scaler.transform(test_input.reshape(1, -1)))).detach().numpy()
    
    # Inverse transform to get original scale
    pred_output_original = out_scaler.inverse_transform(pred_output.reshape(1, -1)).flatten()
    
    # Split back into T and P components
    pred_output_T = pred_output_original[:O]
    pred_output_P = pred_output_original[O:]

    #Storing residuals 
    res_T[test_idx, :] = pred_output_T - test_output_T
    res_P[test_idx, :] = pred_output_P - test_output_P
    #Plotting
    if (test_idx % substep == 0):
        fig, axs = plt.subplot_mosaic([['res_pressure', '.'],
                                       ['results', 'res_temperature']],
                              figsize=(8, 6),
                              width_ratios=(3, 1), height_ratios=(1, 3),
                              layout='constrained')        
        axs['results'].plot(test_output_T, test_output_P, '.', linestyle='-', color='blue', linewidth=2)
        axs['results'].plot(pred_output_T, pred_output_P, color='green', linewidth=2)
        axs['results'].invert_yaxis()
        axs['results'].set_ylabel(r'log$_{10}$ Pressure (bar)')
        axs['results'].set_xlabel('Temperature (K)')
        axs['results'].legend()
        axs['results'].grid()

        axs['res_temperature'].plot(res_T[test_idx, :], test_output_P, '.', linestyle='-', color='green', linewidth=2)
        axs['res_temperature'].set_xlabel('Residuals (K)')
        axs['res_temperature'].invert_yaxis()
        axs['res_temperature'].grid()
        axs['res_temperature'].axvline(0, color='black', linestyle='dashed', zorder=2)
        axs['res_temperature'].yaxis.tick_right()
        axs['res_temperature'].yaxis.set_label_position("right")
        axs['res_temperature'].sharey(axs['results'])

        axs['res_pressure'].plot(test_output_T, res_P[test_idx, :], '.', linestyle='-', color='green', linewidth=2)
        axs['res_pressure'].set_ylabel('Residuals (bar)')
        axs['res_pressure'].invert_yaxis()
        axs['res_pressure'].grid()
        axs['res_pressure'].axhline(0, color='black', linestyle='dashed', zorder=2)
        axs['res_pressure'].xaxis.tick_top()
        axs['res_pressure'].xaxis.set_label_position("top")
        axs['res_pressure'].sharex(axs['results'])

        plt.suptitle(rf'H$_2$ : {test_input[0]} bar, CO$_2$ : {test_input[1]} bar, LoD : {test_input[2]:.0f} days, Obliquity : {test_input[3]} deg')
        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{test_idx}.pdf')
    
    
print('\n','--- NN Residuals ---')
print(f'Temperature Residuals : Median = {np.median(res_T):.2f} K, Std = {np.std(res_T):.2f} K')
print(rf'Pressure Residuals : Median = {np.median(res_P):.3f} $log_{10}$ bar, Std = {np.std(res_P):.2f} $log_{10}$ bar')

#Plot residuals
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[12, 8])
ax1.plot(res_T.T, alpha=0.1, color='green')
ax2.plot(res_P.T, alpha=0.1, color='green')
for ax in [ax1, ax2]:
    ax.axhline(0, color='black', linestyle='dashed')
    ax.set_xlabel('Index')
    ax.grid()
ax1.set_ylabel('Temperature')
ax2.set_ylabel('log$_{10}$ Pressure (bar)')
plt.subplots_adjust(hspace=0.1, bottom=0.25)

# Add statistics text at the bottom
stats_text = (
    f"--- NN Residuals ---\n"
    f"Temperature Residuals : Median = {np.median(res_T):.2f} K, Std = {np.std(res_T):.2f} K\n"
    f"Pressure Residuals : Median = {np.median(res_P):.3f} $log_{{10}}$ bar, Std = {np.std(res_P):.2f} $log_{{10}}$ bar"
)

fig.text(0.1, 0.05, stats_text, fontsize=10, family='monospace',
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(plot_save_path+f'/res_NN.pdf', bbox_inches='tight')
plt.show()