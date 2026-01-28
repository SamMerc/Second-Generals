#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD, Adam
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import scipy
from torchinfo import summary
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns




##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/home/merci228/WORK/2G_ML/'
#File containing surface temperature map
raw_ST_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/GP_ST_stand_norm/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/GP_ST_stand_norm/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
raw_inputs = raw_ST_data[:, :4] #has shape 46 x 72 = 3,312
raw_outputs = raw_ST_data[:, 5:]

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs.shape[1] #Number of outputs

## HYPER-PARAMETERS ##
#Defining partition of data used for 1. training and 2. testing
data_partition = [0.8, 0.2]

#Definine sub-partitiion for splitting NN dataset
sub_data_partitions = [0.7, 0.1, 0.2]

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
show_plot = False

#Number of nearest neighbors to choose
N_neigbors = 10

#Distance metric to use
distance_metric = 'euclidean' #options: 'euclidean', 'mahalanobis', 'logged_euclidean', 'logged_mahalanobis'

#Optimizer learning rate
learning_rate = 1e-3

#Regularization coefficient
regularization_coeff_l1 = 0.0
regularization_coeff_l2 = 0.0

#Smoothness constraint coefficient
smoothness_coeff = 0.0

#Weight decay 
weight_decay = 0.0

#Batch size 
batch_size = 200

#Number of epochs 
n_epochs = 2000

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
train_idx, test_idx = torch.utils.data.random_split(range(N), data_partition, generator=partition_rng)
## Generate the data partitions
### Training
train_inputs = raw_inputs[train_idx]
train_outputs = raw_outputs[train_idx]

### Testing
test_inputs = raw_inputs[test_idx]
test_outputs = raw_outputs[test_idx]


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

    query_labels = obs_labels + (Cyx @ scipy.linalg.pinv(Cxx) @ (query_features - obs_features))

    query_cov_labels = Cyy - Cyx @ scipy.linalg.pinv(Cxx) @ Cxy

    return query_labels, query_cov_labels







###################
#### Build CNN ####
###################
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_channels, generator=None):
        super(SimpleCNN, self).__init__()
        
        # Set seed if generator provided
        if generator is not None:
            torch.manual_seed(generator.initial_seed())

        # Direct CNN - no dimensionality reduction
        self.cnn = nn.Sequential(
            # Input: input_channels x 48 x 69
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: 32 x 48 x 69
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: 64 x 48 x 69
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: 64 x 48 x 69
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: 32 x 48 x 69
            
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()  # Output values between 0 and 1
            # Output: output_channels x 48 x 69
        )
    
    def forward(self, x):
        """
        Forward pass through the CNN.
        Args:
            x: Input tensor of shape (batch_size, input_channels, 46, 72)
        Returns:
            Output images of shape (batch_size, output_channels, 46, 72)
        """
        x = self.cnn(x)
        return x
    
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs, 
                 test_inputs, test_outputs, batch_size, rng, reshape_for_cnn=False, 
                 img_channels=1, img_height=None, img_width=None):
        super().__init__()

        #Store original shapes for reshaping 
        self.batch_size = batch_size
        self.rng = rng
        self.reshape_for_cnn = reshape_for_cnn
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        
        # Standardizing the output
        ## Create scaler
        out_scaler = StandardScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        out_scaler.fit(train_outputs.cpu().numpy())
        
        ## Transform all datasets and convert back to tensors
        train_outputs = torch.tensor(out_scaler.transform(train_outputs.cpu().numpy()), dtype=torch.float32)
        valid_outputs = torch.tensor(out_scaler.transform(valid_outputs.cpu().numpy()), dtype=torch.float32)
        test_outputs = torch.tensor(out_scaler.transform(test_outputs.cpu().numpy()), dtype=torch.float32)
        
        # Store the scaler if you need to inverse transform later
        self.out_scaler = out_scaler
        
        # Normalizing the input
        ## Create scaler
        in_scaler = MinMaxScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        in_scaler.fit(train_inputs.cpu().numpy())
        
        ## Transform all datasets and convert back to tensors
        train_inputs = torch.tensor(in_scaler.transform(train_inputs.cpu().numpy()), dtype=torch.float32)
        valid_inputs = torch.tensor(in_scaler.transform(valid_inputs.cpu().numpy()), dtype=torch.float32)
        test_inputs = torch.tensor(in_scaler.transform(test_inputs.cpu().numpy()), dtype=torch.float32)
        
        # Store the scaler if you need to inverse transform later
        self.in_scaler = in_scaler
        
        #Store the inputs
        self.train_inputs = train_inputs
        self.valid_inputs = valid_inputs
        self.test_inputs = test_inputs

        # Reshape data if needed for CNN
        if reshape_for_cnn:
            # Reshape inputs
            if img_height is None or img_width is None:
                # Auto-calculate square dimensions if not provided
                total_features = train_inputs.shape[1]
                img_size = int(np.sqrt(total_features / img_channels))
                if img_size * img_size * img_channels != total_features:
                    raise ValueError(f"Cannot reshape {total_features} features into square image. "
                                     f"Please provide img_height and img_width explicitly.")
                self.img_height = img_size
                self.img_width = img_size
            
            self.train_inputs = train_inputs.reshape(-1, img_channels, self.img_height, self.img_width)
            self.valid_inputs = valid_inputs.reshape(-1, img_channels, self.img_height, self.img_width)
            self.test_inputs = test_inputs.reshape(-1, img_channels, self.img_height, self.img_width)
            
            self.train_outputs = train_outputs.reshape(-1, img_channels, self.img_height, self.img_width)
            self.valid_outputs = valid_outputs.reshape(-1, img_channels, self.img_height, self.img_width)
            self.test_outputs = test_outputs.reshape(-1, img_channels, self.img_height, self.img_width)

        else:
            self.train_inputs = train_inputs
            self.valid_inputs = valid_inputs
            self.test_inputs = test_inputs
            self.train_outputs = train_outputs
            self.valid_outputs = valid_outputs
            self.test_outputs = test_outputs
    
    def train_dataloader(self):
        dataset = TensorDataset(self.train_inputs, self.train_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=self.rng)
    
    def val_dataloader(self):
        dataset = TensorDataset(self.valid_inputs, self.valid_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, generator=self.rng)
    
    def test_dataloader(self):
        dataset = TensorDataset(self.test_inputs, self.test_outputs)
        return DataLoader(dataset, batch_size=self.batch_size, generator=self.rng)

model = SimpleCNN(1,1,generator=NN_rng)
summary(model)




############################
#### Build training set ####
############################
#Initialize array to store residuals
train_NN_inputs = np.zeros(train_outputs.shape, dtype=float)

for query_idx, (query_input, query_output) in enumerate(zip(train_inputs, train_outputs)):

    #Calculate proximity of query point to observations
    # Euclidian distance
    if 'euclidean' in distance_metric:
        distances = np.sqrt( (query_input[0] - train_inputs[:,0])**2 + (query_input[1] - train_inputs[:,1])**2 + (query_input[2] - train_inputs[:,2])**2 + (query_input[3] - train_inputs[:,3])**2 )
    # Mahalanobis distance
    elif 'mahalanobis' in distance_metric:
        distances = np.sqrt( (query_input - np.mean(train_inputs, axis=0)).T @ scipy.linalg.inv((train_inputs @ train_inputs.T) / (train_inputs.shape[0] - 1)) @ (query_input - np.mean(train_inputs, axis=0)) )
    else:raise('Invalid distance metric')

    #Choose the N closest points
    N_closest_idx = np.argsort(distances)[:N_neigbors]
    prox_train_inputs = train_inputs[N_closest_idx, :]
    prox_train_outputs = train_outputs[N_closest_idx, :]
    
    #Find the query labels from nearest neigbours
    mean_test_output, cov_test_output = Sai_CGP(prox_train_inputs.T, prox_train_outputs.T, query_input.reshape((1, 4)).T)
    model_test_output = np.mean(mean_test_output,axis=1)
    model_test_output_err = np.sqrt(np.diag(cov_test_output))
    train_NN_inputs[query_idx, :] = model_test_output

    #Diagnostic plot
    if show_plot:

        #Convert shape
        plot_test_output = query_output.reshape((46, 72))
        plot_model_test_output = model_test_output.reshape((46, 72))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, layout='constrained')        
        # Compute global vmin/vmax across all datasets
        vmin = np.min(query_output)
        vmax = np.max(query_output)
        
        # Plot heatmaps
        ax1.set_title('Data')
        hm1 = sns.heatmap(plot_test_output, ax=ax1)
        cbar = hm1.collections[0].colorbar
        cbar.set_label('Temperature (K)')
        ax2.set_title('Model')
        hm2 = sns.heatmap(plot_model_test_output, ax=ax2)
        cbar = hm2.collections[0].colorbar
        cbar.set_label('Temperature (K)')

        ax2.set_xticks(np.linspace(0, 72, 5))
        ax2.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
        ax2.set_xlabel('Longitude (degrees)')
        
        # Fix latitude ticks
        for ax in [ax1, ax2]:
            ax.set_yticks(np.linspace(0, 46, 5))
            ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
            ax.set_ylabel('Latitude (degrees)')
        plt.suptitle(rf'H$_2$O : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg')
        plt.show()

# Split training dataset into training, validation, and testing, and format it correctly

## Retrieving indices of data partitions
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(train_inputs.shape[0]), sub_data_partitions, generator=partition_rng)

## Generate the data partitions
### Training
NN_train_inputs = torch.tensor(train_NN_inputs[train_idx], dtype=torch.float32)
NN_train_outputs = torch.tensor(train_outputs[train_idx], dtype=torch.float32)
### Validation
NN_valid_inputs = torch.tensor(train_NN_inputs[valid_idx], dtype=torch.float32)
NN_valid_outputs = torch.tensor(train_outputs[valid_idx], dtype=torch.float32)
### Testing
NN_test_og_inputs = torch.tensor(train_inputs[test_idx], dtype=torch.float32) 
NN_test_inputs = torch.tensor(train_NN_inputs[test_idx], dtype=torch.float32)
NN_test_outputs = torch.tensor(train_outputs[test_idx], dtype=torch.float32)

# Create DataModule
data_module = CustomDataModule(
    NN_train_inputs, NN_train_outputs,
    NN_valid_inputs, NN_valid_outputs,
    NN_test_inputs, NN_test_outputs,
    batch_size, batch_rng, reshape_for_cnn=True,
    img_channels=1, img_height=46, img_width=72
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

#Comparing GP predicted ST maps vs NN predicted ST maps vs true ST maps with residuals
substep = 100

# Get the scalers from data module
out_scaler = data_module.out_scaler
in_scaler = data_module.in_scaler

# Move model to CPU for inference to avoid GPU memory issues
model.cpu()
model.eval()

#Converting tensors to numpy arrays if this isn't already done
if (type(NN_test_outputs) != np.ndarray):
    NN_test_outputs = NN_test_outputs.cpu().numpy()

GP_res = np.zeros(NN_test_outputs.shape, dtype=float)
NN_res = np.zeros(NN_test_outputs.shape, dtype=float)

for NN_test_idx, (NN_test_input, GP_test_output, NN_test_output) in enumerate(zip(NN_test_og_inputs, NN_test_inputs, NN_test_outputs)):

    # Flatten to 2D for the scaler: (1 sample, 3312 features)
    GP_test_output_np = GP_test_output.cpu().numpy().reshape(1, -1)

    # Scale the input
    scaled_input = in_scaler.transform(GP_test_output_np)

    # If your model expects 4D input, reshape back
    # Otherwise, keep it as 2D if that's what the model expects
    model_input = torch.tensor(scaled_input, dtype=torch.float32).reshape(1, 1, 46, 72)

    # Get prediction
    NN_pred_output_scaled = model(model_input).detach().numpy().reshape(3312)

    # Inverse transform to get original scale
    NN_pred_output = out_scaler.inverse_transform(NN_pred_output_scaled.reshape(1, -1)).flatten()

    #Convert to numpy
    NN_test_input = NN_test_input.cpu().numpy()

    #Storing residuals 
    GP_res[NN_test_idx, :] = GP_test_output - NN_test_output
    NN_res[NN_test_idx, :] = NN_pred_output - NN_test_output

    #Plotting
    if (NN_test_idx % substep == 0):

        #Convert shape
        plot_test_output = NN_test_output.reshape((46, 72))
        plot_NN_test_output = NN_pred_output.reshape((46, 72))
        plot_GP_test_output = GP_test_output.reshape((46, 72))
        plot_res_GP = GP_res[NN_test_idx, :].reshape((46, 72))
        plot_res_NN = NN_res[NN_test_idx, :].reshape((46, 72))
        
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 8), sharex=True, layout='constrained')        
        
        # Compute global vmin/vmax across all datasets
        vmin = np.min(NN_test_output)
        vmax = np.max(NN_test_output)
        
        # Plot heatmaps
        ax1.set_title('Data')
        hm1 = sns.heatmap(plot_test_output, ax=ax1)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm1.collections[0].colorbar
        cbar.set_label('Temperature (K)')
        ax2.set_title('GP Model')
        hm2 = sns.heatmap(plot_GP_test_output, ax=ax2)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm2.collections[0].colorbar
        cbar.set_label('Temperature (K)')
        ax3.set_title('NN Model')
        hm3 = sns.heatmap(plot_NN_test_output, ax=ax3)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm3.collections[0].colorbar
        cbar.set_label('Temperature (K)')
        ax4.set_title('GP Residuals')
        hm4 = sns.heatmap(plot_res_GP, ax=ax4)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm4.collections[0].colorbar
        cbar.set_label('Temperature (K)')
        ax5.set_title('NN Residuals')
        hm5 = sns.heatmap(plot_res_NN, ax=ax5)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm5.collections[0].colorbar
        cbar.set_label('Temperature (K)')
        # Shared colorbar (use the last heatmap's mappable)
        # cbar = fig.colorbar(hm3.get_children()[0], ax=[ax1, ax2, ax3], location='right')
        # cbar.set_label("Temperature")
        # Fix longitude ticks
        ax5.set_xticks(np.linspace(0, 72, 5))
        ax5.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
        ax5.set_xlabel('Longitude (degrees)')
        # Fix latitude ticks
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_yticks(np.linspace(0, 46, 5))
            ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
            ax.set_ylabel('Latitude (degrees)')
        plt.suptitle(rf'H$_2$ : {NN_test_input[0]} bar, CO$_2$ : {NN_test_input[1]} bar, LoD : {NN_test_input[2]:.0f} days, Obliquity : {NN_test_input[3]} deg')

        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{NN_test_idx}.pdf')
       