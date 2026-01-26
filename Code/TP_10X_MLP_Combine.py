#############################
#### Importing libraries ####
#############################

import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pytorch_lightning.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler, MinMaxScaler

##########################
#### Hyper-parameters ####
##########################
##Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)

#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'

#File containing temperature values
raw_T_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')

#File containing pressure values
raw_P_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')

#Path to stored models
model_saved_path = base_dir+'Model_Storage/NN_10X/'

#Path to store meta-model
model_save_path = base_dir+'Model_Storage/NN_10X_Comb/'
check_and_make_dir(model_save_path)

#Path to store plots
plot_save_path = base_dir+'Plots/NN_10X_Comb/'
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
n_models = 10 #Number of models

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
NN_seed = 16
NN_rng = torch.Generator(device=device)
NN_rng.manual_seed(NN_seed)

# Whether to give combiner access to original inputs
with_orig_inputs = False

#Neural network width and depth
nn_width = 102
nn_depth = 5

#Optimizer learning rate
old_learning_rate = 1e-5
new_learning_rate = 1e-5

#Regularization coefficient
old_regularization_coeff_l1 = 0.0
old_regularization_coeff_l2 = 0.0
new_regularization_coeff_l1 = 0.0
new_regularization_coeff_l2 = 0.0

#Weight decay 
old_weight_decay = 0.0
new_weight_decay = 0.0

#Batch size 
old_batch_size = 200
new_batch_size = 200

#Number of epochs 
old_n_epochs = 10000
new_n_epochs = 10000

#Mode for optimization
run_mode = 'use'

########################################
#### Class and Function definitions ####
########################################

# PyTorch Lightning DataModule
class OriginalDataModule(pl.LightningDataModule):
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


class EnsembleDataModule(pl.LightningDataModule):
    """
    DataModule for training the ensemble combiner.
    
    The inputs here are the ensemble predictions (not the original 4 features).
    The targets are the outputs aka the TP profiles. 
    The original inputs are the original inputs used to train the 10 NNs previously. You can choose to provide these or not.
    """
    def __init__(self, train_ensemble_preds, train_targets,
                 valid_ensemble_preds, valid_targets,
                 test_ensemble_preds, test_targets,
                 batch_size, rng,
                 train_original_inputs=None,
                 valid_original_inputs=None,
                 test_original_inputs=None):
        super().__init__()
        

        # Standardizing the output
        ## Create scaler
        out_scaler = StandardScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        out_scaler.fit(train_targets.numpy())
        
        ## Transform all datasets and convert back to tensors
        train_targets = torch.tensor(out_scaler.transform(train_targets.numpy()), dtype=torch.float32)
        valid_targets = torch.tensor(out_scaler.transform(valid_targets.numpy()), dtype=torch.float32)
        test_targets = torch.tensor(out_scaler.transform(test_targets.numpy()), dtype=torch.float32)
        
        # Store the scaler if you need to inverse transform later
        self.out_scaler = out_scaler
        
        # Normalizing the input
        ## Create scaler
        in_scaler = MinMaxScaler()
        
        ## Fit scaler on training dataset (convert to numpy)
        in_scaler.fit(train_ensemble_preds.numpy())
        
        ## Transform all datasets and convert back to tensors
        train_ensemble_preds = torch.tensor(in_scaler.transform(train_ensemble_preds.numpy()), dtype=torch.float32)
        valid_ensemble_preds = torch.tensor(in_scaler.transform(valid_ensemble_preds.numpy()), dtype=torch.float32)
        test_ensemble_preds = torch.tensor(in_scaler.transform(test_ensemble_preds.numpy()), dtype=torch.float32)
        
        # Store the scaler if you need to inverse transform later
        self.in_scaler = in_scaler

        if train_original_inputs is not None:

            # Normalizing the original input
            ## Create scaler
            past_in_scaler = MinMaxScaler()
            
            ## Fit scaler on training dataset (convert to numpy)
            past_in_scaler.fit(train_original_inputs.numpy())
            
            ## Transform all datasets and convert back to tensors
            train_original_inputs = torch.tensor(past_in_scaler.transform(train_original_inputs.numpy()), dtype=torch.float32)
            valid_original_inputs = torch.tensor(past_in_scaler.transform(valid_original_inputs.numpy()), dtype=torch.float32)
            test_original_inputs = torch.tensor(past_in_scaler.transform(test_original_inputs.numpy()), dtype=torch.float32)
            
            # Store the scaler if you need to inverse transform later
            self.past_in_scaler = past_in_scaler

        self.train_ensemble_preds = train_ensemble_preds
        self.train_targets = train_targets
        self.valid_ensemble_preds = valid_ensemble_preds
        self.valid_targets = valid_targets
        self.test_ensemble_preds = test_ensemble_preds
        self.test_targets = test_targets
        
        # Store original inputs if we want to pass them to combiner too
        self.train_original_inputs = train_original_inputs
        self.valid_original_inputs = valid_original_inputs
        self.test_original_inputs = test_original_inputs
        
        self.batch_size = batch_size
        self.rng = rng
    
    def train_dataloader(self):
        if self.train_original_inputs is not None:
            dataset = TensorDataset(self.train_ensemble_preds, 
                                   self.train_original_inputs, 
                                   self.train_targets)
        else:
            dataset = TensorDataset(self.train_ensemble_preds, self.train_targets)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                         generator=self.rng)
    
    def val_dataloader(self):
        if self.valid_original_inputs is not None:
            dataset = TensorDataset(self.valid_ensemble_preds,
                                   self.valid_original_inputs,
                                   self.valid_targets)
        else:
            dataset = TensorDataset(self.valid_ensemble_preds, self.valid_targets)
        return DataLoader(dataset, batch_size=self.batch_size, generator=self.rng)
    
    def test_dataloader(self):
        if self.test_original_inputs is not None:
            dataset = TensorDataset(self.test_ensemble_preds,
                                   self.test_original_inputs,
                                   self.test_targets)
        else:
            dataset = TensorDataset(self.test_ensemble_preds, self.test_targets)
        return DataLoader(dataset, batch_size=self.batch_size, generator=self.rng)
    


class EnsembleCombiner(nn.Module):
    """
    Meta-learner that combines predictions from multiple neural networks.
    
    Args:
        n_models: Number of ensemble models
        output_dim: Dimension of each model's output (2*O in your case)
        depth: Depth of the neural network
        hidden_dim: Hidden layer size for the combiner
        include_inputs: Whether to also pass original inputs to combiner
        input_dim: Dimension of original inputs (only needed if include_inputs=True)
    """
    def __init__(self, n_models, output_dim, depth, hidden_dim, include_inputs=False, input_dim=None, generator=None):
        super().__init__()
        
        # Set seed if generator provided
        if generator is not None:
            torch.manual_seed(generator.initial_seed())

        self.n_models = n_models
        self.output_dim = output_dim
        self.include_inputs = include_inputs
        
        # Calculate input dimension for combiner
        combiner_input_dim = n_models * output_dim
        if include_inputs:
            assert input_dim is not None, "Must provide input_dim if include_inputs=True"
            combiner_input_dim += input_dim
        
        layers = []
        layers.append(nn.Linear(combiner_input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        # Pack all layers into a Sequential container
        self.combiner = nn.Sequential(*layers)
    
    def forward(self, ensemble_outputs, original_inputs=None):
        """
        Args:
            ensemble_outputs: Tensor of shape (batch_size, n_models, output_dim)
            original_inputs: Optional tensor of shape (batch_size, input_dim)
        
        Returns:
            Combined prediction of shape (batch_size, output_dim)
        """
        # Flatten ensemble outputs
        batch_size = ensemble_outputs.shape[0]
        flat_outputs = ensemble_outputs.reshape(batch_size, -1)
        
        # Optionally concatenate with original inputs
        if self.include_inputs and original_inputs is not None:
            combiner_input = torch.cat([flat_outputs, original_inputs], dim=1)
        else:
            combiner_input = flat_outputs
        
        # Pass through combiner network
        combined = self.combiner(combiner_input)
        return combined


class EnsembleWrapper:
    """
    Wrapper to load and use multiple trained models as an ensemble.
    """
    def __init__(self, model_paths, device='cpu'):
        """
        Args:
            model_paths: List of paths to saved model checkpoints
            device: Device to load models on
        """
        self.device = device
        self.models = []
        
        # Load all models
        for path in model_paths:
            # Load the lightning module which contains the model
            lightning_module = pl.LightningModule.load_from_checkpoint(
                path,
                map_location=device
            )
            model = lightning_module.model
            model.eval()  # Set to evaluation mode
            model.to(device)
            self.models.append(model)
        
        self.n_models = len(self.models)
    
    @torch.no_grad()
    def get_ensemble_predictions(self, inputs):
        """
        Get predictions from all ensemble models.
        
        Args:
            inputs: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, n_models, output_dim)
        """
        predictions = []
        for model in self.models:
            pred = model(inputs)
            predictions.append(pred)
        
        # Stack along new dimension
        return torch.stack(predictions, dim=1)


class EnsembleLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the ensemble combiner.
    """
    def __init__(self, ensemble_wrapper, optimizer, combiner_model, weight_decay=0.0, learning_rate=1e-4, reg_coeff_l1=0.0, reg_coeff_l2=0.0):
        super().__init__()
        self.ensemble_wrapper = ensemble_wrapper
        self.combiner = combiner_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reg_coeff_l1 = reg_coeff_l1
        self.reg_coeff_l2 = reg_coeff_l2
        self.loss_fn = nn.MSELoss()
        self.optimizer_class = optimizer
        
        # Freeze ensemble models (we only train the combiner)
        for model in self.ensemble_wrapper.models:
            for param in model.parameters():
                param.requires_grad = False
    
    def compute_gradient_penalty(self, ensemble_preds, original_inputs=None):
        """
        Compute the gradient of combiner output with respect to its inputs.
        Returns the L1 and L2 norm of the gradients as regularization terms.
        """
        if self.reg_coeff_l1 == 0 and self.reg_coeff_l2 == 0:
            return torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)
        
        # Clone and enable gradient computation for inputs
        ensemble_preds_grad = ensemble_preds.clone().detach().requires_grad_(True)
        original_inputs_grad = None
        if original_inputs is not None:
            original_inputs_grad = original_inputs.clone().detach().requires_grad_(True)
        
        # Temporarily enable gradients (needed for validation/test steps)
        with torch.enable_grad():
            # Compute output (need to recompute to track gradients w.r.t. inputs)
            output = self.combiner(ensemble_preds_grad, original_inputs_grad)
            
            # Compute gradients of output with respect to inputs
            grad_outputs = torch.ones_like(output)
            
            # Get gradients w.r.t. ensemble_preds
            inputs_to_grad = [ensemble_preds_grad]
            if original_inputs_grad is not None:
                inputs_to_grad.append(original_inputs_grad)
            
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=inputs_to_grad,
                grad_outputs=grad_outputs,
                create_graph=True,  # Keep computation graph for backprop
                retain_graph=True,
                only_inputs=True
            )
            
            # Combine gradients from both inputs
            if len(gradients) == 2:
                all_gradients = torch.cat([gradients[0].flatten(1), gradients[1].flatten(1)], dim=1)
            else:
                all_gradients = gradients[0]
            
            # Compute L1 and L2 norm of gradients
            gradient_penalty_l1 = torch.mean(all_gradients.abs())
            gradient_penalty_l2 = torch.mean(all_gradients ** 2)
        
        return self.reg_coeff_l1 * gradient_penalty_l1, self.reg_coeff_l2 * gradient_penalty_l2
    
    def forward(self, ensemble_preds, original_inputs=None):
        # Combine predictions with the meta-learner
        combined = self.combiner(ensemble_preds, original_inputs)
        return combined
    
    def training_step(self, batch):
        # Batch can have 2 or 3 elements depending on include_inputs
        if len(batch) == 3:
            ensemble_preds, original_inputs, y = batch
        else:
            ensemble_preds, y = batch
            original_inputs = None
        pred = self(ensemble_preds, original_inputs)
        loss = self.loss_fn(pred, y)
        
        # Add gradient regularization
        grad_penalty_l1, grad_penalty_l2 = self.compute_gradient_penalty(ensemble_preds, original_inputs)
        loss += grad_penalty_l1 + grad_penalty_l2

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        # Batch can have 2 or 3 elements depending on include_inputs
        if len(batch) == 3:
            ensemble_preds, original_inputs, y = batch
        else:
            ensemble_preds, y = batch
            original_inputs = None
        pred = self(ensemble_preds, original_inputs)
        loss = self.loss_fn(pred, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch):
        # Batch can have 2 or 3 elements depending on include_inputs
        if len(batch) == 3:
            ensemble_preds, original_inputs, y = batch
        else:
            ensemble_preds, y = batch
            original_inputs = None
        pred = self(ensemble_preds, original_inputs)
        loss = self.loss_fn(pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer_class(self.combiner.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


#######################
#### Usage Script #####
#######################

def prepare_ensemble_data(ensemble_wrapper, original_data_module, device='cpu'):
    """
    Generate predictions from all ensemble models to create training data for the combiner.
    
    Args:
        ensemble_wrapper: EnsembleWrapper containing all trained models
        original_data_module: Your original CustomDataModule with the raw inputs
        device: Device to run on
    
    Returns:
        Dictionary containing ensemble predictions and targets for train/val/test
    """
    for model in ensemble_wrapper.models:
        model.eval()   
         
    data = {}
    
    # Process each split
    for split_name, loader in [
        ('train', original_data_module.train_dataloader()),
        ('val', original_data_module.val_dataloader()),
        ('test', original_data_module.test_dataloader())
    ]:
        all_inputs = []
        all_ensemble_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                # Get predictions from all 10 models
                ensemble_preds = ensemble_wrapper.get_ensemble_predictions(batch_inputs)
                
                all_inputs.append(batch_inputs)
                all_ensemble_preds.append(ensemble_preds)
                all_targets.append(batch_targets)
        
        # Concatenate all batches
        data[f'{split_name}_inputs'] = torch.cat(all_inputs, dim=0)
        data[f'{split_name}_ensemble_preds'] = torch.cat(all_ensemble_preds, dim=0)
        data[f'{split_name}_targets'] = torch.cat(all_targets, dim=0)
    
    return data

#Splitting the original data 
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
orig_data_module = OriginalDataModule(
    train_inputs, train_outputs,
    valid_inputs, valid_outputs,
    test_inputs, test_outputs,
    old_batch_size, batch_rng
)

# Collect paths to all trained models
model_paths = []
for i in range(1, n_models + 1):
    model_path = model_saved_path + f'NeuralNetwork_{i}/{old_n_epochs}epochs_{old_weight_decay}WD_{old_regularization_coeff_l1+old_regularization_coeff_l2}RC_{old_learning_rate}LR_{old_batch_size}BS.ckpt'
    if os.path.exists(model_path):
        model_paths.append(model_path)
    else:
        print(f"Warning: Model {i} not found at {model_path}")

print(f"Found {len(model_paths)} trained models")

# Create ensemble wrapper
ensemble_wrapper = EnsembleWrapper(
    model_paths=model_paths,
    base_model_class=None,  # Not needed since we load from checkpoint
    device=device
)

#Generate dictionary to pass to new Data Module
prep_data = prepare_ensemble_data(ensemble_wrapper, orig_data_module, device)

#Build new Data Module
if with_orig_inputs:
        new_data_module = EnsembleDataModule(
        prep_data['train_ensemble_preds'], prep_data['train_targets'],
        prep_data['valid_ensemble_preds'], prep_data['valid_targets'],
        prep_data['test_ensemble_preds'], prep_data['test_targets'],
        new_batch_size, batch_rng,
        prep_data['train_inputs'], prep_data['valid_inputs'], prep_data['test_inputs'] 
    )
else:
    new_data_module = EnsembleDataModule(
        prep_data['train_ensemble_preds'], prep_data['train_targets'],
        prep_data['valid_ensemble_preds'], prep_data['valid_targets'],
        prep_data['test_ensemble_preds'], prep_data['test_targets'],
        new_batch_size, batch_rng,
    )

# 3. Create combiner network
combiner = EnsembleCombiner(
    n_models=len(model_paths),
    output_dim=2 * O,
    hidden_dim=nn_width,
    depth=nn_depth,
    include_inputs=with_orig_inputs,  
    input_dim=D,
    generator=NN_rng
).to(device)

# 4. Create Lightning module
ensemble_lightning = EnsembleLightningModule(
    ensemble_wrapper=ensemble_wrapper,
    combiner_model=combiner,
    learning_rate=new_learning_rate,
    optimizer=Adam, 
    weight_decay=new_weight_decay,
)

# 5. Train the combiner
#Setup logger
logger = CSVLogger(model_save_path + 'logs', name='NeuralNetwork')

# Set all seeds for complete reproducibility
pl.seed_everything(NN_seed, workers=True)

# Create Trainer and train
trainer = Trainer(
    max_epochs=new_n_epochs,  # Typically needs fewer epochs than base models
    logger=logger,
    deterministic=True
)

if run_mode == 'use':
    
    trainer.fit(ensemble_lightning, datamodule=new_data_module)
    
    # Save model (PyTorch Lightning style)
    trainer.save_checkpoint(model_save_path + f'{new_n_epochs}epochs_{new_weight_decay}WD_{new_regularization_coeff_l1+new_regularization_coeff_l2}RC_{new_learning_rate}LR_{new_batch_size}BS.ckpt')
    
    print("Done!")
    
else:
    # Load model
    lightning_module = EnsembleLightningModule.load_from_checkpoint(
        model_save_path + f'{new_n_epochs}epochs_{new_weight_decay}WD_{new_regularization_coeff_l1+new_regularization_coeff_l2}RC_{new_learning_rate}LR_{new_batch_size}BS.ckpt',
        ensemble_wrapper=ensemble_wrapper,
        combiner_model=combiner,
        learning_rate=new_learning_rate,
        optimizer=Adam, 
        weight_decay=new_weight_decay
    )
    print("Model loaded!")

#Testing model on test dataset
trainer.test(lightning_module, datamodule=new_data_module)

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
n_batches = len(train_losses) // new_n_epochs

# Create x-axis in terms of epochs (0 to n_epochs)
x_all = np.linspace(0, new_n_epochs, len(train_losses))
x_epoch = np.arange(new_n_epochs+1)

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

# Get the scalers from original and new data module
orig_out_scaler = orig_data_module.out_scaler
orig_in_scaler = orig_data_module.in_scaler

new_out_scaler = new_data_module.out_scaler
new_in_scaler = new_data_module.in_scaler
if with_orig_inputs:past_in_scaler = new_data_module.past_in_scaler

#Converting tensors to numpy arrays if this isn't already done
if (type(test_outputs_T) != np.ndarray):
    test_outputs_T = test_outputs_T.numpy()
    test_outputs_P = test_outputs_P.numpy()

#Set up residual array
res_Ts = np.zeros((n_models+1, test_outputs_T.shape[0], test_outputs_T.shape[1]), dtype=float)
res_Ps = np.zeros((n_models+1, test_outputs_P.shape[0], test_outputs_P.shape[1]), dtype=float)

for test_idx, (test_input, test_output_T, test_output_P) in enumerate(zip(test_inputs, test_outputs_T, test_outputs_P)):

    #Convert to numpy and reshape
    test_input = test_input.numpy()

    pred_outputs_T = np.zeros((n_models+1,O), dtype=float)
    pred_outputs_P = np.zeros((n_models+1,O), dtype=float)

    #Retrieve predictions for each model
    for imodel, model in enumerate(ensemble_wrapper.models):
        pred_output = model(torch.tensor(orig_in_scaler.transform(test_input.reshape(1, -1)))).detach().numpy()
    
        # Inverse transform to get original scale
        pred_output_original =orig_out_scaler.inverse_transform(pred_output.reshape(1, -1)).flatten()
        
        # Split back into T and P components
        pred_outputs_T[imodel,:] = pred_output_original[:O]
        pred_outputs_P[imodel,:] = pred_output_original[O:]

    #Retrieve prediction from combiner
    if with_orig_inputs:
        pred_output = combiner(
                            torch.tensor(new_in_scaler.transform(np.concatenate([pred_outputs_T[imodel,:], pred_outputs_P[imodel,:]]).reshape(1, -1))), 
                            torch.tensor(past_in_scaler.transform(test_input.reshape(1, -1)))
                            ).detach().numpy()
    else:
        pred_output = combiner(torch.tensor(new_in_scaler.transform(torch.cat((pred_outputs_T, pred_outputs_P)).reshape(1, -1)))).detach().numpy()
                           
    # Inverse transform to get original scale
    pred_output_original =new_out_scaler.inverse_transform(pred_output.reshape(1, -1)).flatten()

    # Split back into T and P components
    pred_outputs_T[-1,:] = pred_output_original[:O]
    pred_outputs_P[-1,:] = pred_output_original[O:]
    
    #Storing residuals 
    res_Ts[:, test_idx, :] = pred_outputs_T - test_output_T
    res_Ps[:, test_idx, :] = pred_outputs_P - test_output_P

    #Plotting
    if (test_idx % substep == 0):
        fig, axs = plt.subplot_mosaic([['res_pressure', '.'],
                                    ['results', 'res_temperature']],
                            figsize=(8, 6),
                            width_ratios=(3, 1), height_ratios=(1, 3),
                            layout='constrained') 

        for imodel in range(n_models+1):       
            axs['results'].plot(test_output_T, test_output_P, '.', linestyle='-', color='blue', linewidth=2)
            axs['results'].plot(pred_outputs_T[imodel, :], pred_outputs_P[imodel, :], color='green', linewidth=2)

            axs['res_temperature'].plot(res_Ts[imodel, test_idx, :], test_output_P, '.', linestyle='-', color='green', linewidth=2)

            axs['res_pressure'].plot(test_output_T, res_Ps[imodel, test_idx, :], '.', linestyle='-', color='green', linewidth=2)

        axs['results'].invert_yaxis()
        axs['results'].set_ylabel(r'log$_{10}$ Pressure (bar)')
        axs['results'].set_xlabel('Temperature (K)')
        axs['results'].legend()
        axs['results'].grid()

        axs['res_temperature'].set_xlabel('Residuals (K)')
        axs['res_temperature'].invert_yaxis()
        axs['res_temperature'].grid()
        axs['res_temperature'].axvline(0, color='black', linestyle='dashed', zorder=2)
        axs['res_temperature'].yaxis.tick_right()
        axs['res_temperature'].yaxis.set_label_position("right")
        axs['res_temperature'].sharey(axs['results'])

        axs['res_pressure'].set_ylabel('Residuals (bar)')
        axs['res_pressure'].invert_yaxis()
        axs['res_pressure'].grid()
        axs['res_pressure'].axhline(0, color='black', linestyle='dashed', zorder=2)
        axs['res_pressure'].xaxis.tick_top()
        axs['res_pressure'].xaxis.set_label_position("top")
        axs['res_pressure'].sharex(axs['results'])
    
        plt.suptitle(rf'H$_2$ : {test_input[0]} bar, CO$_2$ : {test_input[1]} bar, LoD : {test_input[2]:.0f} days, Obliquity : {test_input[3]} deg')
        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{test_idx}.pdf')
    
    
fig, axes = plt.subplots(11, 2, sharex=True, figsize=[50, 8])
# Initialize statistics text 
stats_text = (
    f"--- NN Residuals ---\n"
)
print('\n','--- NN Residuals ---')

#Define colours 
colours = plt.get_cmap('viridis')(np.linspace(0., 1, n_models+1))

for imodel in range(n_models+1):
    print('\n',f'    --- Model {imodel+1} ---')
    print(f'    Temperature Residuals : Median = {np.median(res_Ts[imodel, :, :]):.2f} K, Std = {np.std(res_Ts[imodel, :, :]):.2f} K')
    print(rf'   Pressure Residuals : Median = {np.median(res_Ps[imodel, :, :]):.3f} $log_{10}$ bar, Std = {np.std(res_Ps[imodel, :, :]):.2f} $log_{10}$ bar')

    label = f'Model {imodel+1}' if imodel < n_models else 'Combiner'

    #Plot residuals
    axes[imodel, 0].set_title(label)
    axes[imodel, 0].plot(res_Ts[imodel, :, :].T, alpha=0.1, color=colours[imodel])
    axes[imodel, 1].plot(res_Ts[imodel, :, :].T, alpha=0.1, color=colours[imodel])
    for ax in [axes[imodel, 0], axes[imodel, 1]]:
        ax.axhline(0, color='black', linestyle='dashed')
        ax.set_xlabel('Index')
        ax.grid()
    axes[imodel, 0].set_ylabel('Temperature')
    axes[imodel, 1].set_ylabel('log$_{10}$ Pressure (bar)')

    stats_text += (
        f'    --- Model {imodel+1} ---'
        f"Temperature Residuals : Median = {np.median(res_Ts[imodel, :, :]):.2f} K, Std = {np.std(res_Ts[imodel, :, :]):.2f} K\n"
        f"Pressure Residuals : Median = {np.median(res_Ps[imodel, :, :]):.3f} $log_{{10}}$ bar, Std = {np.std(res_Ps[imodel, :, :]):.2f} $log_{{10}}$ bar"
    )

plt.subplots_adjust(hspace=0.1, bottom=0.25)

#Add statistics text at the bottom
fig.text(0.1, 0.05, stats_text, fontsize=10, family='monospace',
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.legend()
plt.savefig(plot_save_path+f'/res_NN.pdf', bbox_inches='tight')
plt.show()