#############################
#### Importing libraries ####
#############################
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from time import time
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
torch.set_float32_matmul_precision('high')

##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing surface temperature map
raw_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_ST2D.csv', delimiter=',')
raw_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/nn_only_CNN/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/nn_only_CNN/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values,
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
# Extract the 4 physical inputs and append stellar temperature as 5th column
inputs_3000 = np.hstack([raw_data3000[:, :4], np.full((len(raw_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_data4500[:, :4], np.full((len(raw_data4500), 1), 4500.0)])

# Concatenate along the sample axis
raw_inputs  = np.vstack([inputs_3000,           inputs_4500          ])
raw_outputs = np.vstack([raw_data3000[:, 5:],   raw_data4500[:, 5:] ])

#Storing useful quantities
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs.shape[1] #Number of outputs

# Map geometry
IMG_H, IMG_W = 46, 72
assert O == IMG_H * IMG_W, f"Output dim {O} != {IMG_H}x{IMG_W}"

# Shuffle data
shuffle_seed = 3
np.random.seed(shuffle_seed)
rp = np.random.permutation(N) #random permutation of the indices
# Apply random permutation to shuffle the data
raw_inputs  = raw_inputs[rp, :]
raw_outputs = raw_outputs[rp, :]

## HYPER-PARAMETERS for NN ##
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
show_plot = False

#CNN width (feature channels) and depth (number of residual conv blocks)
cnn_hidden_channels = 64
cnn_depth = 7

# Optimizer learning rate schedule - ReduceLROnPlateau
lr_init      = 1e-3   # initial LR — ReduceLROnPlateau will reduce from here
lr_patience  = 15     # epochs to wait before reducing LR
lr_factor    = 0.7    # multiply LR by this when plateauing
lr_min       = 1e-7   # floor

#Regularization coefficient
regularization_coeff_l1 = 0.0
regularization_coeff_l2 = 5e-5

#Smoothness constraint coefficient
smoothness_coeff = 1e-3

#Weight decay
weight_decay = 0.0

#Batch size
batch_size = 128

#Number of epochs
n_epochs = 1000

#Early stopping patience (in epochs)
early_stopping_patience = 50

#Mode for optimization
run_mode = 'use'




###################
#### Build CNN ####
###################
class ResidualConvBlock(nn.Module):
    """2-conv residual block with skip connection (mirrors TP_MLP.py's ResidualBlock)."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))   # ← skip connection

class CNN(nn.Module):
    """
    Mirrors the MLP architecture pattern from TP_MLP.py:
      input_proj -> depth x ResidualBlock -> output_proj

    Unlike ST2D_GP_CNN.py, there is no ens-CGP prediction/uncertainty map to
    supply as spatial input channels, so the single input_proj layer takes
    over that role: it projects the D raw physical inputs directly to a
    full-resolution (hidden_channels, H, W) feature map, analogous to how
    TP_MLP.py's input_proj (D -> hidden_dim) replaces the ens-CGP inputs of
    TP_GP_MLP.py. From there the residual conv blocks and output_proj are
    the direct 2-D counterparts of TP_MLP.py's residual blocks and
    output_proj.
    """
    def __init__(self, input_dim, hidden_channels, depth, img_height, img_width, generator=None):
        super().__init__()
        if generator is not None:
            torch.manual_seed(generator.initial_seed())

        self.hidden_channels = hidden_channels
        self.img_height = img_height
        self.img_width = img_width

        # Single projection layer: D physical inputs -> full-resolution feature map
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_channels * img_height * img_width),
            nn.GELU(),
        )
        # Stack of residual conv blocks, operating at full (H, W) resolution
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_channels) for _ in range(depth)])
        # Project to a single output channel (temperature map)
        self.output_proj = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.view(-1, self.hidden_channels, self.img_height, self.img_width)
        x = self.blocks(x)
        return self.output_proj(x)

# PyTorch Lightning DataModule
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs,
                 test_inputs, test_outputs, batch_size, rng, img_height, img_width):
        super().__init__()

        self.batch_size = batch_size
        self.rng = rng
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
        test_outputs  = torch.tensor(out_scaler.transform(test_outputs.cpu().numpy()),  dtype=torch.float32)

        # Store the scaler if you need to inverse transform later
        self.out_scaler = out_scaler

        # --- Input scaling ---
        # This is the "scaling layer at the start of the network": the CNN has
        # no ens-CGP step to standardize its inputs, so a StandardScaler here
        # feeds directly into the model's input_proj (D -> hidden_channels x
        # H x W) layer, which is the learned projection-to-image-space step
        # analogous to what ens-CGP did.
        in_scaler = StandardScaler()
        in_scaler.fit(train_inputs.cpu().numpy())

        def scale_inputs(X):
            return torch.tensor(in_scaler.transform(X.cpu().numpy()), dtype=torch.float32)

        self.train_inputs = scale_inputs(train_inputs)
        self.valid_inputs = scale_inputs(valid_inputs)
        self.test_inputs  = scale_inputs(test_inputs)

        # Store scaler for inference
        self.in_scaler = in_scaler

        # Reshape outputs to (N, 1, H, W) for the CNN
        self.train_outputs = train_outputs.reshape(-1, 1, img_height, img_width)
        self.valid_outputs = valid_outputs.reshape(-1, 1, img_height, img_width)
        self.test_outputs  = test_outputs.reshape(-1, 1, img_height, img_width)

    def train_dataloader(self):
        dataset = TensorDataset(self.train_inputs, self.train_outputs)
        return DataLoader(
         dataset,
         batch_size=self.batch_size,
         shuffle=True,
         generator=self.rng,
         pin_memory=True,
         )

    def val_dataloader(self):
        dataset = TensorDataset(self.valid_inputs, self.valid_outputs)
        return DataLoader(
         dataset,
         batch_size=self.batch_size,
         generator=self.rng,
         pin_memory=True,
         )

    def test_dataloader(self):
        dataset = TensorDataset(self.test_inputs, self.test_outputs)
        return DataLoader(
         dataset,
         batch_size=self.batch_size,
         generator=self.rng,
         pin_memory=True,
         )

model = CNN(D, cnn_hidden_channels, cnn_depth, IMG_H, IMG_W, generator=NN_rng)
summary(model, input_size=(1, D))




################################
### Build/Load training set ####
################################

# Split dataset into training, validation, and testing
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(N), data_partition, generator=partition_rng)

# --- Inputs: physical parameters only. Unlike ST2D_GP_CNN.py, there are no ---
# --- ens-CGP prediction/error maps to append since the CNN is doing the   ---
# --- full D -> (H, W) mapping by itself.                                 ---
NN_train_inputs = torch.tensor(raw_inputs[train_idx], dtype=torch.float32)
NN_valid_inputs = torch.tensor(raw_inputs[valid_idx], dtype=torch.float32)
NN_test_inputs  = torch.tensor(raw_inputs[test_idx],  dtype=torch.float32)

# --- Outputs: with no ens-CGP baseline to correct, the CNN targets the raw ---
# --- temperature map directly instead of a residual.                      ---
NN_train_outputs = torch.tensor(raw_outputs[train_idx], dtype=torch.float32)
NN_valid_outputs = torch.tensor(raw_outputs[valid_idx], dtype=torch.float32)
NN_test_outputs  = torch.tensor(raw_outputs[test_idx],  dtype=torch.float32)

# Create DataModule
data_module = CustomDataModule(
    NN_train_inputs, NN_train_outputs,
    NN_valid_inputs, NN_valid_outputs,
    NN_test_inputs, NN_test_outputs,
    batch_size, batch_rng,
    img_height=IMG_H, img_width=IMG_W,
)




###################################
#### Define optimization block ####
###################################
# PyTorch Lightning Module
class RegressionModule(pl.LightningModule):
    def __init__(self, model, optimizer, learning_rate, weight_decay=0.0,
                 reg_coeff_l1=0.0, reg_coeff_l2=0.0, smoothness_coeff=0.0,
                 out_scaler=None, img_height=None, img_width=None,
                 lr_patience=10, lr_factor=0.5, lr_min=1e-7):
        super().__init__()
        self.model            = model
        self.learning_rate    = learning_rate
        self.reg_coeff_l1     = reg_coeff_l1
        self.reg_coeff_l2     = reg_coeff_l2
        self.smoothness_coeff = smoothness_coeff
        self.weight_decay     = weight_decay
        self.loss_fn          = nn.MSELoss()
        self.optimizer_class  = optimizer
        self.lr_patience      = lr_patience
        self.lr_factor        = lr_factor
        self.lr_min           = lr_min

        # Buffer to un-scale the predicted map back to physical units (K), so
        # the smoothness penalty is computed on the reconstructed temperature
        # map S = pred (no GP baseline to add) rather than on the
        # standardized network output.
        self.register_buffer('out_mean',  torch.tensor(out_scaler.mean_,  dtype=torch.float32).view(1, 1, img_height, img_width))
        self.register_buffer('out_scale', torch.tensor(out_scaler.scale_, dtype=torch.float32).view(1, 1, img_height, img_width))

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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        X, y = batch
        pred = self(X)

        # Base loss: ||y - s||
        mse = self.loss_fn(pred, y)

        # Add weight regularization (L1/L2 on network parameters)
        l1_penalty, l2_penalty = self.compute_weight_regularization()
        loss = mse + l1_penalty + l2_penalty

        # Smoothness penalty: L2 norm of the predicted map's spatial gradient.
        # Longitude is genuinely periodic (edge cells wrap to their neighbor
        # on the opposite side of the map), but latitude is not — the two
        # poles are distinct physical points, not neighbors — so dS/dy uses a
        # plain forward difference with no wrap.
        if self.smoothness_coeff > 0:
            S_pred = pred * self.out_scale + self.out_mean

            dSdx_pred = torch.roll(S_pred, shifts=-1, dims=3) - S_pred   # periodic in longitude
            dSdy_pred = S_pred[:, :, 1:, :] - S_pred[:, :, :-1, :]        # no pole wrap-around → (batch, 1, H-1, W)

            # dSdx and dSdy live on different-sized grids (dSdy has no row at
            # the last pole boundary), so sum their squares separately
            # before adding.
            sum_sq = dSdx_pred.pow(2).sum(dim=(1, 2, 3)) + dSdy_pred.pow(2).sum(dim=(1, 2, 3))
            field_norm = torch.sqrt(sum_sq)   # (batch,)
            smoothness_penalty = self.smoothness_coeff * field_norm.mean()
            loss += smoothness_penalty

            self.log('train_smoothness', smoothness_penalty, on_step=True, on_epoch=True, prog_bar=True)

        # Log metrics
        self.log('train_mse', mse, on_step=True, on_epoch=True, prog_bar=True)
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
        optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss',   # ReduceLROnPlateau needs a metric to watch
                'interval': 'epoch',
                'frequency': 1,
            }
        }




######################
#### Run training ####
######################
# Create Lightning Module
lightning_module = RegressionModule(
    model=model,
    optimizer=Adam,
    learning_rate=lr_init,
    reg_coeff_l1=regularization_coeff_l1,
    reg_coeff_l2=regularization_coeff_l2,
    weight_decay=weight_decay,
    smoothness_coeff=smoothness_coeff,
    out_scaler=data_module.out_scaler,
    img_height=IMG_H,
    img_width=IMG_W,
    lr_patience=lr_patience,
    lr_factor=lr_factor,
    lr_min=lr_min,
)

# Setup logger
logger = CSVLogger(model_save_path+'logs', name='NeuralNetwork')

# Set all seeds for complete reproducibility
pl.seed_everything(NN_seed, workers=True)

#Define early stopping callback
early_stopping = EarlyStopping(
    monitor='valid_loss',
    patience=early_stopping_patience,
    mode='min',
    verbose=True,
)

# Create Trainer and train
trainer = Trainer(
    max_epochs=n_epochs,
    logger=logger,
    deterministic=True,
    enable_checkpointing=True,
    callbacks=[
        ModelCheckpoint(
            dirpath=model_save_path,
            save_top_k=1,
            monitor='valid_loss',
            mode='min',
        ),
        early_stopping,
    ],
    enable_progress_bar=True,
)

#Start time
t0 = time()

if run_mode == 'use':

    # Try to resume from last checkpoint if it exists
    last_ckpt = None
    if os.path.exists(model_save_path + 'last.ckpt'):
        last_ckpt = model_save_path + 'last.ckpt'

    trainer.fit(lightning_module, datamodule=data_module, ckpt_path=last_ckpt)

    # Get the best checkpoint path from ModelCheckpoint callback
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")

    # Save best path for later loading
    with open(model_save_path + 'best_ckpt_path.txt', 'w') as f:
        f.write(best_model_path)

    finish_time_s = time() - t0
    finish_time_min = finish_time_s/60
    finish_time_hrs = finish_time_s/3600
    finish_time_days = finish_time_s/(3600*24)
    print(f"Done! In {finish_time_s:.3f} s/{finish_time_min:.3f} min/{finish_time_hrs:.3f} hrs/{finish_time_days:.3f} days")

else:
    with open(model_save_path + 'best_ckpt_path.txt', 'r') as f:
        best_ckpt_path = f.read().strip()

    # Load model
    lightning_module = RegressionModule.load_from_checkpoint(
        best_ckpt_path,
        model=model,
        optimizer=Adam,
        learning_rate=lr_init,
        reg_coeff_l1=regularization_coeff_l1,
        reg_coeff_l2=regularization_coeff_l2,
        weight_decay=weight_decay,
        smoothness_coeff=smoothness_coeff,
        out_scaler=data_module.out_scaler,
        img_height=IMG_H,
        img_width=IMG_W,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        lr_min=lr_min,
    )
    print("Model loaded!")

model = lightning_module.model
model.cpu()
model.eval()

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
train_losses = metrics_df[metrics_df['train_mse_epoch'].notna()]['train_mse_epoch'].tolist()
eval_losses = metrics_df[metrics_df['valid_loss'].notna()]['valid_loss'].tolist()

# Smoothness penalty is only logged when its coeff > 0 (see training_step)
plot_smoothness = smoothness_coeff > 0 and 'train_smoothness_epoch' in metrics_df.columns
if plot_smoothness:
    smoothness_losses = metrics_df[metrics_df['train_smoothness_epoch'].notna()]['train_smoothness_epoch'].tolist()




##########################
#### Diagnostic plots ####
##########################
# Loss curves
n_extra_panels = int(plot_smoothness)
n_rows        = 2 + n_extra_panels
height_ratios = [3, 1] + [1] * n_extra_panels
fig, axes = plt.subplots(
    n_rows, 1, sharex=True, gridspec_kw={'height_ratios': height_ratios},
    figsize=(10, 6 + 2 * n_extra_panels)
)
ax1, ax2 = axes[0], axes[1]

# Calculate number of batches per epoch
actual_epochs = len(eval_losses)  # one entry per epoch
n_batches = len(train_losses) // actual_epochs  # batches per epoch
n_batches = max(1, n_batches)     # safety guard

# Create x-axis in terms of epochs (0 to n_epochs)
x_all = np.linspace(0, actual_epochs, len(train_losses))
x_epoch = np.arange(actual_epochs + 1)

# Plot transparent background showing all batch losses
ax1.plot(x_all, train_losses, alpha=0.3, color='C0', linewidth=0.5)
ax1.plot(x_all, eval_losses, alpha=0.3, color='C1', linewidth=0.5)

# Plot solid lines showing epoch-level losses (every n_batches steps)
train_epoch = [train_losses[0]] + train_losses[n_batches-1::n_batches]
eval_epoch  = [eval_losses[0]]  + eval_losses[n_batches-1::n_batches]
ax1.plot(x_epoch, train_epoch, label="Train", color='C0', linewidth=2, marker='o')
ax1.plot(x_epoch, eval_epoch, label="Validation", color='C1', linewidth=2, marker='o')

# Same for difference plot
diff_epoch  = np.abs(np.array(train_epoch) - np.array(eval_epoch))
ax2.plot(x_epoch, diff_epoch, color='C2', linewidth=2, marker='o')

ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_ylabel("MSE Loss")
ax2.set_ylabel("Loss Diff.")
ax1.legend()
ax1.grid()
ax2.grid()

if plot_smoothness:
    ax3 = axes[2]
    smoothness_epoch = [smoothness_losses[0]] + smoothness_losses[n_batches-1::n_batches]
    ax3.plot(x_all, smoothness_losses, alpha=0.3, color='C3', linewidth=0.5)
    ax3.plot(x_epoch, smoothness_epoch, label="Smoothness Penalty", color='C3', linewidth=2, marker='o')
    ax3.set_yscale('log')
    ax3.set_ylabel("Smoothness\nPenalty")
    ax3.legend()
    ax3.grid()

axes[-1].set_xlabel("Epoch")
plt.subplots_adjust(hspace=0)
plt.savefig(plot_save_path+'/loss.pdf')
plt.close()

#Comparing NN predicted ST maps vs true ST maps with residuals
substep = 100

# Get the scalers from data module
out_scaler = data_module.out_scaler
in_scaler = data_module.in_scaler

#Converting tensors to numpy arrays if this isn't already done
if (type(NN_test_outputs) != np.ndarray):
    NN_test_outputs = NN_test_outputs.cpu().numpy()

NN_res = np.zeros(NN_test_outputs.shape, dtype=float)

for NN_test_idx, (NN_test_input, true_output) in enumerate(zip(
    NN_test_inputs, NN_test_outputs
)):

    scaled_input = torch.tensor(
        in_scaler.transform(NN_test_input.numpy().reshape(1, -1)),
        dtype=torch.float32,
    )

    with torch.no_grad():
        NN_pred_output_scaled = model(scaled_input).numpy().reshape(1, -1)

    #Inverse scaling - CNN predicts the map directly (no GP baseline to add)
    NN_pred_output = out_scaler.inverse_transform(NN_pred_output_scaled).flatten()

    #Convert to numpy
    true_np = true_output
    NN_test_input_np = NN_test_input.cpu().numpy()

    #Storing residuals
    NN_res[NN_test_idx, :] = NN_pred_output - true_np

    #Plotting
    if (NN_test_idx % substep == 0):
        plot_true = true_np.reshape((IMG_H, IMG_W))
        plot_pred = NN_pred_output.reshape((IMG_H, IMG_W))
        plot_res  = NN_res[NN_test_idx, :].reshape((IMG_H, IMG_W))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout='constrained')

        ax1.set_title('Data')
        hm1 = sns.heatmap(plot_true, ax=ax1)
        hm1.collections[0].colorbar.set_label('Temperature (K)')

        ax2.set_title('CNN Model')
        hm2 = sns.heatmap(plot_pred, ax=ax2)
        hm2.collections[0].colorbar.set_label('Temperature (K)')

        ax3.set_title('CNN Residuals')
        hm3 = sns.heatmap(plot_res, ax=ax3)
        hm3.collections[0].colorbar.set_label('Temperature (K)')

        ax3.set_xticks(np.linspace(0, IMG_W, 5))
        ax3.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
        ax3.set_xlabel('Longitude (degrees)')
        for ax in [ax1, ax2, ax3]:
            ax.set_yticks(np.linspace(0, IMG_H, 5))
            ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
            ax.set_ylabel('Latitude (degrees)')

        plt.suptitle(rf'H$_2$ : {NN_test_input_np[0]} bar, CO$_2$ : {NN_test_input_np[1]} bar, LoD : {NN_test_input_np[2]:.0f} days, Obliquity : {NN_test_input_np[3]} deg, Teff : {NN_test_input_np[4]} K')
        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{NN_test_idx}.pdf')
        plt.close()


#Plot residuals
fig, ax = plt.subplots(figsize=[10, 6])
for qid in range(len(NN_test_outputs)):
    ax.plot(NN_res[qid, :], alpha=0.1, color='blue')
ax.axhline(0, color='black', linestyle='dashed')
ax.grid()
ax.set_xlabel('Pixel Index')
ax.set_ylabel('Temperature Residual (K)')
plt.subplots_adjust(bottom=0.25)

# Add statistics text at the bottom
stats_text = (
    f"--- CNN Residuals ---\n"
    f"Temperature Residuals : Median = {np.median(NN_res):.2f} K, "
    f"Std = {np.std(NN_res):.2f} K, "
    f"RMSE = {np.sqrt(np.mean(NN_res**2)):.2f} K"
)

fig.text(0.1, 0.05, stats_text, fontsize=10, family='monospace',
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(plot_save_path+f'/res_NN.pdf', bbox_inches='tight')
plt.close()
