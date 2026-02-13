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
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns


##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing surface temperature map
raw_ST_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/NN_ST_FlowMatching/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/NN_ST_FlowMatching/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
raw_inputs = raw_ST_data[:, :4] #has shape 46 x 72 = 3,312
raw_outputs = raw_ST_data[:, 5:]

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
IMG_H, IMG_W = 46, 72

## HYPER-PARAMETERS ##
#Defining partition of data used for 1. training, 2. validation and 3. testing
data_partitions = [0.7, 0.1, 0.2]

#Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 1
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

#Optimizer learning rate
learning_rate = 1e-3

#Regularization coefficient
regularization_coeff_l1 = 0.0
regularization_coeff_l2 = 0.0

#Weight decay 
weight_decay = 0.0

#Batch size 
batch_size = 64

#Number of epochs 
n_epochs = 200

#Number of Euler steps for integral evaluation at inference
# More steps = more accurate but slower. 50–100 is usually plenty.
NUM_INFERENCE_STEPS = 50 

#Mode for optimization
run_mode = 'use'







##############################################
#### Partition data and generate datasets ####
##############################################
# PyTorch Lightning DataModule
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
        out_scaler = MinMaxScaler()
        
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
        in_scaler = StandardScaler()
        
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
            
            self.train_outputs = train_outputs.reshape(-1, img_channels, self.img_height, self.img_width)
            self.valid_outputs = valid_outputs.reshape(-1, img_channels, self.img_height, self.img_width)
            self.test_outputs = test_outputs.reshape(-1, img_channels, self.img_height, self.img_width)

        else:
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

#Splitting the data 

## Retrieving indices of data partitions
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(N), data_partitions, generator=partition_rng)

## Generate the data partitions
### Training
train_inputs = torch.tensor(raw_inputs[train_idx], dtype=torch.float32)
train_outputs = torch.tensor(raw_outputs[train_idx], dtype=torch.float32)
### Validation
valid_inputs = torch.tensor(raw_inputs[valid_idx], dtype=torch.float32)
valid_outputs = torch.tensor(raw_outputs[valid_idx], dtype=torch.float32)
### Testing
test_inputs = torch.tensor(raw_inputs[test_idx], dtype=torch.float32)
test_outputs = torch.tensor(raw_outputs[test_idx], dtype=torch.float32)

# Create DataModule
data_module = CustomDataModule(
    train_inputs, train_outputs,
    valid_inputs, valid_outputs,
    test_inputs, test_outputs,
    batch_size, batch_rng, reshape_for_cnn=True,
    img_channels=1, img_height=46, img_width=72
)


#################################################
#### Build Conditional Velocity U-Net        ####
#################################################
#
# This network predicts the *velocity field* v_θ(x_t, t, cond).
#
# Architecture overview:
#
#   cond (B,4) + t (B,)
#       │
#       ▼
#   ConditionEncoder → cond_emb (B, COND_DIM)
#       │
#       │   injected at every scale via FiLM (γ, β per channel)
#       │
#   x_t (B,1,46,72)
#   ──────────────────────────────────────────
#   Encoder:  1→32→64→128 (stride-2 convolutions)
#   Bottleneck: 128→128
#   Decoder:  128→64→32→1  (bilinear upsampling + skip connections)
#   ──────────────────────────────────────────
#       │
#       ▼
#   velocity (B,1,46,72)   (same shape as x_t and the clean target x1)
#
# FiLM conditioning: for each conv block, a linear layer maps cond_emb to
# (γ, β) of the same channel width, then applies  γ * feat + β.
# This is the standard lightweight way to inject conditioning into a U-Net.


class SinusoidalEmbedding(nn.Module):
    """Map scalar t ∈ [0,1] → (B, dim) sinusoidal embedding."""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:   # t: (B,)
        half   = self.dim // 2
        freqs  = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args   = t[:, None] * freqs[None]                  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1) # (B, dim)


class ConditionEncoder(nn.Module):
    """
    Encodes the 4D physical parameters + sinusoidal time embedding into
    a single conditioning vector of size `out_dim`.
    """
    def __init__(self, cond_dim: int = 4, time_emb_dim: int = 64, out_dim: int = 256):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(cond_dim + time_emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
            nn.SiLU(),
        )

    def forward(self, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        cond: (B, 4), t: (B,)  →  (B, out_dim)
        out_dim = 64 + 4 = 68
        """
        t_emb = self.time_emb(t)
        return self.net(torch.cat([cond, t_emb], dim=-1))


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Scales and shifts a (B, C, H, W) feature map using a conditioning vector.
    """
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)  # predicts γ and β

    def forward(self, feat: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W),  cond_emb: (B, cond_dim)
        gamma, beta = self.proj(cond_emb).chunk(2, dim=-1)       # each (B, C)
        gamma = gamma[:, :, None, None]                           # broadcast over H, W
        beta  = beta [:, :, None, None]
        return gamma * feat + beta


class ConvBlock(nn.Module):
    """Conv → GroupNorm → SiLU, with optional FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )
        self.film = FiLMLayer(cond_dim, out_ch)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        return self.film(self.conv(x), cond_emb)


class VelocityUNet(nn.Module):
    """
    Lightweight conditional U-Net for the velocity field.

    Input:  x_t  (B, 1,  46, 72)  — noisy temperature map at time t
            cond (B, 4)            — physical parameters (already Z-scored)
            t    (B,)              — flow time in [0, 1]
    Output: v    (B, 1,  46, 72)  — predicted velocity
    """

    COND_DIM = 256   # dimension of the shared conditioning embedding

    def __init__(self, img_channels: int = 1, cond_input_dim: int = 4, generator=None):
        super().__init__()

        # Set seed if generator provided
        if generator is not None:
            torch.manual_seed(generator.initial_seed())

        C = self.COND_DIM

        # ── Conditioning encoder ─────────────────────────────────────────
        self.cond_encoder = ConditionEncoder(cond_dim=cond_input_dim, out_dim=C)

        # ── Encoder (stride-2 downsampling) ──────────────────────────────
        self.enc1 = ConvBlock(img_channels, 32,  C)   # (B, 32,  46, 72)
        self.down1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # (B, 32,  23, 36)

        self.enc2 = ConvBlock(32, 64,  C)             # (B, 64,  23, 36)
        self.down2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # (B, 64,  12, 18)

        self.enc3 = ConvBlock(64, 128, C)             # (B, 128, 12, 18)
        self.down3 = nn.Conv2d(128, 128, 3, stride=2, padding=1) # (B, 128,  6,  9)

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(128, 128, C)      # (B, 128,  6,  9)

        # ── Decoder (bilinear upsample + skip concat + conv) ──────────────
        self.up3   = nn.Upsample(size=(12, 18), mode='bilinear', align_corners=False)
        self.dec3  = ConvBlock(128 + 128, 64, C)     # skip from enc3

        self.up2   = nn.Upsample(size=(23, 36), mode='bilinear', align_corners=False)
        self.dec2  = ConvBlock(64 + 64, 32, C)       # skip from enc2

        self.up1   = nn.Upsample(size=(46, 72), mode='bilinear', align_corners=False)
        self.dec1  = ConvBlock(32 + 32, 32, C)       # skip from enc1

        # ── Output projection ─────────────────────────────────────────────
        # No activation — velocity can be negative!
        self.out_conv = nn.Conv2d(32, img_channels, 1)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Shared conditioning embedding
        emb = self.cond_encoder(cond, t)        # (B, COND_DIM)

        # Encoder
        s1   = self.enc1(x_t, emb)              # (B, 32,  46, 72)
        s2   = self.enc2(self.down1(s1), emb)   # (B, 64,  23, 36)
        s3   = self.enc3(self.down2(s2), emb)   # (B, 128, 12, 18)

        # Bottleneck
        b    = self.bottleneck(self.down3(s3), emb)  # (B, 128, 6, 9)

        # Decoder with skip connections
        d3   = self.dec3(torch.cat([self.up3(b),  s3], dim=1), emb)  # (B, 64,  12, 18)
        d2   = self.dec2(torch.cat([self.up2(d3), s2], dim=1), emb)  # (B, 32,  23, 36)
        d1   = self.dec1(torch.cat([self.up1(d2), s1], dim=1), emb)  # (B, 32,  46, 72)

        return self.out_conv(d1)                # (B, 1,   46, 72)


model = VelocityUNet(img_channels=1, cond_input_dim=D, generator=NN_rng)
summary(model, input_data=[
    torch.zeros(2, 1, IMG_H, IMG_W),
    torch.zeros(2, D),
    torch.zeros(2),
])



###################################
#### Define optimization block ####
###################################
class FlowMatchingModule(pl.LightningModule):
    def __init__(self, model, optimizer, learning_rate, weight_decay=0.0, reg_coeff_l1=0.0, reg_coeff_l2=0.0, num_inference_steps=50):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.reg_coeff_l1 = reg_coeff_l1
        self.reg_coeff_l2 = reg_coeff_l2
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()
        self.optimizer_class = optimizer

    # ── Flow matching helpers ─────────────────────────────────────────────

    def _sample_timesteps(self, B: int) -> torch.Tensor:
        return torch.rand(B, device=self.device)

    def _forward_process(self, x1: torch.Tensor, z0: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        """x_t = (1-t)*z0 + t*x1.  t is broadcast over (C, H, W)."""
        t_view = t.view(-1, 1, 1, 1)
        return (1.0 - t_view) * z0 + t_view * x1

    def _velocity_target(self, x1: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """v* = x1 - z0  (constant along each linear path)."""
        return x1 - z0

    # ── Optional weight regularization (kept from original) ──────────────

    def _weight_reg(self):
        l1 = l2 = torch.tensor(0., device=self.device)
        if self.reg_coeff_l1 > 0 or self.reg_coeff_l2 > 0:
            for p in self.model.parameters():
                if self.reg_coeff_l1 > 0: l1 += p.abs().sum()
                if self.reg_coeff_l2 > 0: l2 += (p ** 2).sum()
        return self.reg_coeff_l1 * l1, self.reg_coeff_l2 * l2

    # ── Lightning steps ───────────────────────────────────────────────────

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, cond, t)

    def training_step(self, batch):
        cond, x1 = batch                              # (B,4), (B,1,H,W)
        B = cond.size(0)

        z0 = torch.randn_like(x1)                     # pure noise
        t  = self._sample_timesteps(B)                # t ~ U(0,1)
        x_t = self._forward_process(x1, z0, t)        # noisy interpolation

        v_target = self._velocity_target(x1, z0)      # ground-truth velocity
        v_pred   = self(x_t, cond, t)                 # predicted velocity

        loss = self.loss_fn(v_pred, v_target)         # MSE between predicted and true velocity
        l1, l2 = self._weight_reg()
        loss = loss + l1 + l2

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        cond, x1 = batch
        B = cond.size(0)
        z0 = torch.randn_like(x1)
        t  = self._sample_timesteps(B)
        x_t = self._forward_process(x1, z0, t)
        v_target = self._velocity_target(x1, z0)
        v_pred   = self(x_t, cond, t)
        loss = self.loss_fn(v_pred, v_target)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        cond, x1 = batch
        B = cond.size(0)
        z0 = torch.randn_like(x1)
        t  = self._sample_timesteps(B)
        x_t = self._forward_process(x1, z0, t)
        v_target = self._velocity_target(x1, z0)
        v_pred   = self(x_t, cond, t)
        loss = self.loss_fn(v_pred, v_target)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(),
                    lr=self.learning_rate, weight_decay=self.weight_decay)

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, cond: torch.Tensor,
                num_steps: int | None = None) -> torch.Tensor:
        """
        Generate a temperature map for each row of `cond`.

        Args:
            cond:      (B, 4) tensor of *already Z-scored* physical parameters
            num_steps: Euler integration steps (default: self.num_inference_steps)

        Returns:
            Predicted image tensor (B, 1, 46, 72) in the scaler's [0,1] space.
            Use data_module.out_scaler.inverse_transform() to recover Kelvin.
        """
        steps = num_steps or self.num_inference_steps
        B = cond.size(0)
        dt = 1.0 / steps

        # Start from pure Gaussian noise
        x = torch.randn(B, 1, IMG_H, IMG_W, device=cond.device, dtype=cond.dtype)

        # Euler integration: t goes from 0 → 1
        for i in range(steps):
            t = torch.full((B,), i * dt, device=cond.device, dtype=cond.dtype)
            v = self.model(x, cond, t)
            x = x + v * dt

        return x   # (B, 1, 46, 72)






######################
#### Run training ####
######################
lightning_module = FlowMatchingModule(
    model=model,
    optimizer=Adam,
    learning_rate=learning_rate,
    reg_coeff_l1=regularization_coeff_l1,
    reg_coeff_l2=regularization_coeff_l2,
    weight_decay=weight_decay,
    num_inference_steps=NUM_INFERENCE_STEPS,
)

# Setup logger
logger = CSVLogger(model_save_path+'logs', name='FlowMatching')

# Set all seeds for complete reproducibility
pl.seed_everything(NN_seed, workers=True)

# Create Trainer and train
trainer = Trainer(
    max_epochs=n_epochs,
    logger=logger,
    deterministic=True  # For reproducibility
)

ckpt_name = (f'{n_epochs}epochs_{weight_decay}WD_{NUM_INFERENCE_STEPS}IS'
             f'{regularization_coeff_l1+regularization_coeff_l2}RC_'
             f'{learning_rate}LR_{batch_size}BS.ckpt')

if run_mode == 'use':
    
    trainer.fit(lightning_module, datamodule=data_module)
    
    # Save model (PyTorch Lightning style)
    trainer.save_checkpoint(model_save_path + ckpt_name)
    
    print("Done!")
    
else:
    # Load model
    lightning_module = FlowMatchingModule.load_from_checkpoint(
    model_save_path + ckpt_name,
    model=model,
    optimizer=Adam,
    learning_rate=learning_rate,
    reg_coeff_l1=regularization_coeff_l1,
    reg_coeff_l2=regularization_coeff_l2,
    weight_decay=weight_decay,
    num_inference_steps=NUM_INFERENCE_STEPS,
    )
    print("Model loaded!")


#Testing model on test dataset
trainer.test(lightning_module, datamodule=data_module)

# --- Accessing Training History After Training ---
# Find the version directory (e.g., version_0, version_1, etc.)
log_dir = model_save_path+'logs/FlowMatching'
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
substep = 1000

# Get the scalers from data module
out_scaler = data_module.out_scaler
in_scaler = data_module.in_scaler

#Converting tensors to numpy arrays if this isn't already done
if (type(test_outputs) != np.ndarray):
    test_outputs = test_outputs.cpu().numpy()

res = np.zeros(test_outputs.shape, dtype=float)

lightning_module.eval()

for test_idx, (test_input, test_output) in enumerate(zip(test_inputs, test_outputs)):

    #Convert to numpy
    test_input = test_input.cpu().numpy()

    # Scale input
    cond_scaled = torch.tensor(
        in_scaler.transform(test_input.reshape(1, -1)), dtype=torch.float32
    ).to(device)

    #Retrieve prediction
    pred_scaled = lightning_module.predict(cond_scaled, num_steps=NUM_INFERENCE_STEPS)
    
    pred_flat = pred_scaled.cpu().numpy().reshape(1, -1)  # (1, 3312)

    # Inverse transform to get original scale
    pred_output = out_scaler.inverse_transform(pred_flat).flatten()  # (3312,)

    #Storing residuals 
    res[test_idx, :] = pred_output - test_output

    #Plotting
    if (test_idx % substep == 0):

        #Convert shape
        plot_test_output = test_output.reshape((46, 72))
        plot_pred_output = pred_output.reshape((46, 72))
        plot_res = res[test_idx, :].reshape((46, 72))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout='constrained')        
        
        # Compute global vmin/vmax across all datasets
        vmin = np.min(test_output)
        vmax = np.max(test_output)
        
        # Plot heatmaps
        ax1.set_title('Data')
        hm1 = sns.heatmap(plot_test_output, ax=ax1)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm1.collections[0].colorbar
        cbar.set_label('Temperature (K)')

        ax2.set_title('NN Model')
        hm3 = sns.heatmap(plot_pred_output, ax=ax2)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm3.collections[0].colorbar
        cbar.set_label('Temperature (K)')

        ax3.set_title('NN Residuals')
        hm5 = sns.heatmap(plot_res, ax=ax3)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm5.collections[0].colorbar
        cbar.set_label('Temperature (K)')

        ax3.set_xticks(np.linspace(0, 72, 5))
        ax3.set_xticklabels(np.linspace(-180, 180, 5).astype(int))
        ax3.set_xlabel('Longitude (degrees)')
        # Fix latitude ticks
        for ax in [ax1, ax2, ax3]:
            ax.set_yticks(np.linspace(0, 46, 5))
            ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int))
            ax.set_ylabel('Latitude (degrees)')
        plt.suptitle(rf'H$_2$O : {test_input[0]} bar, CO$_2$ : {test_input[1]} bar, LoD : {test_input[2]:.0f} days, Obliquity : {test_input[3]} deg')
        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{test_idx}.pdf')
    