#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from pathlib import Path
from torch.optim import AdamW
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import os
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import seaborn as sns
from diffusers import UNet2DModel
import torch.nn.functional as F

#########################################################
#### Part 1: Create HDF5 File from CSV (Run Once)    ####
#########################################################

def create_hdf5_from_csv(
    csv_path: str,
    output_hdf5_path: str,
    partition_seed: int = 4,
):
    """
    Convert CSV data to HDF5 format.
    """
    print("="*60)
    print("Creating HDF5 file from CSV...")
    print("="*60)
    
    # Load CSV data
    print(f"Loading CSV: {csv_path}")
    raw_data = np.loadtxt(csv_path, delimiter=',')
    
    # Split into inputs and outputs
    raw_inputs = raw_data[:, :4]  # (N, 4) - H2, CO2, LoD, Obliquity
    raw_outputs = raw_data[:, 5:]  # (N, 3312) - Temperature maps
    
    N = raw_inputs.shape[0]
    IMG_H, IMG_W = 46, 72
    
    #Create random indices to make splitting easier later on
    rng = torch.Generator()
    rng.manual_seed(partition_seed)
    indices = torch.randperm(N, generator=rng).numpy()

    # Reshape outputs to image format (N, 1, H, W)
    images = raw_outputs.reshape(N, 1, IMG_H, IMG_W).astype(np.float32)[indices]
    parameters = raw_inputs.astype(np.float32)[indices]

    #Store normalization constants
    img_mean = float(np.mean(images))
    img_std = float(np.std(images))
    
    scalar_means = parameters.mean(axis=0).tolist()  # List of 4 means
    scalar_stds = parameters.std(axis=0).tolist()    # List of 4 stds
    
    # Save to HDF5
    print(f"\nSaving to HDF5: {output_hdf5_path}")
    Path(output_hdf5_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_hdf5_path, 'w') as f:
        # Save full dataset
        f.create_dataset('images', data=images, dtype='float32', compression='gzip')
        f.create_dataset('parameters', data=parameters, dtype='float32', compression='gzip')
        
        # Save normalization statistics as attributes
        f.attrs['img_mean'] = img_mean
        f.attrs['img_std'] = img_std
        f.attrs['scalar_means'] = scalar_means
        f.attrs['scalar_stds'] = scalar_stds
        f.attrs['num_data'] = N
    
    print("\n✓ HDF5 file created successfully!")
    print("="*60)
    
    return None


##########################################################
#### Part 2: PyTorch Lightning Data Module           ####
##########################################################

class TemperatureMapDataModule(Dataset):
    """DataModule for temperature maps with HDF5 backend."""
    
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = {},
        idx_list: list = None,
    ):
        super().__init__()

        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.norm_dict = norm_dict

        with h5py.File(hdf5_path, 'r') as f:
            # Check strictly if dataset exists
            if 'images' not in f or 'parameters' not in f:
                raise KeyError(f"HDF5 file must contain 'images' and 'parameters' keys. Found: {list(f.keys())}")

            # 1. Load Data
            if idx_list is not None:
                # Use sorted indices to optimize HDF5 read speed
                sorted_indices = sorted(idx_list) # Note: idx list needs to be in ascending order. if you call it with range it will automatically be so
                self.images = torch.from_numpy(f['images'][sorted_indices]).float()
                self.scalars = torch.from_numpy(f['parameters'][sorted_indices]).float() # 'parameters' is the key for the 4 scalar values / or however you saved them in the hdf5 file, maybe you even saved in 4 different keys
            else:
                self.images = torch.from_numpy(f['images'][:]).float()
                self.scalars = torch.from_numpy(f['parameters'][:]).float()

        self.num_images = len(self.images)

        # 2. Print Stats - not sure how accurate this will be but anyway it will print once at the beginning of training and you will know the datset was initialized correctly
        img_mem = self.images.element_size() * self.images.numel() / (1024**3)
        scalar_mem = self.scalars.element_size() * self.scalars.numel() / (1024**3)
        print(f"Loaded {self.num_images} items.")
        print(f"Images Shape: {self.images.shape} | Scalars Shape: {self.scalars.shape}")
        print(f"Memory usage: ~{img_mem + scalar_mem:.3f} GB")

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # 1. Get raw data
        # shape: (C, H, W) / make sure its this way because the model expects (C, H, W ) and not (H, W, C) if i remember correctly (you should doublecheck just in case)
        target_image = self.images[idx]
        scalar_cond = self.scalars[idx] # shape: (4,)

        # 2. Normalize Image
        # norm_dict['images'] is tuple (mean, std)
        if 'images' in self.norm_dict and self.norm_dict['images'] is not None:
            mean, std = self.norm_dict['images']
            target_image = (target_image - mean) / std

        # 3. Normalize Scalars
        if 'scalars' in self.norm_dict and self.norm_dict['scalars'] is not None:
            s_mean = torch.tensor(self.norm_dict['scalars'][0], dtype=torch.float32)
            s_std = torch.tensor(self.norm_dict['scalars'][1], dtype=torch.float32)
            scalar_cond = (scalar_cond - s_mean) / s_std

        # 4. Pad the images to a 48x72 size
        target_image = F.pad(target_image, (0, 0, 1, 1))

        return target_image, scalar_cond


##########################################################
#### Part 3: Flow Matching Model                     ####
##########################################################

class ConditionalFlowMatchingModule(pl.LightningModule):
    """Conditional Flow Matching using Diffusers UNet2DModel."""
    
    def __init__(
        self,
        # DATA PARAMS
        in_channels: int = 1,
        cond_channels: int = 4,
        image_height: int = 48,
        image_width: int = 72,
        # UNET PARAMS
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        # OPTIMIZATION
        lr: float = 1e-4,
        num_inference_steps: int = 100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.num_inference_steps = num_inference_steps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.image_height = image_height
        self.image_width = image_width
        
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        
        self.velocity_model = UNet2DModel(
            sample_size=(image_height, image_width),
            in_channels=in_channels + cond_channels,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,

            # Standard ResNet/Attention blocks
            down_block_types=(
                "DownBlock2D",        # ResNet only
                "AttnDownBlock2D",    # ResNet + Self-Attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            attention_head_dim=attention_head_dim,

        )

        #You might need to change it for:
        # down_block_types=(
        #             "DownBlock2D",
        #             "DownBlock2D",
        #             "DownBlock2D",
        #             "DownBlock2D",
        #         ),
        #         mid_block_type='UNetMidBlock2D',
        #         up_block_types=(
        #             "UpBlock2D",
        #             "UpBlock2D",
        #             "UpBlock2D",
        #             "UpBlock2D",
        #         ),
        # but im not sure

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_scalars: torch.Tensor):

        cond_spatial = cond_scalars[:, :, None, None].expand(
        -1, -1, x_t.shape[2], x_t.shape[3]
        )
        x_input = torch.cat([x_t, cond_spatial], dim=1)  # (B, 5, H, W)

        # UNet2DModel expects timesteps to be scaled roughly to 0-1000 range usually
        # but pure Flow Matching often works with [0,1]. Diffusers defaults usually prefer larger ints.
        timesteps = t * 1000
        return self.velocity_model(
            x_input,
            timesteps, 
            ).sample
    
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        target_image, cond_scalars = batch # Expects (Image, Scalars)
        batch_size = target_image.shape[0]
        
        x_0 = torch.randn_like(target_image)
        t = torch.rand(batch_size, device=target_image.device)
        
        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * target_image
        target_velocity = target_image - x_0
        
        predicted_velocity = self(x_t, t, cond_scalars)
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss
    
    def training_step(self, batch: tuple, batch_idx: int):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int):
        loss = self.compute_loss(batch)
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch: tuple, batch_idx: int):
        loss = self.compute_loss(batch)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def sample(self, cond_scalars: torch.Tensor, num_steps: int = None):
        num_steps = num_steps or self.num_inference_steps
        num_samples = cond_scalars.shape[0]
        device = cond_scalars.device
        
        # Start from Noise
        x = torch.randn(
            num_samples, self.in_channels, self.image_height, self.image_width,
            device=device,
        )
        
        dt = 1.0 / num_steps

        # Euler Integration
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            velocity = self(x, t, cond_scalars)
            x = x + velocity * dt
        
        return x
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



##########################################################
#### Part 5: Main Training Script                    ####
##########################################################

if __name__ == "__main__":
    
    # ============================================
    # Configuration
    # ============================================
    
    # Hyperparameters
    IMG_H, IMG_W = 48, 72
    partition_seed = 4
    batch_seed = 5
    NN_seed = 6

    base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
    
    hdf5_save_path = base_dir + 'Data/bt-4500k/HDF5_datasets/'
    model_save_path = base_dir + 'Model_Storage/NN_ST_FlowMatching_Diffusers/'
    plot_save_path = base_dir + 'Plots/NN_ST_FlowMatching_Diffusers/'
    
    os.makedirs(hdf5_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)
    
    csv_path = base_dir + 'Data/bt-4500k/training_data_ST2D.csv'
    hdf5_path = Path(hdf5_save_path + f'temperature_maps.h5')
    
    # Optimizer parameters
    learning_rate = 1e-4
    batch_size = 64
    n_epochs = 200
    NUM_INFERENCE_STEPS = 100
    num_workers = 1  # Adjust based on your CPU
    
    run_mode = 'use'  # 'use' to train from scratch, 'load' to load checkpoint
    
    device = 'cpu'
    print(f"\nUsing device: {device}")
    
    # =================================== #
    # Step 1: Create HDF5 file (run once) #
    # =================================== #
    
    if not Path(hdf5_path).exists():
        print("\nHDF5 file not found. Creating from CSV...")
        create_hdf5_from_csv(
            csv_path=csv_path,
            output_hdf5_path=hdf5_path,
        )
    else:
        print(f"\nHDF5 file already exists: {hdf5_path}")
    
    print("Loading normalization statistics from HDF5...") 
    with h5py.File(hdf5_path, 'r') as f:
        norm_dict = {
            'images': (f.attrs['img_mean'], f.attrs['img_std']),
            'scalars': (
                list(f.attrs['scalar_means']),
                list(f.attrs['scalar_stds'])
            )
        }
        num_data = f.attrs['num_data']
        
    print(f"Normalization statistics:")
    print(f"  Images: mean={norm_dict['images'][0]:.6f}, std={norm_dict['images'][1]:.6f}")
    print(f"  Scalars means: {norm_dict['scalars'][0]}")
    print(f"  Scalars stds: {norm_dict['scalars'][1]}")
    
    # =================================================================== #
    # Step 2: Extract training, validation and testing sets and load them #
    # =================================================================== #
    
    print("\n" + "="*60)
    print("Creating DataModule...")
    print("="*60)

    #Generation partition
    # Data has been randomly permutated so can just use ordered list to extract the datasets
    partition = (num_data * np.cumsum([0.7, 0.2, 0.1])).astype(int) # 70/20/10 split for train/validation/test

    train_dataset = TemperatureMapDataModule(
        hdf5_path=hdf5_path,
        norm_dict=norm_dict,
        idx_list=range(partition[0]),
    )

    val_dataset = TemperatureMapDataModule(
        hdf5_path=hdf5_path,
        norm_dict=norm_dict,
        idx_list=range(partition[0], partition[1]),
    )
    
    test_dataset = TemperatureMapDataModule(
        hdf5_path=hdf5_path,
        norm_dict=norm_dict,
        idx_list=range(partition[1], num_data),
    )

    rng = torch.Generator()
    loader_generator=rng.manual_seed(batch_seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, generator=loader_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, generator=loader_generator)
    # ============================================
    # Step 3: Create Model + Optimizer
    # ============================================
    
    print("\n" + "="*60)
    print("Creating Flow Matching Model...")
    print("="*60)
    
    model = ConditionalFlowMatchingModule(
        in_channels=1,
        cond_channels=4,
        image_height=IMG_H,
        image_width=IMG_W,
        model_channels=64,
        channel_mult=(1, 2, 4),
        lr=learning_rate,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ============================================
    # Step 4: Setup Training
    # ============================================

    csv_logger = CSVLogger(save_dir=model_save_path + 'logs', name="FlowMatching")

    pl.seed_everything(NN_seed, workers=True)
    
    trainer = Trainer(
        max_epochs=n_epochs,
        logger=csv_logger,
        deterministic=True,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
    )
    
    ckpt_name = f'{n_epochs}epochs_{learning_rate}LR_{batch_size}BS.ckpt'
    ckpt_path = model_save_path + ckpt_name
    
    # ============================================
    # Step 5: Train or Load Model
    # ============================================
    
    if run_mode == 'use':
        print("\n" + "="*60)
        print("Starting Training...")
        print("="*60)
        
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(ckpt_path)
        
        print("\n✓ Training complete!")
        print(f"✓ Model saved to: {ckpt_path}")
    else:
        print("\n" + "="*60)
        print("Loading pre-trained model...")
        print("="*60)
        
        model = ConditionalFlowMatchingModule.load_from_checkpoint(
            ckpt_path,
            in_channels=1,
            cond_channels=4,
            image_height=IMG_H,
            image_width=IMG_W,
            model_channels=64,
            channel_mult=(1, 2, 4),
            lr=learning_rate,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )
        print(f"✓ Model loaded from: {ckpt_path}")
    
    # ============================================
    # Step 6: Test Model
    # ============================================
    
    print("\n" + "="*60)
    print("Testing Model...")
    print("="*60)
    
    trainer.test(model, test_loader)
    
    # ============================================
    # Step 7: Generate Loss Plot
    # ============================================
    
    print("\n" + "="*60)
    print("Generating Diagnostic Plots...")
    print("="*60)
    
    # Loss curves
    log_dir = model_save_path + 'logs/FlowMatching'
    versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    latest_version = sorted(versions)[-1]
    csv_path_metrics = os.path.join(log_dir, latest_version, 'metrics.csv')
    
    metrics_df = pd.read_csv(csv_path_metrics)
    train_losses = metrics_df[metrics_df['train_loss_epoch'].notna()]['train_loss_epoch'].tolist()
    eval_losses = metrics_df[metrics_df['valid_loss'].notna()]['valid_loss'].tolist()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
                                    gridspec_kw={'height_ratios':[3, 1]}, 
                                    figsize=(10, 6))
    
    n_batches = len(train_losses) // n_epochs
    x_all = np.linspace(0, n_epochs, len(train_losses))
    x_epoch = np.arange(n_epochs + 1)
    
    ax1.plot(x_all, train_losses, alpha=0.3, color='C0', linewidth=0.5)
    ax1.plot(x_all, eval_losses, alpha=0.3, color='C1', linewidth=0.5)
    
    train_epoch = [train_losses[0]] + train_losses[n_batches-1::n_batches]
    eval_epoch = [eval_losses[0]] + eval_losses[n_batches-1::n_batches]
    ax1.plot(x_epoch, train_epoch, label="Train", color='C0', linewidth=2, marker='o')
    ax1.plot(x_epoch, eval_epoch, label="Validation", color='C1', linewidth=2, marker='o')
    
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
    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_save_path + 'loss.pdf')
    print(f"✓ Loss plot saved to: {plot_save_path}loss.pdf")
    
    # ============================================
    # Step 8: Generate Predictions Plot
    # ============================================
    
    print("\n" + "="*60)
    print("Generating Predictions on Test Set...")
    print("="*60)
    
    model.eval()
        
    # Denormalization functions
    img_mean, img_std = norm_dict['images']
    scalar_means, scalar_stds = norm_dict['scalars']
    
    def denorm_image(img_norm):
        return img_norm * img_std + img_mean
    
    def denorm_scalars(scalars_norm):
        s_mean = torch.tensor(scalar_means, dtype=torch.float32)
        s_std = torch.tensor(scalar_stds, dtype=torch.float32)
        return scalars_norm * s_std + s_mean
    
    # Generate predictions for a few test samples
    n_samples_to_plot = min(5, len(test_dataset))
    
    for i in range(n_samples_to_plot):
        # Get normalized data
        img_norm, scalars_norm = test_dataset[i]
        
        # Denormalize
        img_true = denorm_image(img_norm).cpu().numpy()[0]  # (H, W)
        img_true = img_true[1:-1, :]
        scalars_true = denorm_scalars(scalars_norm).cpu().numpy()
        
        # Generate prediction
        scalars_norm_batch = scalars_norm.unsqueeze(0).to(device)
        img_pred_norm = model.sample(scalars_norm_batch, num_steps=NUM_INFERENCE_STEPS)
        img_pred = denorm_image(img_pred_norm[0]).cpu().numpy()[0]  # (H, W)
        img_pred = img_pred[1:-1, :]

        # Calculate residual
        residual = img_pred - img_true
        
        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), 
                                             sharex=True, layout='constrained')
        
        vmin, vmax = img_true.min(), img_true.max()
        
        # Plot heatmaps
        ax1.set_title('Data')
        hm1 = sns.heatmap(img_true, ax=ax1)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm1.collections[0].colorbar
        cbar.set_label('Temperature (K)')

        ax2.set_title('NN Model')
        hm3 = sns.heatmap(img_pred, ax=ax2)#, cbar=False, vmin=vmin, vmax=vmax)
        cbar = hm3.collections[0].colorbar
        cbar.set_label('Temperature (K)')

        ax3.set_title('NN Residuals')
        hm5 = sns.heatmap(residual, ax=ax3)#, cbar=False, vmin=vmin, vmax=vmax)
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
        plt.suptitle(rf'H$_2$O : {scalars_true[0]} bar, CO$_2$ : {scalars_true[1]} bar, LoD : {scalars_true[2]:.0f} days, Obliquity : {scalars_true[3]} deg')
        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{i}.pdf')