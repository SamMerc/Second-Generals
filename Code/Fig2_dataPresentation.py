#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

##########################################################
#### Importing raw data and defining hyper-parameters ####
##########################################################
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing temperature values
raw_T_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_T.csv', delimiter=',')
raw_T_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_P.csv', delimiter=',')
raw_P_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#File containing ST maps
raw_ST_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_ST2D.csv', delimiter=',')
raw_ST_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')
#Path to store plots
plot_save_path = base_dir+'Plots/'
check_and_make_dir(plot_save_path)

# Extract inputs and concatenate
inputs_3000 = np.hstack([raw_T_data3000[:, :4], np.full((len(raw_T_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_T_data4500[:, :4], np.full((len(raw_T_data4500), 1), 4500.0)])

raw_inputs    = np.vstack([inputs_3000,            inputs_4500           ]) 
raw_outputs_T = np.vstack([raw_T_data3000[:, 5:],  raw_T_data4500[:, 5:]]) 
raw_outputs_P = np.vstack([raw_P_data3000[:, 5:],  raw_P_data4500[:, 5:]]) 
raw_outputs_ST = np.vstack([raw_ST_data3000[:, 5:],  raw_ST_data4500[:, 5:]]) 

raw_outputs_P = np.log10(raw_outputs_P/1000)

N = raw_inputs.shape[0]
IMG_H, IMG_W = 46, 72

np.random.seed(3)
rp = np.random.permutation(N)
raw_inputs = raw_inputs[rp, :]
raw_outputs_T = raw_outputs_T[rp, :]
raw_outputs_P = raw_outputs_P[rp, :]
raw_outputs_ST = raw_outputs_ST[rp, :]

INPUT_LABELS = [
    r'H$_2$ Pressure (bar)',
    r'CO$_2$ Pressure (bar)',
    r'LoD (days)',
    r'Obliquity (deg)',
    r'T$_{eff}$ (K)',
]

#######################
#### Plot Figure 2 ####
#######################

FS = 12

# 1. Define a 4-column mosaic: 
# Col 0: Plot A | Col 1: The Gap between A&C | Col 2: Plot C | Col 3: Colorbars
# 'G' is a dummy axis for the gap that we won't draw on.
fig, axes = plt.subplot_mosaic(
    [['A', 'G', 'C', 'cbar_top'],
     ['B', 'B', 'B', 'cbar_bot']], 
    figsize=(12, 8),
    gridspec_kw={
        # Widths: [A, gap_width, C, cbar_width]
        'width_ratios': [1, 0.15, 1, 0.04], 
        # wspace is now the gap between C and the colorbar. 
        # Making this tiny pulls the colorbar in.
        'wspace': 0.02,
        'hspace': 0.1
    }
)

# Hide the dummy gap axis
axes['G'].axis('off')

# --- Shared Colormap setup ---
norm = mcolors.Normalize(vmin=min(raw_inputs[:, 0]), vmax=max(raw_inputs[:, 0]))
cmap = cm.get_cmap('coolwarm')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# --- T profiles (Plot A) ---
ax = axes['A']
for i, raw_output_T in enumerate(raw_outputs_T):
    ax.plot(raw_output_T, color=cmap(norm(raw_inputs[i, 0])))
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('Index', fontsize=FS)
ax.set_ylabel('Temperature (K)', fontsize=FS)
ax.tick_params(axis='both', labelsize=FS)

# --- P profiles (Plot C) ---
ax = axes['C']
for i, raw_output_P in enumerate(raw_outputs_P):
    ax.plot(raw_output_P, color=cmap(norm(raw_inputs[i, 0])))
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('Index', fontsize=FS)
ax.set_ylabel(r'log$_{10}$ Pressure (bar)', fontsize=FS)
ax.invert_yaxis()
ax.tick_params(axis='both', labelsize=FS)

# --- Top Colorbar ---
cbar_t = fig.colorbar(sm, cax=axes['cbar_top'])
cbar_t.set_label(INPUT_LABELS[0], fontsize=FS)
cbar_t.ax.tick_params(labelsize=FS)

# --- ST Map (Plot B) ---
# Note: In the mosaic, B spans 3 columns to match the length of A + Gap + C
ax = axes['B']
hm = sns.heatmap(
    raw_outputs_ST[2, :].reshape((IMG_H, IMG_W)), 
    ax=ax, 
    cbar_ax=axes['cbar_bot']
)
ax.set_ylabel('Latitude (deg)', fontsize=FS)
ax.set_xlabel('Longitude (deg)', fontsize=FS)
ax.set_yticks(np.linspace(0, IMG_H, 5))
ax.set_yticklabels(np.linspace(-90, 90, 5).astype(int), fontsize=FS)
ax.set_xticks(np.linspace(0, IMG_W, 5))
ax.set_xticklabels(np.linspace(-180, 180, 5).astype(int), fontsize=FS)

# --- Bottom Colorbar ---
axes['cbar_bot'].set_ylabel('Temperature (K)', fontsize=FS)
axes['cbar_bot'].tick_params(labelsize=FS)

# Final adjustment
# We use constrained_layout or tight_layout, but since we have a custom gap, 
# tight_layout might try to "fix" it. We'll use subplots_adjust for the final polish.
plt.subplots_adjust(top=0.90, bottom=0.1, left=0.08, right=0.92)

plt.show()