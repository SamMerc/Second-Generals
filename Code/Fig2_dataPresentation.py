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

# Updated mosaic: 'Top' is now one single wide plot spanning the width of A+G+C
fig, axes = plt.subplot_mosaic(
    [['Top', 'Top', 'Top', 'cbar_top'],
     ['B', 'B', 'B', 'cbar_bot']], 
    figsize=(12, 8),
    gridspec_kw={
        'width_ratios': [1, 0.15, 1, 0.04], 
        'wspace': 0.02,
        'hspace': 0.1
    }
)

# --- Top Plot: T-P Density Plot ---
ax_tp = axes['Top']

# Flatten all data to get every single T-P coordinate pair in the dataset
# raw_outputs_T and raw_outputs_P share the same shape
all_T = raw_outputs_T.flatten()
all_P = raw_outputs_P.flatten()

# Use hexbin to calculate density - 'plasma' colormap as requested
# gridsize: higher means more detail; mincnt=1 hides empty cells
hb = ax_tp.hexbin(all_P, all_T, gridsize=100, cmap='plasma', mincnt=1, bins='log')

# Formatting the T-P plot
ax_tp.set_xlabel(r'log$_{10}$ Pressure (bar)', fontsize=FS)
ax_tp.set_ylabel('Temperature (K)', fontsize=FS)

# Move X-axis to top
ax_tp.xaxis.tick_top()
ax_tp.xaxis.set_label_position('top')

# Invert Y-axis so Pressure increases downwards
ax_tp.invert_xaxis()
ax_tp.tick_params(axis='both', labelsize=FS)

# # --- Top Colorbar (Density) ---
cbar_t = fig.colorbar(hb, cax=axes['cbar_top'])
cbar_t.set_label('Point Density (log$_{10}$ count)', fontsize=FS)
cbar_t.ax.tick_params(labelsize=FS)

# --- ST Map (Plot B) ---
ax_st = axes['B']
hm = sns.heatmap(
    raw_outputs_ST[4, :].reshape((IMG_H, IMG_W)), 
    ax=ax_st, 
    cbar_ax=axes['cbar_bot']
)
ax_st.set_ylabel('Latitude (deg)', fontsize=FS)
ax_st.set_xlabel('Longitude (deg)', fontsize=FS)

ax_st.set_yticks(np.linspace(0, IMG_H, 5))
ax_st.set_yticklabels(np.linspace(-90, 90, 5).astype(int), fontsize=FS)
ax_st.set_xticks(np.linspace(0, IMG_W, 5))
ax_st.set_xticklabels(np.linspace(-180, 180, 5).astype(int), fontsize=FS)

# --- Bottom Colorbar ---
axes['cbar_bot'].set_ylabel('Temperature (K)', fontsize=FS)
axes['cbar_bot'].tick_params(labelsize=FS)

plt.subplots_adjust(top=0.88, bottom=0.1, left=0.1, right=0.92)

plt.savefig(plot_save_path+'Fig2_density.pdf')
