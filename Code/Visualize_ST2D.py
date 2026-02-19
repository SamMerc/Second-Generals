#############################
#### Importing libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from tqdm import tqdm
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
#File containing surface temperature map
raw_ST_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_ST2D.csv', delimiter=',')
#Path to store plots
plot_save_path = base_dir+'Plots/'
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
#Definine partitiion for splitting NN dataset
data_partition = [0.7, 0.1, 0.2]

# Variable to show plots or not 
show_plot = True

#Number of nearest neighbors to choose
N_neigbors = np.linspace(5, 200, 5, dtype=int).tolist()

#Distance metric to use
distance_metric = 'euclidean' #options: 'euclidean', 'mahalanobis', 'logged_euclidean', 'logged_mahalanobis'

#Convert raw inputs for H2 and CO2 pressures to log10 scale so don't have to deal with it later
if 'logged' in distance_metric:
    raw_inputs[:, 0] = np.log10(raw_inputs[:, 0]) #H2
    raw_inputs[:, 1] = np.log10(raw_inputs[:, 1]) #CO2


INPUT_LABELS = [
    r'H$_2$ Pressure (bar)',
    r'CO$_2$ Pressure (bar)',
    r'LoD (days)',
    r'Obliquity (deg)',
]

############################################################
#### Plot curves, covariance matrices and eigenspectrum ####
############################################################

# --- ST2D maps ---
fig, ax = plt.subplots(figsize=(8, 6))
for i, raw_output in enumerate(raw_outputs):
    ax.plot(raw_output)
ax.set_xlabel('Index')
ax.set_ylabel('Temperature (K)')
plt.savefig(plot_save_path + 'ALL_ST2D_maps.pdf')
plt.show()


## Correlations with input parameters
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for j, ax in enumerate(axes.flatten()):
    
    # Normalize based on actual input values
    norm = mcolors.Normalize(vmin=min(raw_inputs[:, j]), vmax=max(raw_inputs[:, j]))
    cmap = cm.get_cmap('coolwarm')

    for i, raw_output in enumerate(raw_outputs):
        ax.plot(raw_output, color=cmap(norm(raw_inputs[i, j])))
    
    if j > 1:ax.set_xlabel('Index')
    if j==0 or j== 2:ax.set_ylabel(r'Temperature (K)')

    sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=INPUT_LABELS[j])
plt.subplots_adjust(hspace=0.1)
plt.savefig(plot_save_path + 'CORRELATED_ST2D_profiles.pdf')
plt.show()

# --- ST2D Covariance heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = sns.heatmap(
    np.cov(raw_outputs.T),
    cmap='coolwarm',
    ax=ax,
)
cbar = heatmap.collections[0].colorbar
cbar.set_label('Covariance', fontsize=11)
ax.set_xlabel('Index')
ax.set_ylabel('Index')
plt.savefig(plot_save_path + 'COV_ST2D_profiles.pdf')
plt.show()

# --- SVD Decomposition for ST2D ---
_, S_ST2D, _ = np.linalg.svd(raw_outputs, full_matrices=False)

fig, axes = plt.subplots(figsize=(8,6))
axes.plot(S_ST2D, color='blue')
axes.set_xlabel('Component Index')
axes.set_ylabel('Singular Value')
axes.set_yscale('log')

plt.savefig(plot_save_path + 'SVD_ST2D.pdf')
plt.show()