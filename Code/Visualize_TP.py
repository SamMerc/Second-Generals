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
#File containing temperature values
raw_T_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#Path to store plots
plot_save_path = base_dir+'Plots/'
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

# --- P profiles ---
## Base 
fig, ax = plt.subplots(figsize=(8, 6))
for i, raw_output_P in enumerate(raw_outputs_P):
    ax.plot(raw_output_P, color=cm.get_cmap('coolwarm')(i / (len(raw_outputs_P) - 1)))
ax.set_xlabel('Index')
ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
ax.invert_yaxis()
# Add colorbar to show index scale
sm = cm.ScalarMappable(cmap='coolwarm', norm=mcolors.Normalize(vmin=0, vmax=len(raw_outputs_P) - 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Profile Index')
plt.savefig(plot_save_path + 'ALL_P_profiles.pdf')
plt.show()

## Correlations with input parameters
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for j, ax in enumerate(axes.flatten()):
    
    # Normalize based on actual input values
    norm = mcolors.Normalize(vmin=min(raw_inputs[:, j]), vmax=max(raw_inputs[:, j]))
    cmap = cm.get_cmap('coolwarm')

    for i, raw_output_P in enumerate(raw_outputs_P):
        ax.plot(raw_output_P, color=cmap(norm(raw_inputs[i, j])))
    
    if j > 1:ax.set_xlabel('Index')
    if j==0 or j== 2:ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
    ax.invert_yaxis()

    sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=INPUT_LABELS[j])
plt.subplots_adjust(hspace=0.1)
plt.savefig(plot_save_path + 'CORRELATED_P_profiles.pdf')
plt.show()

# --- T profiles ---
fig, ax = plt.subplots(figsize=(8, 6))
for i, raw_output_T in enumerate(raw_outputs_T):
    ax.plot(raw_output_T)
ax.set_xlabel('Index')
ax.set_ylabel('Temperature (K)')
plt.savefig(plot_save_path + 'ALL_T_profiles.pdf')
plt.show()

## Correlations with input parameters
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for j, ax in enumerate(axes.flatten()):
    
    # Normalize based on actual input values
    norm = mcolors.Normalize(vmin=min(raw_inputs[:, j]), vmax=max(raw_inputs[:, j]))
    cmap = cm.get_cmap('coolwarm')

    for i, raw_output_T in enumerate(raw_outputs_T):
        ax.plot(raw_output_T, color=cmap(norm(raw_inputs[i, j])))
    
    if j > 1:ax.set_xlabel('Index')
    if j==0 or j== 2:ax.set_ylabel(r'Temperature (K)')

    sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=INPUT_LABELS[j])
plt.subplots_adjust(hspace=0.1)
plt.savefig(plot_save_path + 'CORRELATED_T_profiles.pdf')
plt.show()

# --- P Covariance plot ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.cov(raw_outputs_P.T), color='blue')
ax.set_xlabel('Index')
ax.set_ylabel('Covariance')
plt.savefig(plot_save_path + 'COV_P_profiles.pdf')
plt.show()

# --- T Covariance heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = sns.heatmap(
    np.cov(raw_outputs_T.T),
    cmap='coolwarm',
    ax=ax,
)
cbar = heatmap.collections[0].colorbar
cbar.set_label('Covariance', fontsize=11)
ax.set_xlabel('Index')
ax.set_ylabel('Index')
plt.savefig(plot_save_path + 'COV_T_profiles.pdf')
plt.show()

# --- SVD Decomposition for T ---
_, S_T, _ = np.linalg.svd(raw_outputs_T, full_matrices=False)

fig, axes = plt.subplots(figsize=(8,6))
axes.plot(S_T, color='blue')
axes.set_xlabel('Component Index')
axes.set_ylabel('Singular Value')
axes.set_yscale('log')

plt.savefig(plot_save_path + 'SVD_T.pdf')
plt.show()

# --- SVD Decomposition for P ---
_, S_P, _ = np.linalg.svd(raw_outputs_P, full_matrices=False)

fig, axes = plt.subplots(figsize=(8,6))
axes.plot(S_P, color='blue')
axes.set_xlabel('Component Index')
axes.set_ylabel('Singular Value')
axes.set_yscale('log')

plt.savefig(plot_save_path + 'SVD_P.pdf')
plt.show()

raise KeyboardInterrupt



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
    Cxx += 1e-6 * np.eye(Cxx.shape[0]) 

    query_labels = obs_labels + (Cyx @ scipy.linalg.pinv(Cxx) @ (query_features - obs_features))

    query_cov_labels = Cyy - Cyx @ scipy.linalg.pinv(Cxx) @ Cxy

    return query_labels, query_cov_labels




############################
#### Build training set ####
############################

if type(N_neigbors)==int:
    print('BUILDING GP TRAINING SET')
    N_neighbor = N_neigbors
    # Initialize array to store NN inputs / GP outputs
    NN_inputs_T = np.zeros(raw_outputs_T.shape, dtype=float)
    NN_inputs_P = np.zeros(raw_outputs_P.shape, dtype=float)

    for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(raw_inputs, raw_outputs_T, raw_outputs_P)):

        #Calculate proximity of query point to observations
        distances = np.sqrt( (query_input[0] - raw_inputs[:,0])**2 + (query_input[1] - raw_inputs[:,1])**2 + (query_input[2] - raw_inputs[:,2])**2 + (query_input[3] - raw_inputs[:,3])**2 )

        #Choose the N closest points
        # skip the first point since it corresponds to the query point itself
        N_closest_idx = np.argsort(distances)[1:N_neighbor+1]
        prox_train_inputs = raw_inputs[N_closest_idx, :]
        prox_train_outputs_T = raw_outputs_T[N_closest_idx, :]
        prox_train_outputs_P = raw_outputs_P[N_closest_idx, :]

        #Find the query labels from nearest neigbours
        mean_test_output, cov_test_output = Sai_CGP(prox_train_inputs.T, np.concat((prox_train_outputs_T, prox_train_outputs_P), axis=1).T, query_input.reshape((1, 4)).T)
        model_test_output_T = np.mean(mean_test_output[:O],axis=1)
        model_test_output_P = np.mean(mean_test_output[O:],axis=1)
        model_test_output_Terr = np.sqrt(np.diag(cov_test_output))[:O]
        model_test_output_Perr = np.sqrt(np.diag(cov_test_output))[O:]
        NN_inputs_T[query_idx, :] = model_test_output_T
        NN_inputs_P[query_idx, :] = model_test_output_P

        #Diagnostic plot
        if show_plot:

            #Plot TP profiles
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
            
            #ax1 : prediction, truth and the neighbors
            for prox_idx in range(N_neighbor):
                ax1.plot(prox_train_outputs_T[prox_idx], prox_train_outputs_P[prox_idx], '.', linestyle='-', color='red', alpha=0.1, linewidth=2, zorder=1, label='Ensemble' if prox_idx==0 else None)
            
            #ax2 : prediction, truth and prediction errorbars in T
            ax2.errorbar(model_test_output_T, model_test_output_P, xerr=model_test_output_Terr, fmt='.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=2, alpha=0.4)
            ax2.fill_betweenx(model_test_output_P, model_test_output_T-model_test_output_Terr, model_test_output_T+model_test_output_Terr, color='green', zorder=2, alpha=0.2)
            #ax3 : prediction, truth and prediction errorbars in P
            ax3.errorbar(model_test_output_T, model_test_output_P, yerr=model_test_output_Perr, fmt='.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=2, alpha=0.4)
            ax3.fill_between(model_test_output_T, model_test_output_P-model_test_output_Perr, model_test_output_P+model_test_output_Perr, color='green', zorder=2, alpha=0.2)

            for ax in [ax1, ax2, ax3]:
                ax.plot(model_test_output_T, model_test_output_P, '.', linestyle='-', color='green', linewidth=2, markersize=10, zorder=3, label='Prediction')

                ax.plot(query_output_T, query_output_P, '.', linestyle='-', color='blue', linewidth=2, zorder=3, markersize=10, label='Truth')

                ax.invert_yaxis()
                
                if ax == ax1 : ax.set_ylabel(r'log$_{10}$ Pressure (bar)')
                ax.set_xlabel('Temperature (K)')
                
                ax.grid()
                ax.legend()        

            plt.suptitle(rf'H$_2$ : {query_input[0]} bar, CO$_2$ : {query_input[1]} bar, LoD : {query_input[2]:.0f} days, Obliquity : {query_input[3]} deg')
            plt.subplots_adjust(wspace=0.2)
            plt.show()

elif type(N_neigbors) == list:
    print('DIAGNOSTIC PLOTTING')
    #bias estimator
    bias_estim_T = np.zeros(len(N_neigbors), dtype=float)
    bias_estim_P = np.zeros(len(N_neigbors), dtype=float)
    
    #variance estimator
    var_estim_T = np.zeros(len(N_neigbors), dtype=float)
    var_estim_P = np.zeros(len(N_neigbors), dtype=float)

    #Loop over possible N_Neighbors values
    for n_idx, N_neighbor in enumerate(tqdm(N_neigbors)):
        
        # Initialize array to store NN inputs / GP outputs
        NN_inputs_T = np.zeros(raw_outputs_T.shape, dtype=float)
        NN_inputs_P = np.zeros(raw_outputs_P.shape, dtype=float)
        NN_inputs_Terr = np.zeros(raw_outputs_T.shape, dtype=float)
        NN_inputs_Perr = np.zeros(raw_outputs_P.shape, dtype=float)

        for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(raw_inputs, raw_outputs_T, raw_outputs_P)):

            #Calculate proximity of query point to observations
            distances = np.sqrt( (query_input[0] - raw_inputs[:,0])**2 + (query_input[1] - raw_inputs[:,1])**2 + (query_input[2] - raw_inputs[:,2])**2 + (query_input[3] - raw_inputs[:,3])**2 )

            #Choose the N closest points
            # skip the first point since it corresponds to the query point itself
            N_closest_idx = np.argsort(distances)[1:N_neighbor+1]
            prox_train_inputs = raw_inputs[N_closest_idx, :]
            prox_train_outputs_T = raw_outputs_T[N_closest_idx, :]
            prox_train_outputs_P = raw_outputs_P[N_closest_idx, :]

            #Find the query labels from nearest neigbours
            mean_test_output, cov_test_output = Sai_CGP(prox_train_inputs.T, np.concat((prox_train_outputs_T, prox_train_outputs_P), axis=1).T, query_input.reshape((1, 4)).T)
            
            model_test_output_T = np.mean(mean_test_output[:O],axis=1)
            model_test_output_P = np.mean(mean_test_output[O:],axis=1)
            model_test_output_Terr = np.sqrt(np.diag(cov_test_output))[:O]
            model_test_output_Perr = np.sqrt(np.diag(cov_test_output))[O:]
            
            NN_inputs_T[query_idx, :] = model_test_output_T
            NN_inputs_P[query_idx, :] = model_test_output_P
            NN_inputs_Terr[query_idx, :] = model_test_output_Terr
            NN_inputs_Perr[query_idx, :] = model_test_output_Perr

        # Calculate bias estimator (for T and P separately)
        bias_estim_T[n_idx] = np.sqrt(np.mean((NN_inputs_T - raw_outputs_T)**2))  # RMSE over all points & levels
        bias_estim_P[n_idx] = np.sqrt(np.mean((NN_inputs_P - raw_outputs_P)**2))

        # Calculate variance estimator (for T and P separately)
        var_estim_T[N_neigbors.index(N_neighbor)] = np.mean(NN_inputs_Terr**2)
        var_estim_P[n_idx] = np.mean(NN_inputs_Perr**2)

    #Combine bias estimators to get a bias for the full prediction
    # Normalize T and P bias estimators with the mean values to remove scales that could lead to one of the estimators biasing the total
    bias_estim = ((bias_estim_T / np.mean(raw_outputs_T)) + (bias_estim_P / np.mean(raw_outputs_P))) / 2.
    var_estim = (var_estim_T + var_estim_P) / 2.
    print(bias_estim_T, bias_estim_P, bias_estim, np.mean(raw_outputs_T), np.mean(raw_outputs_P))
    print(var_estim_T, var_estim_P, var_estim)
    
    #Plot the bias and variance estimators as a function of the number of neighbors 
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(12, 8))
    
    axes[0,0].plot(N_neigbors, bias_estim_T, markersize=10, linestyle='-', color='blue')
    axes[0,1].plot(N_neigbors, bias_estim_P, markersize=10, linestyle='-', color='blue')
    axes[0,2].plot(N_neigbors, bias_estim, markersize=10, linestyle='-', color='blue')
    
    axes[1,0].plot(N_neigbors, var_estim_T, markersize=10, linestyle='-', color='blue')
    axes[1,1].plot(N_neigbors, var_estim_P, markersize=10, linestyle='-', color='blue')
    axes[1,2].plot(N_neigbors, var_estim, markersize=10, linestyle='-', color='blue')
    
    titles_row0 = ['Bias T', 'Bias P', 'Bias Combined']
    titles_row1 = ['Var T', 'Var P', 'Var Combined']
    for j in range(3):
        axes[0,j].set_title(titles_row0[j])
        axes[1,j].set_title(titles_row1[j])
        axes[0,j].grid(True)
        axes[1,j].grid(True)
        axes[1,j].set_xlabel('N neighbors')
    
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()

    raise KeyboardInterrupt('Done diagnostic plotting. Re-run with fixed N_neighbors for model training.')

else:
    raise KeyboardInterrupt('Invalid number of neighbors variable. Should be a list of int type.')