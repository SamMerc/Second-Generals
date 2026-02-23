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
from sklearn.neighbors import NearestNeighbors

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

plot_1 = False

############################################################
#### Plot curves, covariance matrices and eigenspectrum ####
############################################################

if plot_1:
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

###############################################
#### Ensemble Conditional Gaussian Process ####
########### (Visualization Example) ###########
###############################################

# Num_points = 50000

# #Generate some base data for visualization (replace with actual data in practice)
# t = np.linspace(-2*np.pi, 2*np.pi, Num_points)
# x1 = np.cos(t) + 0.1 * np.random.randn(Num_points) #shape: (N,)
# x2 = np.sin(t) + 0.1 * np.random.randn(Num_points) #shape: (N,)
# X = np.vstack([x1, x2]) #shape: (2, N)

# y = np.sin(x1 * x2) + 5 * np.exp(-x1**2 - x2**2) #shape: (N,)
# Y = y.reshape(1, -1) #shape: (1, N)

# # Train/test split
# XTr = X[:, :40000];  XTs = X[:, 10000:]
# YTr = Y[:, :40000];  YTs = Y[:, 10000:]

# #Define number of neighbors to use in KNN search
# K = 10

# # Pick a random test query
# idx = np.random.randint(0, XTs.shape[1])
# Xq = XTs[:, idx:idx+1]   # shape: (2, 1)

# # Initial KNN search using Mahalanobis distance
# def mahalanobis_knn(X_train, X_query, k):
#     """Find k nearest neighbors using Mahalanobis distance."""
#     nbrs = NearestNeighbors(n_neighbors=k, metric='mahalanobis', metric_params={'VI': np.linalg.inv(np.cov(X_train))})
#     nbrs.fit(X_train.T)
#     distances, indices = nbrs.kneighbors(X_query.T)
#     return indices[0]   # shape: (k,)

# idxs = mahalanobis_knn(XTr, Xq, K)

# # Precompute inverse covariance for Mahalanobis (used repeatedly)
# VI = np.linalg.inv(np.cov(XTr))

# def mahalanobis_knn_with_VI(X_train, X_query, k, VI):
#     nbrs = NearestNeighbors(n_neighbors=k, metric='mahalanobis', metric_params={'VI': VI})
#     nbrs.fit(X_train.T)
#     distances, indices = nbrs.kneighbors(X_query.T)
#     return indices   # shape: (n_queries, k)

# # Iterative refinement loop
# for i in range(20):
#     XTrSel = XTr[:, idxs]    # shape: (4, len(idxs))
#     YTrSel = YTr[:, idxs]    # shape: (M, len(idxs))

#     Xm = XTrSel.mean(axis=1, keepdims=True)   # (4, 1)
#     Ym = YTrSel.mean(axis=1, keepdims=True)   # (M, 1)

#     dX = XTrSel - Xm   # (4, K)
#     dY = YTrSel - Ym   # (M, K)

#     Cxx = dX @ dX.T    # (4, 4)
#     Cyx = dY @ dX.T    # (M, 4)
#     Cyy = dY @ dY.T    # (M, M)
#     Cxy = dX @ dY.T    # (4, M)

#     rdgx = np.min(np.linalg.eigvals(Cxx).real)
#     rdgy = np.min(np.linalg.eigvals(Cyy).real)

#     Mf = Cyx @ np.linalg.pinv(Cxx + rdgx * np.eye(Cxx.shape[0]))  # (M, 4)
#     Mb = Cxy @ np.linalg.pinv(Cyy + rdgy * np.eye(Cyy.shape[0]))  # (4, M)

#     YhSel = YTrSel + Mf @ (Xq - XTrSel)          # (M, K)
#     XhSel = XTrSel + Mb @ (Ym - YhSel)            # (4, K)

#     # KNN search: for each column of XhSel, find 1 nearest neighbor in XTr
#     idxs2 = mahalanobis_knn_with_VI(XTr, XhSel, 1, VI).flatten()

#     idxs = np.unique(idxs2)
#     lx = len(idxs)

#     if lx < K:
#         idxs3 = mahalanobis_knn_with_VI(XTr, Xq, K - lx, VI).flatten()
#         idxs = np.unique(np.concatenate([idxs, idxs3]))

# # Final prediction
# Yh = Ym + Mf @ (Xq - Xm)   # (M, 1)

# # Extract outputs (MATLAB 1-indexed → Python 0-indexed)
# lP  = YTs[0:51,  idx]    # ground truth log-pressures
# eT  = YTs[51:,   idx]    # ground truth temperatures
# lPh = Yh[0:51,   0]      # predicted log-pressures
# eTh = Yh[51:,    0]      # predicted temperatures

# # Plot
# plt.figure()
# plt.plot(eT,  lP,  'g', label='Ground Truth')
# plt.plot(eTh, lPh, 'r', label='Predicted')
# plt.xlabel('Temperature')
# plt.ylabel('log(Pressure)')
# plt.legend()
# plt.tight_layout()
# plt.show()


###############################################
#### Ensemble Conditional Gaussian Process ####
###############################################

# Extract features and targets
x = raw_inputs                                     # shape: (N, 4)
y = np.hstack([raw_outputs_P, raw_outputs_T])      # shape: (N, M) where M = 51 (P) + 51 (T) = 102
X = x.T   # shape: (4, N)
Y = y.T   # shape: (M, N)

# Shuffle data
rp = np.random.permutation(X.shape[1]) #random permutation of the indices
# Apply random permutation to shuffle the data
X = X[:, rp]
Y = Y[:, rp]

# Train/test split
XTr = X[:, :9000];  XTs = X[:, 9000:]
YTr = Y[:, :9000];  YTs = Y[:, 9000:]

#Define number of neighbors to use in KNN search
K = 10

# Pick a random test query
idx = np.random.randint(0, XTs.shape[1])
Xq = XTs[:, idx:idx+1]   # shape: (4, 1)

# Initial KNN search using Mahalanobis distance
def mahalanobis_knn(X_train, X_query, k):
    """Find k nearest neighbors using Mahalanobis distance."""
    nbrs = NearestNeighbors(n_neighbors=k, metric='mahalanobis', metric_params={'VI': np.linalg.inv(np.cov(X_train))})
    nbrs.fit(X_train.T)
    distances, indices = nbrs.kneighbors(X_query.T)
    return indices[0]   # shape: (k,)

idxs = mahalanobis_knn(XTr, Xq, K)

# Precompute inverse covariance for Mahalanobis (used repeatedly)
VI = np.linalg.inv(np.cov(XTr))

def mahalanobis_knn_with_VI(X_train, X_query, k, VI):
    nbrs = NearestNeighbors(n_neighbors=k, metric='mahalanobis', metric_params={'VI': VI})
    nbrs.fit(X_train.T)
    distances, indices = nbrs.kneighbors(X_query.T)
    return indices   # shape: (n_queries, k)

# Iterative refinement loop
for i in range(20):
    XTrSel = XTr[:, idxs]    # shape: (4, len(idxs))
    YTrSel = YTr[:, idxs]    # shape: (M, len(idxs))

    Xm = XTrSel.mean(axis=1, keepdims=True)   # (4, 1)
    Ym = YTrSel.mean(axis=1, keepdims=True)   # (M, 1)

    dX = XTrSel - Xm   # (4, K)
    dY = YTrSel - Ym   # (M, K)

    Cxx = dX @ dX.T    # (4, 4)
    Cyx = dY @ dX.T    # (M, 4)
    Cyy = dY @ dY.T    # (M, M)
    Cxy = dX @ dY.T    # (4, M)

    rdgx = np.min(np.linalg.eigvals(Cxx).real)
    rdgy = np.min(np.linalg.eigvals(Cyy).real)

    Mf = Cyx @ np.linalg.pinv(Cxx + rdgx * np.eye(Cxx.shape[0]))  # (M, 4)
    Mb = Cxy @ np.linalg.pinv(Cyy + rdgy * np.eye(Cyy.shape[0]))  # (4, M)

    YhSel = YTrSel + Mf @ (Xq - XTrSel)          # (M, K)
    XhSel = XTrSel + Mb @ (Ym - YhSel)            # (4, K)

    # KNN search: for each column of XhSel, find 1 nearest neighbor in XTr
    idxs2 = mahalanobis_knn_with_VI(XTr, XhSel, 1, VI).flatten()

    idxs = np.unique(idxs2)
    lx = len(idxs)

    if lx < K:
        idxs3 = mahalanobis_knn_with_VI(XTr, Xq, K - lx, VI).flatten()
        idxs = np.unique(np.concatenate([idxs, idxs3]))

# Final prediction
Yh = Ym + Mf @ (Xq - Xm)   # (M, 1)

# Extract outputs (MATLAB 1-indexed → Python 0-indexed)
lP  = YTs[0:51,  idx]    # ground truth log-pressures
eT  = YTs[51:,   idx]    # ground truth temperatures
lPh = Yh[0:51,   0]      # predicted log-pressures
eTh = Yh[51:,    0]      # predicted temperatures

# Plot
plt.figure()
plt.plot(eT,  lP,  'g', label='Ground Truth')
plt.plot(eTh, lPh, 'r', label='Predicted')
plt.xlabel('Temperature')
plt.ylabel('log(Pressure)')
plt.legend()
plt.tight_layout()
plt.show()

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

N_neighbor = N_neigbors
# Initialize array to store NN inputs / GP outputs
NN_inputs_T = np.zeros(raw_outputs_T.shape, dtype=float)
NN_inputs_P = np.zeros(raw_outputs_P.shape, dtype=float)

for query_idx, (query_input, query_output_T, query_output_P) in enumerate(zip(raw_inputs, raw_outputs_T, raw_outputs_P)):

    # My CGP
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

    #Sai CGP
    # Initial KNN search using Mahalanobis distance
    def mahalanobis_knn(X_train, X_query, k):
        """Find k nearest neighbors using Mahalanobis distance."""
        nbrs = NearestNeighbors(n_neighbors=k, metric='mahalanobis', metric_params={'VI': np.linalg.inv(np.cov(X_train))})
        nbrs.fit(X_train.T)
        distances, indices = nbrs.kneighbors(X_query.T)
        return indices[0]   # shape: (k,)

    idxs = mahalanobis_knn(XTr, Xq, K)

    # Precompute inverse covariance for Mahalanobis (used repeatedly)
    VI = np.linalg.inv(np.cov(XTr))

    def mahalanobis_knn_with_VI(X_train, X_query, k, VI):
        nbrs = NearestNeighbors(n_neighbors=k, metric='mahalanobis', metric_params={'VI': VI})
        nbrs.fit(X_train.T)
        distances, indices = nbrs.kneighbors(X_query.T)
        return indices   # shape: (n_queries, k)

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