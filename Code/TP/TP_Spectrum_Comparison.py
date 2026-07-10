##############################################################################
#### Transmission spectra from true vs NN-predicted
#### TP profiles for a single test case, using petitRADTRANS.
####
#### The trained BNN (TP_GP_BNN.py) predicts a TP profile (GP baseline + NN
#### residual correction) for held-out test cases. This script reloads that
#### trained model, picks one test case, and feeds both the ground-truth ExoCAM
####  TP profile and the BNN-predicted TP profile (mean +/- 1 MC std) through
####  petitRADTRANS to see how the TP discrepancy propagates into a transmission spectrum.
####
#### Atmosphere composition for both spectra is identical and fixed by the
#### test case's known inputs: H2 and CO2 partial pressures (bar) plus 1 bar
#### of N2 (fixed in the ExoCAM setup). 
##############################################################################
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.math import convolve_and_sample_variable_resolution_breads

##########################################################
#### Reproducing the BNN's data split / GP / scalers ####
##########################################################

def check_and_make_dir(dir):
    if not os.path.isdir(dir): os.mkdir(dir)

base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
raw_T_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_T.csv', delimiter=',')
raw_T_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
raw_P_data3000 = np.loadtxt(base_dir+'Data/bt-3000k/training_data_P.csv', delimiter=',')
raw_P_data4500 = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')

model_save_path = base_dir+'Model_Storage/BNN_GP_MLP/'
plot_save_path = base_dir+'Plots/TP_Spectrum_Comparison/'
check_and_make_dir(plot_save_path)

inputs_3000 = np.hstack([raw_T_data3000[:, :4], np.full((len(raw_T_data3000), 1), 3000.0)])
inputs_4500 = np.hstack([raw_T_data4500[:, :4], np.full((len(raw_T_data4500), 1), 4500.0)])

raw_inputs    = np.vstack([inputs_3000,            inputs_4500           ])
raw_outputs_T = np.vstack([raw_T_data3000[:, 5:],  raw_T_data4500[:, 5:]])
raw_outputs_P = np.vstack([raw_P_data3000[:, 5:],  raw_P_data4500[:, 5:]])
raw_outputs_P = np.log10(raw_outputs_P/1000)

N = raw_inputs.shape[0]
D = raw_inputs.shape[1]
O = raw_outputs_T.shape[1]

shuffle_seed = 3
np.random.seed(shuffle_seed)
rp = np.random.permutation(N)
raw_inputs = raw_inputs[rp, :]
raw_outputs_T = raw_outputs_T[rp, :]
raw_outputs_P = raw_outputs_P[rp, :]

N_neighbor = 4
data_partition = [0.7, 0.1, 0.2]
partition_seed = 4
partition_rng = torch.Generator()
partition_rng.manual_seed(partition_seed)
batch_seed = 5
batch_rng = torch.Generator()
batch_rng.manual_seed(batch_seed)
NN_seed = 6
NN_rng = torch.Generator()
NN_rng.manual_seed(NN_seed)

nn_width = 209
nn_depth = 32
bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Reparameterization",
    "moped_enable": False,
    "moped_delta": 0.5,
}
batch_size = 200
mc_samples = 100

# --- Load cached ens-CGP baseline ---
gp_cache_path = base_dir + f'Model_Storage/gp_cache_Nn{N_neighbor}_seed{shuffle_seed}.npz'
cache = np.load(gp_cache_path)
GP_outputs_T    = cache['GP_outputs_T']
GP_outputs_P    = cache['GP_outputs_P']
GP_outputs_Terr = cache['GP_outputs_Terr']
GP_outputs_Perr = cache['GP_outputs_Perr']

residuals_T = raw_outputs_T - GP_outputs_T
residuals_P = raw_outputs_P - GP_outputs_P

train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(N), data_partition, generator=partition_rng)

NN_train_inputs_phys = torch.tensor(raw_inputs[train_idx], dtype=torch.float32)
NN_valid_inputs_phys = torch.tensor(raw_inputs[valid_idx], dtype=torch.float32)
NN_test_inputs_phys  = torch.tensor(raw_inputs[test_idx],  dtype=torch.float32)

NN_train_inputs_T = torch.tensor(GP_outputs_T[train_idx], dtype=torch.float32)
NN_train_inputs_P = torch.tensor(GP_outputs_P[train_idx], dtype=torch.float32)
NN_valid_inputs_T = torch.tensor(GP_outputs_T[valid_idx], dtype=torch.float32)
NN_valid_inputs_P = torch.tensor(GP_outputs_P[valid_idx], dtype=torch.float32)
NN_test_inputs_T  = torch.tensor(GP_outputs_T[test_idx],  dtype=torch.float32)
NN_test_inputs_P  = torch.tensor(GP_outputs_P[test_idx],  dtype=torch.float32)

NN_train_inputs_Terr = torch.tensor(GP_outputs_Terr[train_idx], dtype=torch.float32)
NN_train_inputs_Perr = torch.tensor(GP_outputs_Perr[train_idx], dtype=torch.float32)
NN_valid_inputs_Terr = torch.tensor(GP_outputs_Terr[valid_idx], dtype=torch.float32)
NN_valid_inputs_Perr = torch.tensor(GP_outputs_Perr[valid_idx], dtype=torch.float32)
NN_test_inputs_Terr  = torch.tensor(GP_outputs_Terr[test_idx],  dtype=torch.float32)
NN_test_inputs_Perr  = torch.tensor(GP_outputs_Perr[test_idx],  dtype=torch.float32)

NN_train_outputs_T = torch.tensor(residuals_T[train_idx], dtype=torch.float32)
NN_train_outputs_P = torch.tensor(residuals_P[train_idx], dtype=torch.float32)
NN_valid_outputs_T = torch.tensor(residuals_T[valid_idx], dtype=torch.float32)
NN_valid_outputs_P = torch.tensor(residuals_P[valid_idx], dtype=torch.float32)
NN_test_outputs_T  = torch.tensor(residuals_T[test_idx],  dtype=torch.float32)
NN_test_outputs_P  = torch.tensor(residuals_P[test_idx],  dtype=torch.float32)

NN_test_true_T = torch.tensor(raw_outputs_T[test_idx], dtype=torch.float32)
NN_test_true_P = torch.tensor(raw_outputs_P[test_idx], dtype=torch.float32)

NN_train_inputs = torch.cat([NN_train_inputs_phys, NN_train_inputs_T, NN_train_inputs_P, NN_train_inputs_Terr, NN_train_inputs_Perr], dim=1)
NN_valid_inputs = torch.cat([NN_valid_inputs_phys, NN_valid_inputs_T, NN_valid_inputs_P, NN_valid_inputs_Terr, NN_valid_inputs_Perr], dim=1)
NN_test_inputs  = torch.cat([NN_test_inputs_phys,  NN_test_inputs_T,  NN_test_inputs_P,  NN_test_inputs_Terr,  NN_test_inputs_Perr],  dim=1)
NN_train_outputs = torch.cat([NN_train_outputs_T, NN_train_outputs_P], dim=1)
NN_valid_outputs = torch.cat([NN_valid_outputs_T, NN_valid_outputs_P], dim=1)
NN_test_outputs  = torch.cat([NN_test_outputs_T,  NN_test_outputs_P],  dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, generator=None):
        super().__init__()
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_inputs, train_outputs, valid_inputs, valid_outputs, test_inputs, test_outputs, batch_size, rng):
        super().__init__()
        out_scaler_T = StandardScaler()
        out_scaler_P = StandardScaler()
        out_scaler_T.fit(train_outputs[:, :O].cpu().numpy())
        out_scaler_P.fit(train_outputs[:, O:].cpu().numpy())
        test_T_scaled = torch.tensor(out_scaler_T.transform(test_outputs[:, :O].cpu().numpy()), dtype=torch.float32)
        test_P_scaled = torch.tensor(out_scaler_P.transform(test_outputs[:, O:].cpu().numpy()), dtype=torch.float32)
        test_outputs = torch.cat([test_T_scaled, test_P_scaled], dim=1)
        self.out_scaler_T = out_scaler_T
        self.out_scaler_P = out_scaler_P

        i0, i1, i2, i3, i4 = 0, D, D+O, D+2*O, D+3*O
        in_scaler_phys = StandardScaler()
        in_scaler_T    = StandardScaler()
        in_scaler_P    = StandardScaler()
        in_scaler_Terr = StandardScaler()
        in_scaler_Perr = StandardScaler()
        in_scaler_phys.fit(train_inputs[:, i0:i1].cpu().numpy())
        in_scaler_T.fit(   train_inputs[:, i1:i2].cpu().numpy())
        in_scaler_P.fit(   train_inputs[:, i2:i3].cpu().numpy())
        in_scaler_Terr.fit(train_inputs[:, i3:i4].cpu().numpy())
        in_scaler_Perr.fit(train_inputs[:, i4:  ].cpu().numpy())

        def scale_inputs(X):
            X = X.cpu().numpy()
            return torch.tensor(np.hstack([
                in_scaler_phys.transform(X[:, i0:i1]),
                in_scaler_T.transform(   X[:, i1:i2]),
                in_scaler_P.transform(   X[:, i2:i3]),
                in_scaler_Terr.transform(X[:, i3:i4]),
                in_scaler_Perr.transform(X[:, i4:  ]),
            ]), dtype=torch.float32)

        self.test_inputs = scale_inputs(test_inputs)
        self.test_outputs = test_outputs
        _ = scale_inputs(train_inputs)  # fit-order parity with TP_GP_BNN.py; result unused here
        _ = scale_inputs(valid_inputs)


data_module = CustomDataModule(
    NN_train_inputs, NN_train_outputs,
    NN_valid_inputs, NN_valid_outputs,
    NN_test_inputs,  NN_test_outputs,
    batch_size, batch_rng,
)
out_scaler_T = data_module.out_scaler_T
out_scaler_P = data_module.out_scaler_P

model = NeuralNetwork(D + 4*O, nn_width, 2*O, nn_depth, generator=NN_rng)
dnn_to_bnn(model, bnn_prior_parameters)

with open(model_save_path + 'best_ckpt_path.txt', 'r') as f:
    best_ckpt_path = f.read().strip()
ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
state_dict = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
model.load_state_dict(state_dict)
model.cpu()
model.eval()
print(f"Loaded BNN checkpoint: {best_ckpt_path}")


##########################################################
#### Pick one test case and get truth vs BNN prediction ####
##########################################################
TEST_CASE_IDX = 1200

phys = NN_test_inputs_phys[TEST_CASE_IDX].numpy()
H2_bar, CO2_bar, LoD, Obliquity, Teff = phys
print(rf'Test case {TEST_CASE_IDX}: H2={H2_bar:.4g} bar, CO2={CO2_bar:.4g} bar, '
      rf'LoD={LoD:.2f} days, Obliquity={Obliquity:.1f} deg, Teff={Teff:.0f} K')

true_T = NN_test_true_T[TEST_CASE_IDX].numpy()
true_P_log10bar = NN_test_true_P[TEST_CASE_IDX].numpy()

GP_T_this = NN_test_inputs_T[TEST_CASE_IDX].numpy()
GP_P_this = NN_test_inputs_P[TEST_CASE_IDX].numpy()

x_scaled = data_module.test_inputs[TEST_CASE_IDX:TEST_CASE_IDX+1]
with torch.no_grad():
    mc_preds = np.stack([model(x_scaled).numpy()[0] for _ in range(mc_samples)], axis=0)  # (mc_samples, 2*O)

mc_resid_T = out_scaler_T.inverse_transform(mc_preds[:, :O])
mc_resid_P = out_scaler_P.inverse_transform(mc_preds[:, O:])
mc_pred_T = GP_T_this[None, :] + mc_resid_T   # (mc_samples, O)
mc_pred_P = GP_P_this[None, :] + mc_resid_P

pred_mean_T = mc_pred_T.mean(axis=0)
pred_std_T  = mc_pred_T.std(axis=0)
pred_mean_P_log10bar = mc_pred_P.mean(axis=0)
pred_std_P_log10bar  = mc_pred_P.std(axis=0)


##########################################################
#### Atmosphere composition: test-case H2/CO2 + 1 bar N2 ####
##########################################################
# ExoCAM fixes N2 partial pressure at 1 bar and defines the planet radius/gravity
# at the surface, where total pressure = p(H2) + p(CO2) + p(N2). The atmosphere is dry
# and assumed well-mixed, so volume/mass mixing ratios are constant with height.
N2_PARTIAL_PRESSURE_BAR = 1.0
MOLAR_MASS = {'H2': 2.01588, 'CO2': 44.0095, 'N2': 28.0134}  # g/mol

P0_bar = H2_bar + CO2_bar + N2_PARTIAL_PRESSURE_BAR  # total surface pressure (bar)
vmr = {
    'H2':  H2_bar / P0_bar,
    'CO2': CO2_bar / P0_bar,
    'N2':  N2_PARTIAL_PRESSURE_BAR / P0_bar,
}
mean_molar_mass = sum(vmr[s] * MOLAR_MASS[s] for s in vmr)
mmr = {s: vmr[s] * MOLAR_MASS[s] / mean_molar_mass for s in vmr}
print(f"Composition: total P0={P0_bar:.4g} bar, mu={mean_molar_mass:.3f} g/mol, "
      f"MMR H2={mmr['H2']:.4g}, CO2={mmr['CO2']:.4g}, N2={mmr['N2']:.4g}")

# Planet body
PLANET_RADIUS_CM  = 6.37e6 * 100    # 1 R_Earth, ExoCAM's fixed exoplanet radius
PLANET_GRAVITY_CGS = 9.81 * 100     # ExoCAM's fixed exoplanet surface gravity

# ExoCAM fixes instellation flux to 1360 W/m^2 regardless of stellar Teff,
# so no star-planet distance or stellar radius is implied by the GCM setup.
# 1 R_sun is used here as a fixed value for converting transit radius to transit depth
STELLAR_RADIUS_CM = 6.957e10  # 1 R_sun


##########################################################
#### Build shared pressure grid + Radtrans object ####
##########################################################
N_PRESSURE_LEVELS_PRT = 100
PRT_MIN_PRESSURE_BAR = 1e-6   # extends above the GCM's own ~0.01 bar model top; isothermally extrapolated
pressures_bar = np.logspace(np.log10(PRT_MIN_PRESSURE_BAR), np.log10(P0_bar), N_PRESSURE_LEVELS_PRT)

mass_fractions = {s: np.full(N_PRESSURE_LEVELS_PRT, mmr[s]) for s in mmr}
mean_molar_masses = np.full(N_PRESSURE_LEVELS_PRT, mean_molar_mass)

WAVELENGTH_RANGE_UM = (0.6, 5.3)  # JWST NIRSpec PRISM coverage

# The PRISM rebinning below convolves the pRT spectrum with a Gaussian LSF at
# each output wavelength; for output points near the edges of
# WAVELENGTH_RANGE_UM that kernel needs model data beyond the range itself
# (and the output grid can slightly overshoot WAVELENGTH_RANGE_UM[1], since
# it's built by stepping by 1/R until >= the upper bound). Without padding,
# those edge points are convolved against missing data and collapse toward
# ~0, which is the "bug" of transit depth cratering at the plot edges.
# Padding the pRT calculation beyond WAVELENGTH_RANGE_UM on both sides gives
# every output point real data to convolve against.
RADTRANS_WAVELENGTH_PAD_UM = 0.1
radtrans_wavelength_boundaries = [
    WAVELENGTH_RANGE_UM[0] - RADTRANS_WAVELENGTH_PAD_UM,
    WAVELENGTH_RANGE_UM[1] + RADTRANS_WAVELENGTH_PAD_UM,
]

# n68equiv (ExoCAM's radiative transfer scheme) only has line opacity for CO2;
# H2O/CH4/C2H6 are absent from this dataset.
# petitRADTRANS' only low-resolution correlated-k CO2 table is UCL-4000
# (ExoMol-based), not HITRAN2020 -- noted here as a deviation from the GCM's
# exact line list, since no HITRAN-sourced c-k CO2 table is hosted by pRT.
# H2 contributes via collision-induced absorption + Rayleigh scattering
# (pressure broadening of CO2 lines is handled implicitly by petitRADTRANS'
# own line-shape treatment, not by an explicit H2 line list, consistent with
# n68equiv where H2 has no opacity of its own either).
radtrans = Radtrans(
    pressures=pressures_bar,
    line_species=['CO2'],
    rayleigh_species=['H2', 'CO2', 'N2'],
    gas_continuum_contributors=['H2--H2', 'CO2--CO2', 'N2--N2'],
    wavelength_boundaries=radtrans_wavelength_boundaries,
    line_opacity_mode='c-k',
)

def tp_to_prt_grid(T_profile, log10_P_profile_bar):
    """Interpolate a (T, log10 P[bar]) profile onto the shared pRT pressure
    grid, in log10(P) space, extrapolating flat (isothermal) beyond the
    profile's own pressure range."""
    order = np.argsort(log10_P_profile_bar)
    log10_P_sorted = log10_P_profile_bar[order]
    T_sorted = T_profile[order]
    return np.interp(np.log10(pressures_bar), log10_P_sorted, T_sorted)

def transit_depth_ppm(temperatures):
    _, transit_radii_cm, _ = radtrans.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=PLANET_GRAVITY_CGS,
        reference_pressure=P0_bar,
        planet_radius=PLANET_RADIUS_CM,
    )
    return (transit_radii_cm / STELLAR_RADIUS_CM)**2 * 1e6

wavelengths_cm, _, _ = radtrans.calculate_transit_radii(
    temperatures=tp_to_prt_grid(true_T, true_P_log10bar),
    mass_fractions=mass_fractions,
    mean_molar_masses=mean_molar_masses,
    reference_gravity=PLANET_GRAVITY_CGS,
    reference_pressure=P0_bar,
    planet_radius=PLANET_RADIUS_CM,
)
model_wavelengths_um = wavelengths_cm * 1e4

depth_true_ppm       = transit_depth_ppm(tp_to_prt_grid(true_T,     true_P_log10bar))
depth_pred_mean_ppm  = transit_depth_ppm(tp_to_prt_grid(pred_mean_T, pred_mean_P_log10bar))
depth_pred_plus_ppm  = transit_depth_ppm(tp_to_prt_grid(pred_mean_T + pred_std_T, pred_mean_P_log10bar))
depth_pred_minus_ppm = transit_depth_ppm(tp_to_prt_grid(pred_mean_T - pred_std_T, pred_mean_P_log10bar))


##########################################################
#### Rebin to JWST NIRSpec PRISM resolution ####
##########################################################
# Approximate PRISM resolving power: R ~ 30 at 0.6um rising to R ~ 330 at
# 5.3um, linearly interpolated in wavelength. 
# This is a simplification of the real  PRISM R(lambda) curve.
def prism_resolving_power(wavelength_um):
    lam_min, lam_max = WAVELENGTH_RANGE_UM
    R_min, R_max = 30.0, 330.0
    frac = (wavelength_um - lam_min) / (lam_max - lam_min)
    return R_min + (R_max - R_min) * frac

def build_prism_wavelength_grid():
    lam_min, lam_max = WAVELENGTH_RANGE_UM
    wavelengths = [lam_min]
    while wavelengths[-1] < lam_max:
        lam = wavelengths[-1]
        R = prism_resolving_power(lam)
        wavelengths.append(lam * (1.0 + 1.0/R))
    return np.array(wavelengths)

prism_wavelengths_um = build_prism_wavelength_grid()
prism_resolutions = prism_resolving_power(prism_wavelengths_um)

def rebin_to_prism(spectrum):
    return convolve_and_sample_variable_resolution_breads(
        wavelengths=prism_wavelengths_um,
        resolutions=prism_resolutions,
        model_wavelengths=model_wavelengths_um,
        model_fluxes=spectrum,
    )

depth_true_prism       = rebin_to_prism(depth_true_ppm)
depth_pred_mean_prism  = rebin_to_prism(depth_pred_mean_ppm)
depth_pred_plus_prism  = rebin_to_prism(depth_pred_plus_ppm)
depth_pred_minus_prism = rebin_to_prism(depth_pred_minus_ppm)


##############
#### Plot ####
##############
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7),
                                gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(prism_wavelengths_um, depth_true_prism, color='blue', linewidth=1.5, label='Truth (ExoCAM)')
ax1.plot(prism_wavelengths_um, depth_pred_mean_prism, color='green', linewidth=1.5, label='BNN prediction (mean)')
ax1.fill_between(prism_wavelengths_um, depth_pred_minus_prism, depth_pred_plus_prism,
                  color='green', alpha=0.25, label='BNN prediction ($\\pm 1\\sigma$)')
ax1.set_ylabel('Transit depth (ppm)')
ax1.legend()
ax1.grid(alpha=0.3)

# Flat residuals (true - pred) on the left spine, relative residuals
# (true - pred)/true (%) on the right spine. The two aren't a fixed linear
# rescaling of one another (true depth varies with wavelength), so they're
# plotted as two independently-scaled lines sharing the x-axis, with y-limits
# set explicitly so both zero lines align.
diff       = depth_true_prism - depth_pred_mean_prism
diff_upper = depth_true_prism - depth_pred_minus_prism
diff_lower = depth_true_prism - depth_pred_plus_prism

rel_diff       = diff       / depth_true_prism * 100
rel_diff_upper = diff_upper / depth_true_prism * 100
rel_diff_lower = diff_lower / depth_true_prism * 100

ax2b = ax2.twinx()

ax2.plot(prism_wavelengths_um, diff, color='black', linewidth=1.5)
ax2.fill_between(prism_wavelengths_um, diff_lower, diff_upper, color='green', alpha=0.25)
ax2.axhline(0, color='black', linestyle='dashed', linewidth=1)
ax2.set_xlabel(r'Wavelength ($\mu$m)')
ax2.set_ylabel('Residuals (ppm)')
ax2.grid(alpha=0.3)

ax2b.plot(prism_wavelengths_um, rel_diff, alpha = 0., color='darkorange', linewidth=1.2, linestyle='--')
ax2b.set_ylabel('Relative residuals (%)', color='darkorange')
ax2b.tick_params(axis='y', labelcolor='darkorange')
ax2b.spines['right'].set_color('darkorange')

max_abs_ppm = np.nanmax(np.abs(np.concatenate([diff_lower, diff_upper])))
max_abs_pct = np.nanmax(np.abs(np.concatenate([rel_diff_lower, rel_diff_upper])))
ax2.set_ylim(-max_abs_ppm*1.1, max_abs_ppm*1.1)
ax2b.set_ylim(-max_abs_pct*1.1, max_abs_pct*1.1)

plt.suptitle(
    rf'H$_2$: {H2_bar:.4g} bar, CO$_2$: {CO2_bar:.4g} bar, $+$1 bar N$_2$   |   '
    rf'LoD: {LoD:.2f} days, Obliquity: {Obliquity:.1f} deg, Teff: {Teff:.0f} K   |   test case {TEST_CASE_IDX}'
)
plt.tight_layout()
plt.savefig(plot_save_path + f'spectrum_comparison_case{TEST_CASE_IDX}.pdf')
print(f"Saved: {plot_save_path}spectrum_comparison_case{TEST_CASE_IDX}.pdf")
plt.close()
