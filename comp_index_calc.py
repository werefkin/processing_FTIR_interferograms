# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:08:34 2023

@author: r.zor
"""
import numpy as np
import matplotlib.pyplot as plt
from pybaselines import Baseline


def kramers_kronig_transform(omega, k_omega):
    """
    Kramers-Kronig transformation to obtain the real part of the refractive index (n)
    from the imaginary part (k) as a function of frequency (omega in cm-1).

    Parameters:
        omega (ndarray): Array of frequency values.
        k_omega (ndarray): Array of the imaginary part of the refractive index corresponding to each frequency.

    Returns:
        n_omega (ndarray): Array of the real part of the refractive index corresponding to each frequency.
    """
    n_omega = np.zeros_like(omega)

    # Make the integral as in the Kramers-Kronig relations
    for i, freq in enumerate(omega):
        integral = 0.0
        for j, freq_prime in enumerate(omega):
            if freq_prime != freq:
                integral += k_omega[j] * freq_prime / (freq_prime**2 - freq**2)
        n_omega[i] = (1 / np.pi) * integral

    return n_omega


th = 6 * 1e-4  # Sample thichkness im m; required to get the k and delta_n

# Load data
phi_ref = np.load('./example/phi_ref.npy')
phi_sa = np.load('./example/phi_sa.npy')
xaxis = np.load('./example/wn_axis.npy')
xaxis_full = np.load('./example/wn_axis_full.npy')
sa = np.load('./example/spectra_sa.npy')
ref = np.load('./example/spectra_ref.npy')
# cut range
xaxis_start = xaxis[0]
xaxis_end = xaxis[-1]
ind_start = 0
ind_end = -1
ind_start_full = np.where(xaxis_full == xaxis_start)[0][0]
ind_end_full = np.where(xaxis_full == xaxis_end)[0][0] + 1
xaxis_cut = np.load('./example/wn_axis_full.npy')[ind_start_full:ind_end_full]

# Do calculations
delta_phi = phi_sa - phi_ref
delta_n = -delta_phi / (2 * np.pi * xaxis * th)  # Delta n from delta_phi
abso = -np.log(sa / ref)  # changed to natural log


plt.figure()
plt.plot(xaxis_full, abso)
plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Absorption (a.u)')
plt.title('Reconstructed spectrum')
plt.xlim([3300, 2815])
plt.show()

fig, ax = plt.subplots()
ax.plot(xaxis, abso[ind_start_full:ind_end_full])
ax.invert_xaxis()
ax.set_ylabel('absorbance / au')
ax.set_xlabel('wavenumbers / cm$^{-1}$')
plt.show()

# divide by thickness --> to make get k absorption coeff
cut_abs = abso[ind_start_full:ind_end_full] / (4 * np.pi * th * xaxis)

real_part = kramers_kronig_transform(xaxis_cut, cut_abs)

# Make baseline correction to remove a linear phase
baseline_fitter = Baseline(x_data=xaxis_cut)
base = real_part - delta_n
bkg_1 = baseline_fitter.modpoly(base, poly_order=9)[0]

plt.figure()
plt.plot(base)
plt.plot(bkg_1)
plt.show()

fig, ax = plt.subplots()
ax.plot(xaxis, real_part, label='Kramers–Kronig')
ax.plot(xaxis, delta_n + bkg_1, color='orange', label='Measurement + BL')
ax.invert_xaxis()
ax.set_ylabel('delta n')
ax.set_xlabel('wavenumbers / cm$^{-1}$')
plt.legend()
plt.show()
