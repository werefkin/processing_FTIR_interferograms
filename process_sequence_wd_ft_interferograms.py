# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:53:18 2023

@author: Ivan Zorin
"""


import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.fft import fft, fftshift  # , fftfreq, fftshift
import os
from datetime import datetime


# Init params
lambdanehe = 632.8941914224686 * 1e-9  # 1/15800.429417*1e7 REF LASER
crop_size = 30000  # Adjust this value as needed
pad_factor = 4
sigma = 0  # Adjust the standard deviation (sigma) as needed
save = False

# Plot each FT result
plotft = False


# Define the folder paths for reference signal (HeNe laser C3) and IR interferograms (C4)
ref_laser_folder = "./example/c3/"
interferograms_folder = "./example/c1/"


# Get frequency axis
N = 2 * crop_size * pad_factor
upper_frequ = 1 / (lambdanehe / 2 * 1e2)  # *2 #because we sample
delta_nu = upper_frequ / N  # 1e-2 to convertit to cm
# Length of the FFT result
deltaf = 1 / (N * delta_nu)
freqs = fftshift(np.fft.fftfreq(N, d=deltaf))  # shift to make a plot with no artifacts (in the natural order)


# Initialize lists to store dataframes for each folder
dataframes_c3 = []
dataframes_c4 = []


# Get the list of file names in each folder
file_list_c3 = sorted(
    [f for f in os.listdir(ref_laser_folder) if f.endswith(".csv")])[2:]
file_list_c4 = sorted(
    [f for f in os.listdir(interferograms_folder) if f.endswith(".csv")])[2:]


# Load data into dataframes for C3
for file_name in file_list_c3:
    print('Ref load file: ' + file_name)
    file_path = os.path.join(ref_laser_folder, file_name)
    df = pd.read_csv(file_path, skiprows=2, usecols=[0])
    dataframes_c3.append(df)

# Load data into dataframes for C4
for file_name in file_list_c4:
    print('Interferograms load file: ' + file_name)
    file_path = os.path.join(interferograms_folder, file_name)
    df = pd.read_csv(file_path, skiprows=2, usecols=[0])
    dataframes_c4.append(df)

# Determine the maximum length of the dataframes
max_length_c3 = max(len(df) for df in dataframes_c3)
max_length_c4 = max(len(df) for df in dataframes_c4)
max_length = max(max_length_c3, max_length_c4)

# Initialize 3D NumPy arrays filled with zeros
array_3d_c3 = np.zeros((max_length, len(dataframes_c3)))
array_3d_c4 = np.zeros((max_length, len(dataframes_c4)))

# Fill the 3D arrays with data from the dataframes
for i, df in enumerate(dataframes_c3):
    print('C3 converted filenum: ' + str(i))
    array_3d_c3[:len(df), i] = df.iloc[:, 0]

for i, df in enumerate(dataframes_c4):
    print('C4 converted filenum: ' + str(i))
    array_3d_c4[:len(df), i] = df.iloc[:, 0]


# Convert to numpy arrays
array1 = array_3d_c3
array2 = array_3d_c4
# Define the number of elements to keep on each side

# Initialize empthy arrays for the data
spectra = np.zeros([crop_size * 2 * pad_factor, np.shape(array1)[1]])
kernels = np.zeros([crop_size * 2 * pad_factor, np.shape(array1)[1]])
angle_spectra = np.zeros([crop_size * 2 * pad_factor, np.shape(array1)[1]])


bh_window = np.blackman(2 * crop_size)  # blackman_harris window
# Resampling, windowing, padding
for i in range(0, np.shape(array1)[1]):
    # Apply Gaussian filter
    array_ref = gaussian_filter(array1[:, i], sigma)  # reference

    # Find peaks
    peaks, _ = scipy.signal.find_peaks(array_ref, height=0.65, distance=3)

    # Find valleys (invert the array and find peaks in the inverted signal)
    valleys, _ = scipy.signal.find_peaks(-array_ref, height=-0.65, distance=3)

    # Combine indices of peaks and valleys
    combined_indices = sorted(np.concatenate((peaks, valleys)))

    # Sort the combined indices

    # Linear interferogram
    interferogram = array2[combined_indices, i]

    # Find the index of the maximum value in array2
    max_index = np.argmax(interferogram)

    centered_interf = (interferogram[max_index - crop_size:max_index + crop_size] - np.mean(interferogram[max_index - crop_size:max_index + crop_size])) * (bh_window)
    centered_interf_padded = np.pad(centered_interf,
                                    (crop_size * (pad_factor - 1), crop_size * (pad_factor - 1)),
                                    mode='constant')

    kernel = (centered_interf_padded - np.mean(centered_interf_padded))
    kernels[:, i] = kernel


#  FFT
for i in range(0, np.shape(array1)[1]):
    print('Interferograms processed: ', i)
    spec = fftshift(fft(fftshift(kernels[:, i]), norm="ortho"))  # fourier transform with zerofillingng
    spectra[:, i] = np.real(spec)
    angle_spectra[:, i] = np.angle(spec)

    if plotft is True:
        plt.plot(freqs, np.abs(spectra[:, i]))
        plt.xlim([2126, 3400])
        plt.ylim([0, None])
        plt.xlabel('Wavenumber (cm^-1)')
        plt.ylabel('PSD (a.u)')
        plt.title('Reconstructed spectrum')
        plt.savefig('reconstructed_spectrum.png', dpi=300, bbox_inches='tight')

mean_psd = np.mean(spectra, axis=1)

# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# Define the filename with the timestamp
filename = f"spectra_pc_{timestamp}.npy"


ind_min = np.where(freqs < 2650)[0][-1]
ind_max = np.where(freqs > 3080)[0][1]

# Work with phase spectra
phase = np.mean(angle_spectra, axis=1)
phi = np.unwrap(phase[ind_min:ind_max])

plt.figure()
plt.plot(freqs, mean_psd)
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('PSD (a.u)')
plt.title('Reconstructed spectrum')
plt.xlim([2126, 3400])
plt.ylim([0, None])
plt.show()

plt.figure()
plt.plot(freqs, phase)
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Phase spectrum (a.u)')
plt.title('Reconstructed spectrum')
plt.show()

if save is True:
    np.save(r'phi_sa.npy', phi)
    np.save('wn_axis.npy', freqs[ind_min:ind_max])
    np.save('wn_axis.npy', freqs[ind_min:ind_max])
    np.save('wn_axis_full.npy', freqs)
    # Save the array with the timestamp in the filename
    np.save(filename, spectra[:, ind_min:ind_max])
    np.save('wn_axis.npy', freqs[ind_min:ind_max])

    np.save(filename, spectra)
    np.save('wn_axis.npy', freqs)
