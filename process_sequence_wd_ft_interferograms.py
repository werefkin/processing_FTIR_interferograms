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
from scipy.fft import fft  # , fftfreq, fftshift
import os
from datetime import datetime
from scipy import signal

# Init params
lambdanehe = 632.8941914224686 * 1e-9  # 1/15800.429417*1e7 REF LASER
crop_size = 50000  # Adjust this value as needed
pad_factor = 4

# Define the folder paths for reference signal (HeNe laser C3) and IR interferograms (C4)
folder_c3 = "./c3/"
folder_c4 = "./c4/"

# Initialize lists to store dataframes for each folder
dataframes_c3 = []
dataframes_c4 = []

# Get the list of file names in each folder
file_list_c3 = sorted(
    [f for f in os.listdir(folder_c3) if f.endswith(".csv")])[1:]
file_list_c4 = sorted(
    [f for f in os.listdir(folder_c4) if f.endswith(".csv")])[1:]


# Load data into dataframes for C3
for file_name in file_list_c3:
    print('C3 load file: ' + file_name)
    file_path = os.path.join(folder_c3, file_name)
    df = pd.read_csv(file_path, skiprows=2, usecols=[0])
    dataframes_c3.append(df)

# Load data into dataframes for C4
for file_name in file_list_c4:
    print('C4 load file: ' + file_name)
    file_path = os.path.join(folder_c4, file_name)
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
sigma = 0  # Adjust the standard deviation (sigma) as needed
# Define the number of elements to keep on each side

spectra = np.zeros([crop_size * 2 * pad_factor, np.shape(array1)[1]])


N = 2 * crop_size
# the more simple way:
upper_frequ = 1 / (lambdanehe * 1e2)
delta_nu = upper_frequ / N  # 1e-2 to convertit to cm
# *2 #because the fft gives the spectrum twice and i dont cut it
wn_axis = 2 / pad_factor * np.arange(N * pad_factor) * delta_nu
bh_window = signal.windows.blackmanharris(N)  # blackman_harris window


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
    max_index = np.argmin(interferogram)

    centered_interf = interferogram[max_index -
                                    crop_size:max_index + crop_size] * (bh_window)

    centered_interf_padded = np.pad(centered_interf,
                                    (crop_size * (pad_factor - 1),
                                     crop_size * (pad_factor - 1)),
                                    mode='constant')
    kernel = (centered_interf_padded -
              np.mean(centered_interf_padded))

    # abs of fourier transform with zerofilling
    spectra[:, i] = np.abs(fft(kernel, norm="ortho"))
    plt.plot(wn_axis, np.abs(spectra[:, i]))
    plt.xlim([2126, 3400])
    plt.ylim([0, None])
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('PSD (a.u)')
    plt.title('Reconstructed spectrum')
    plt.savefig('reconstructed_spectrum.png', dpi=300, bbox_inches='tight')

# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# Define the filename with the timestamp
filename = f"spectra_pc_{timestamp}.npy"


ind_min = np.where(wn_axis < 2000)[0][-1]
ind_max = np.where(wn_axis > 4500)[0][1]

# Save the array with the timestamp in the filename
np.save(filename, spectra[:, ind_min:ind_max])
np.save('wn_axis.npy', wn_axis[ind_min:ind_max])

plt.figure()
background = np.mean(spectra, axis=1)
plt.plot(wn_axis[ind_min:ind_max], background[ind_min:ind_max])
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('PSD (a.u)')
plt.title('Reconstructed spectrum')
plt.xlim([2126, 3400])
plt.ylim([0, None])
