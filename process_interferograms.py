# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:36:24 2023

@author: Ivan Zorin
"""


import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.fft import fft
import os
from datetime import datetime

# Folders with IR interferograms (C3, in csv format) and reference
# interferograms (C4, in csv format)
folder_c3 = "./C3/"
folder_c4 = "./C4/"

# wavelength of the reference laser
lambdanehe = 632.8941914224686 * 1e-9  # 1/15800.429417*1e7

# dataframes for each folder
dataframes_c3 = []
dataframes_c4 = []

# get names of the files
file_list_c3 = sorted([f for f in os.listdir(folder_c3) if f.endswith(".csv")])
file_list_c4 = sorted([f for f in os.listdir(folder_c4) if f.endswith(".csv")])


# load IR interferograms
for file_name in file_list_c3:
    print('C3 load file: ' + file_name)
    file_path = os.path.join(folder_c3, file_name)
    df = pd.read_csv(file_path, skiprows=2, usecols=[0])
    dataframes_c3.append(df)

# load reference interferograms
for file_name in file_list_c4:
    print('C4 load file: ' + file_name)
    file_path = os.path.join(folder_c4, file_name)
    df = pd.read_csv(file_path, skiprows=2, usecols=[0])
    dataframes_c4.append(df)

# get sizes
max_length_c3 = max(len(df) for df in dataframes_c3)
max_length_c4 = max(len(df) for df in dataframes_c4)
max_length = max(max_length_c3, max_length_c4)

# switch to numpy
array_3d_c3 = np.zeros((max_length, len(dataframes_c3)))
array_3d_c4 = np.zeros((max_length, len(dataframes_c4)))

# switch to numpy
for i, df in enumerate(dataframes_c3):
    print('C3 converted filenum: ' + str(i))
    array_3d_c3[:len(df), i] = df.iloc[:, 0]

for i, df in enumerate(dataframes_c4):
    print('C4 converted filenum: ' + str(i))
    array_3d_c4[:len(df), i] = df.iloc[:, 0]


# Convert to numpy arrays
array1 = array_3d_c3
array2 = array_3d_c4

# standard deviation (sigma) to slightly denoise the reference interferogram
sigma = 6

# Define the number of elements to keep on each side
crop_size = 12000  # adjust this value as needed
N = 2 * crop_size

# empthy array for reconstructed spectra
spectra = np.zeros([crop_size * 2, np.shape(array1)[1]])


# the more simple way:
upper_frequ = 1 / (lambdanehe * 1e2)
delta_nu = upper_frequ / N  # 1e-2 to convertit to cm
# *2 because the fft gives the spectrum twice and i dont cut it
wn_axis = 2 * np.arange(N) * delta_nu

for i in range(0, np.shape(array1)[1]):
    # Apply Gaussian filter
    array_ref = gaussian_filter(array1[:, i], sigma)

    # Find peaks
    peaks, _ = scipy.signal.find_peaks(array_ref, height=0.65, distance=30)

    # Find valleys (invert the array and find peaks in the inverted signal)
    inverted_array_ref = -array_ref
    valleys, _ = scipy.signal.find_peaks(
        inverted_array_ref, height=-0.65, distance=30)

    # Combine indices of peaks and valleys
    combined_indices = sorted(np.concatenate((peaks, valleys)))

    # Sort the combined indices

    # Linear interferogram
    # This means we sample the IR interferogram at equidistant points provided
    # by the reference monochromatic laser (peaks and valleys)
    interferogram = array2[combined_indices, i]

    # Find the index of the central Burst
    max_index = np.argmax(interferogram)

    centered_interf = interferogram[max_index -
                                    crop_size:max_index + crop_size]

    bh_window = np.blackman(len(centered_interf))  # blackman_harris window
    kernel = (centered_interf - np.mean(centered_interf)) * (bh_window)
    spec = fft(kernel, norm="ortho")  # fourier transform with zerofilling
    spectra[:, i] = np.abs(spec)
    plt.plot(wn_axis, np.abs(spec))
    plt.xlim([2126, 3400])
    plt.ylim([0, None])
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('PSD (a.u)')
    plt.title('Reconstructed spectrum')
    plt.savefig('reconstructed_spectrum.png', dpi=300, bbox_inches='tight')

# generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# save spectra as npy
filename = f"spectra_{timestamp}.npy"
np.save(filename, spectra)

plt.figure()
background = np.mean(spectra, axis=1)
plt.plot(wn_axis, background)
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('PSD (a.u)')
plt.title('Reconstructed spectrum')
plt.xlim([2126, 3400])
plt.ylim([0, None])
