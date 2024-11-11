# FTIR spectra reconstruction


`process_sequence_wd_ft_interferograms.py` is a script to reconstruct FTIR spectra from raw recorded interferograms, some examples are given in `c1` and `c3` directories.

The `c3` directory contains ref. laser interferograms produced with a HeNe laser to linearize the IR interferograms stored in the `c1` directory. The signals were co-recorded during scanning using 2 channels of an oscilloscope. The post-processing algorithm includes linearization (with the use of reference interferograms of HeNe laser), filtering (Blackman–Harris), optional zero-padding, reconstruction (FFT) and Mertz correction. The output of the script can be saved and includes:
1. Wavenumber axis (full [incl. negative part] and cropped)
2. Real part of the FFT
3. Imaginary part of FFT
4. Merzt corrected mean spectrum (full and cropped)
5. Unwrapped phase (full and cropped)

`comp_index_calc.py` is a script that implements additional postprocessing steps. Since the spectrometer for which the script is written has the property of measuring phase information, this code is intended to calculate $\Delta n$ from the unwrapped $\Delta \phi$ specta and compare/overlay/verify them with the real part of the refractive index obtained using Kramers-Kronig relations. 
Example data files (obtained with `process_sequence_wd_ft_interferograms.py`) are provided.

Both scripts are provided with sample data to see them in action.

Ivan Zorin and Paul Gattinger