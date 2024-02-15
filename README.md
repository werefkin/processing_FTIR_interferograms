# FTIR spectra reconstruction


`process_sequence_wd_ft_interferograms.py` is a script to reconstruct FTIR spectra from raw recorded interferograms (some examples are given in `c1` and `c3` folders, as co-recorded with an oscilloscope). The post-processing algorithm includes linearization (with use of reference interferograms of HeNe laser), filtering (Blackman–Harris), optional zeropadding and reconstruction (FFT). The output of the script can be saved.

`comp_index_calc.py` is a script that implements additional postprocessing steps. Since the spectrometer for which the script is written has the property of measuring phase spectra, this code is intended to calculate $\Delta n$ from the unwrapped $\Delta \phi$ specta and compare/overlay/verify them with the real part of the refractive index obtained using Kramers-Kronig relations. 
Example data files (obtained with `process_sequence_wd_ft_interferograms.py`) are provided.

Both scripts are provided with sample data to see them in action.