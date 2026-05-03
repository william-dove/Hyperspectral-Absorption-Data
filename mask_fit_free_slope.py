'''  
Adjusted fitting script which does not require theory-consistent slope.
'''

import sys
import os
import h5py
from tqdm import tqdm
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
np.seterr(divide='ignore', invalid='ignore') # Ignore numpy complaining about divide by zero
from scipy.ndimage import (
    gaussian_filter1d,
    label,
    find_objects,
    binary_closing,
    binary_opening,
    binary_dilation
)
import numpy as np
from skimage.measure import block_reduce
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import warnings

# Constants
h = 4.1357e-15      # eV.s (Plank's constant)
c = 299792458.      # m/s (speed of light)
k_B = 8.61733e-5    # eV/K (Boltzmann's constant)
T = 293.            # K (temperature)
# Fitting constants
intended_slope = 1 / (k_B * T)
min_width = 25
thresh = 50 # 80 didn't work
Sx = 512# Cheating because I know the window size already
Sy = 512

# -------------------------------------------------


def fit(energy, amp, idx_start, idx_stop):
        """
        For each point of the cube, the high-energy tail of the curve is selected, and a linear regression is performed.

        :param energy_i: the numpy array of photon energies
        :param amp: the numpy array of y-data to be fitted.
        :param start: the starting index of the high-energy tail to be fitted
        :param stop: the stopping index of the high-energy tail to be fitted
        :return intercept: the intercept of the fitted line (which is the quasi fermi-level divided by - k_B * T)
        :return slope: the slope of the fitted line
        :return R2: the R^2 value of the fit
        """

        # Crop the data to the (manually selected) high-energy tail
        amp_tail = amp[idx_start:idx_stop]
        energy_tail = energy[idx_start:idx_stop]

        if len(energy_tail) <= 1:
            intercept = 0
            slope = 0
            R2 = 0
        else:
            # Perform curve fitting of a 1st degree polynomial
            intercept, slope = poly.polyfit(energy_tail, amp_tail, 1)
            fitted_line = intercept + slope * energy_tail
            ss_res = np.sum((amp_tail - fitted_line)**2)
            ss_tot = np.sum((amp_tail - np.mean(amp_tail))**2)
            R2 = 1 - ss_res / ss_tot 

        return intercept, slope, R2

def contiguous_regions(mask, idx_peak, min_width):
    '''    
    Finds good potential windows given the boolean mask where the data is considered linear.
    idx_peak and mask are both taken after amp is scanned for infs and nans.
    '''
    regions = []
    rel_start = None  # keep track of whether a linear region has already started being tracked.

    for i, val in enumerate(mask): # val = true or false. 
        if (val) and (rel_start is None):  # If val=True (i.e., the point is linear) AND we haven't started already, mark this as a starting point.
            rel_start = i

        elif (not val) and (rel_start is not None): # Once we have started, this won't activate until we hit a val=False (i.e., non-linear) point. We've found a whole region.
            if i - rel_start >= min_width:
                regions.append((rel_start+idx_peak, i+idx_peak))
            rel_start = None

    # handle tail case
    if (rel_start is not None) and (len(mask) - rel_start >= min_width): # Once the final region has been encountered, the loop will have ended and this will activate.
        regions.append((rel_start+idx_peak, len(mask)+idx_peak))

    # Warn if no regions are being selected
    if len(regions) == 0:
        warnings.warn('Warning: The selected curvature percentile presented no linear windows. Consider decreasing the threshold.')
    return regions

def contiguous_smoothing(
    mask,
    score=None,
    idx_peak=0,
    min_width=15,
    closing_size=5,
    opening_size=3,
    dilation_size=2,
    max_gap_fill=True,
    return_scores=False,
):
    """
    Robust extraction of contiguous 'good' regions from a noisy boolean mask.

    Parameters
    ----------
    mask : 1D boolean array
        Binary mask of "good" points (e.g., linear regions)
    score : 1D float array (optional)
        Continuous score (e.g., -|curvature|). Used to rank regions.
    idx_peak : int
        Offset to map tail indices back to full array
    min_width : int
        Minimum allowed region width
    closing_size : int
        Size for morphological closing (fills small gaps)
    opening_size : int
        Size for morphological opening (removes noise spikes)
    dilation_size : int
        Expands regions slightly to recover edges
    max_gap_fill : bool
        Whether to aggressively merge small gaps
    return_scores : bool
        Whether to return region scores

    Returns
    -------
    regions : list of tuples
        [(start, stop, score), ...] sorted by score descending
    """

    mask = np.asarray(mask, dtype=bool)

    # --- Step 1: Morphological smoothing ---
    if closing_size > 1:
        mask = binary_closing(mask, structure=np.ones(closing_size))

    if opening_size > 1:
        mask = binary_opening(mask, structure=np.ones(opening_size))

    if dilation_size > 0:
        mask = binary_dilation(mask, structure=np.ones(dilation_size))

    # --- Step 2: Label connected regions ---
    labeled, _ = label(mask)
    slices = find_objects(labeled)

    regions = []

    for slc in slices:
        start = slc[0].start
        stop = slc[0].stop
        width = stop - start

        if width < min_width:
            continue

        if score is not None:
            region_score = np.mean(score[start:stop])
        else:
            region_score = width  # fallback: longer = better

        regions.append((start + idx_peak, stop + idx_peak, region_score))

    # --- Step 3: Fallback if nothing found ---
    if len(regions) == 0:
        # Relax criteria automatically
        if score is not None:
            best_idx = np.argmax(score)
        else:
            best_idx = len(mask) // 2

        start = max(0, best_idx - min_width // 2)
        stop = min(len(mask), start + min_width)

        regions.append((start + idx_peak, stop + idx_peak, -np.inf))

    # --- Step 4: Sort best → worst ---
    regions.sort(key=lambda x: -x[2])

    return regions if return_scores else [(r[0], r[1]) for r in regions]


def process_pixel(i, energy, amp):

    # Clean data
    mask = np.isfinite(amp)
    if not np.any(mask):
        return None
    amp = amp[mask]
    energy = energy[mask]

    # Crop to high-energy tail
    idx_peak = np.argmin(amp)
    amp_tail = amp[idx_peak:]
    
    # Finding windows
    curvature = np.gradient(np.gradient(amp_tail))
    curvature = gaussian_filter1d(curvature, sigma=2) # Smooth the curvature to reduce noise
    score = -np.abs(curvature)
    threshold = np.percentile(score, thresh) # Returns the value below which {thresh}% of the scores lie.
    linear_mask = score > threshold # Returns a boolean list which is true at the indices in the top 20% of linearity scores.
    # candidate_windows = contiguous_regions(linear_mask, idx_peak, min_width) # Finds good candidate windows within the linear regime of the data.
    # Replaces the ^above^ with the cooler version vvv
    candidate_windows = contiguous_smoothing(
        mask=linear_mask,
        score=score,
        idx_peak=idx_peak,
        min_width=min_width
    )

    # Iterate through potential windows, checking for decent r2
    decent_fits = []
    r2s = []
    for idx_start, idx_stop in candidate_windows:
        intercept, slope, r2 = fit(energy, amp, idx_start, idx_stop)
        if r2 >= 0.95:
                decent_fits.append([idx_start, idx_stop, intercept, slope, r2])
                r2s.append(r2)

    # Finding the best fit based on r2 (no slope consideration)
    x = i // Sy
    y = i % Sy
    if len(decent_fits) == 0:
        return [x, y] + [np.nan]*7
    else:
        r2s = np.array(r2s)
        best_idx = np.argmax(r2s)
        best_fit = decent_fits[best_idx]
        start_energy = energy[best_fit[0]]
        stop_energy = energy[best_fit[1]]
        return [x, y] + decent_fits[best_idx] + [start_energy, stop_energy]

def main():
    # Importing the data
    if len(sys.argv) >= 2:
        spectra_file = sys.argv[1]
        if not os.path.isfile(spectra_file):
            raise ValueError('.h5 file not found.')
    else:
        raise ValueError('Please input the desired .h5 file.')
    
    # Read photoluminescence intensity spectra and wavelengths
    cube = h5py.File(spectra_file, 'r')
    wvl = np.array(cube['Cube']['Wavelength'])           # nm (wavelengths)
    full_intensity = np.array(cube['Cube']['Images'])    # photons / cm^2.s.eV (Photoluminescence intensity spectra)
    full_intensity *= 1e4                                # Conversion factor from cm^2 to m^2
    full_intensity = np.abs(full_intensity)              # Take absolute values of intensity
    # Reducing image size via averaging
    bin_size = 2
    intensity = block_reduce(full_intensity, block_size=(1, bin_size, bin_size), func=np.mean) # '1' is for keeping a single, average intensity spectrum from the 4 pixels
    # Convert wavelengths to photon energy in eV
    photon_energy = h * c / (wvl * 1e-9)[::-1] # eV
    # Get the curve that will be fitted to obtain the fermi quasi-level split
    a = (2 * np.pi * photon_energy[:, None, None] ** 2) / (intensity * h ** 3 * c ** 2)  # Get argument of logarithm
    curve_to_fit = np.log(a) # ln(E^2 / I) vs. E (where E = photon energy, I = how much absorbance (intensity) for these photons)
    # Reshape the 3d data into 2d data to facilitate iterations
    Sw, Sx, Sy = np.shape(curve_to_fit) # Sw = length of wavelength/energy and absorbance/intensity vectors, and therefore curve_to_fit. Sx and Sy: Number of pixels in x and y directions.
    curve_to_fit_2d = curve_to_fit.reshape((Sw, Sx * Sy)) # 'Flattens' the image into a long 1D list of curve_to_fit vectors for each pixel (i.e., a 2D array all in all).
    data = curve_to_fit_2d.T # Optimized for slicing

    # Process all pixels (no parallel for now just to see...)
    print('Processing pixels...')
    best_fit_rows = []
    for i in tqdm(range(Sx*Sy)):
        best_fit_row = process_pixel(i, photon_energy, data[i])
        best_fit_rows.append(best_fit_row)    
    print('Fitting complete! Saving results...')
    
    # Saving results to dataframe
    columns = ['x', 'y', 'start idx', 'stop idx', 'intercept', 'slope', 'R2', 'start energy', 'stop energy']
    best_fits_df = pd.DataFrame(best_fit_rows, columns=columns)

    # Calculating relevant info
    best_fits_df['idx width'] = best_fits_df['stop idx'] - best_fits_df['start idx']
    best_fits_df['energy width'] = best_fits_df['stop energy'] - best_fits_df['start energy']
    best_fits_df['qfls'] = best_fits_df['intercept'] * -k_B * T

    # Saving results to .csv
    path, filename_h5 = os.path.split(spectra_file)
    if len(sys.argv) == 2:
        filename = os.path.splitext(filename_h5)[0]
        results_filename = filename + '_best_fits-min_window_25-512x512_bins-free_slope.csv'
    elif len(sys.argv) == 3:
        results_filename = sys.argv[2]
    else:
        raise ValueError('Ya done fucked up.')
    save_path = os.path.join(path, results_filename)
    best_fits_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()