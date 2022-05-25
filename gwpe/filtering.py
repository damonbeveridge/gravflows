import argparse
from pathlib import Path

from typing import Optional, Dict, Union

import logging

import pandas as pd
import scipy.stats as st

import numpy as np
from numpy.lib.format import open_memmap

import multiprocessing
import concurrent.futures

from functools import partial

from datetime import datetime

from pycbc.filter import matched_filter
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform
from tqdm import tqdm

from gwpe.utils import read_ini_config

# TODO: Learn and understand the different polarisations of strain and if there is a way to combine them or if there is a parallel for SNR time-series
# TODO: We should make a check at the very start to see if the highest chirp mass injection (and hence templates) are capable of the level of early warning requested
# TODO: Also implement early warning such that neglat=0 means no early warning, not merger time cutoff


def chirp_mass(m1, m2):
    return ((m1 * m2)**0.6)/ (m1 + m2)**0.2


def compute_snrs(
        chunk_data,
        static_args: Dict[str, Union[str, float]],
):
    # Takes in a list of sample chirp_masses to sample filters for and percentage chirp mass range
    # Loop over chirp_masses
    # Get best template match by chirp mass
    # Get upper and lower template IDs from chirp mass percentage range
    # Construct split normal PDF
    # Sample n filters from PDF and account for best/optimal filters if required
    """Function samples template filters around the best chirp mass match, based on a split normal distribution covering a specified percentage range in chirp mass.

    Arguments:
        intrinsic: Union[np.record, Dict[str, float]]
            A one dimensional vector (or dictionary of scalars) of intrinsic parameters that parameterise a given waveform.
            We require sample to be one row in a because we want to process one waveform at a time,
            but we want to be able to use the PyCBC prior distribution package which outputs numpy.records.
        static_args: Dict[str, Union[str, float]]
            A dictionary of type-casted (from str to floats or str) arguments loaded from an .ini file.
            We expect keys in this dictionary to specify the approximant, domain, frequency bands, etc.
        inclination: bool
        spins: bool
        spins_aligned: bool
        downcast: bool
            If True, downcast from double precision to full precision.
            E.g. np.complex124 > np.complex64 for frequency, np.float64 > np.float32 for time.

    Returns:
        (hp, hc)
            A tuple of the plus and cross polarizations as pycbc Array types
            depending on the specified waveform domain (i.e. time or frequency).
    """

    snrs = []

    projections = chunk_data[0]
    filters = chunk_data[1]

    for proj in projections:

        temp = []

        # convert sample to PyCBC time series
        strain_time_series = TimeSeries(proj,
                                        delta_t=static_args['delta_t'], epoch=0,
                                        dtype=None, copy=True)
        # convert sample to PyCBC frequency series
        strain_freq_series = strain_time_series.to_frequencyseries()

        for template in filters:

            # Generate optimal matched filtering template
            template_hp, template_hc = get_td_waveform(
                approximant=static_args['approximant'],
                mass1=template[0],
                mass2=template[1],
                spin1z=template[2],
                spin2z=template[3],
                f_lower=static_args['f_lower'],
                delta_t=static_args['delta_t'],
            )
            # Convert template to PyCBC frequency series
            template_freq_series_hp = template_hp.to_frequencyseries(
                delta_f=strain_freq_series.delta_f)
            # Resize template to work with the sample
            template_freq_series_hp.resize(len(strain_freq_series))

            # Time shift the template so that the SNR peak matches the merger time
            template_freq_series_hp = template_freq_series_hp.cyclic_time_shift(template_freq_series_hp.start_time)

            # Compute SNR time-series from optimal matched filtering template
            snr_series = matched_filter(template_freq_series_hp,
                                        strain_freq_series.astype(complex),
                                        psd=None, low_frequency_cutoff=static_args['f_lower'])

            temp.append(np.array(abs(snr_series)))

        cms = []
        for template in filters:
            cms.append(chirp_mass(template[0], template[1]))

        snrs.append(temp)

    return snrs


def generate_filter_parameters(
        static_args_ini: str,
        data_dir: str='data',
        out_dir: Optional[str]=None,
        projections_file: str='projections.npy',
        filters_file: str='filters.npy',
        overwrite: bool=False,
        chunk_size: int=10,
        workers: int=1,
        verbose: bool=True,
):
    """Convenience function to generate parameters from a ParameterGenerator
    object and save them to disk as a .csv file.

    Can also copy corresponding PyCBC prior distribution metadata for completeness.

    Arguments:
        n: int
            Number of templates per sample.
        omf: bool=False
            If true, it adds the optimal matched filter to the list (identical to sample parameters).
        config_files: List[str] or str.
            A file path to a compatible PyCBC params.ini config files.
        data_dir: str
            The output directory to save generated filter parameter data.
        file_name: str
            The output .csv file name to save the generated filter parameters.
        overwrite: bool=True
            If true, completely overwrites the previous directory specified at data_dir.

    """
    # load parameters
    data_dir = Path(data_dir)

    # specify precision of output waveforms
    # dtype = np.float64 if not downcast else np.float32
    dtype = np.float64

    # load waveforms file
    projections = open_memmap(data_dir / projections_file, dtype=dtype, mode='r')
    n_samples = len(projections)
    chunks = int(np.ceil(n_samples/chunk_size))

    # load filters file
    filters = open_memmap(data_dir / filters_file, dtype=dtype, mode='r')
    n_templates = len(filters[0])

    # load static argument file
    _, static_args = read_ini_config(static_args_ini)

    # specify output directory
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)

    # check output numpy arrays
    snrs_file = out_dir / 'snrs.npy'
    if not overwrite:
        if snrs_file.is_file():
            logging.debug(f'Aborting - {snrs_file} exists but overwrite is False.')
            return

    # sample_length = int(2 * static_args["target_sampling_rate"]) # TODO: Set this up as a configurable length in seconds
    sample_length = int(9 * static_args["target_sampling_rate"])

    snrs_memmap = open_memmap(
        filename=snrs_file,
        mode='w+',  # create or overwrite file
        dtype=dtype,
        shape=(n_samples, 2, n_templates, sample_length)  # 2 corresponds to the separate complex values of the time-series
    )

    num_elements = snrs_memmap.size
    num_bytes = num_elements * 8 + 128  # 64 bit floats so 8 bytes for element
    num_gbytes = num_bytes / (1024 ** 3)
    print(f"Estimated SNRs file size is {round(num_gbytes, 2)} GB. " +
          "If this number is too large for your system, please abort the script and adjust your inputs.")

    # multiprocessing generation
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # with open_memmap(data_dir / projections_file, dtype=dtype, mode='r') as projections, \
        #         open_memmap(data_dir / filters_file, dtype=dtype, mode='r') as filters:

        # get parameters from waveforms file
        n_samples = len(projections)
        chunks = int(np.ceil(n_samples/chunk_size))

        # get parameters from filters file
        n_templates = len(filters[0])

        # create buffer in memory to temporarily store before writing to desk
        snrs = np.empty((chunk_size, 2, n_templates, sample_length), dtype=dtype)

        # loop through samples at approx. 10GB at a time and append to numpy array
        progress_desc = f"[{datetime.now().strftime('%H:%M:%S')}] Generating template filter parameters"
        saved = 0  # track number of saved arrays for progress bar
        with tqdm(desc=progress_desc, total=n_samples, miniters=1, postfix={'saved': saved}, disable=not verbose) as progress:
            for i in range(chunks):
                # get index positions for chunks
                start = i*chunk_size
                end = (i+1)*chunk_size

                # check if chunk_size goes over length of samples
                if end > n_samples:
                    # overflow batch may not have a full chunk size - we need to re-instantiate waveform array
                    end = end - chunk_size + (n_samples % chunk_size)

                    snrs = np.empty((end - start, 2, n_templates, sample_length), dtype=dtype)

                # get a chunk of samples
                saved += end - start
                chunk_samples = projections[start:end]
                chunk_filters = filters[start:end]
                chunk_data = zip(chunk_samples, chunk_filters)

                # submit waveform generation jobs to multiple processes while tracking source parameters by index
                filter_sampling_job = partial(
                    compute_snrs,
                    static_args=static_args,
                )

                # store waveform polarisations in correct order while updating progress bar as futures complete
                ordered_futures = {executor.submit(filter_sampling_job, params): i for i, params in enumerate(chunk_data)}  # Looks like this line submits a new job for every item in the current chunk of samples
                # print(ordered_futures)
                for future in concurrent.futures.as_completed(ordered_futures):
                    idx = ordered_futures[future]
                    snrs[idx] = np.stack(future.result())  # assign (n, 4) to array idx
                    progress.update(1)
                progress.refresh()

                snrs_memmap[start:end, :, :] = snrs

                # notify timer that batch has been saved
                progress.set_postfix(saved=saved)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sampling matched filtering templates.')

    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')
    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='The input directory to load sample parameter files.')
    parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, help='The output directory to save generated SNR files.')
    parser.add_argument('-p', '--projections_file', dest='projections_file', default='projections.npy', type=str, help='The input .npy file of generated waveforms projected onto noise.')
    parser.add_argument('-f', '--filters_file', dest='filters_file', default='filters.npy', type=str, help='The input .npy file of sampled template filters for matched filtering.')
    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")  # Technically 'default=False' is not needed as the action takes care of this, but I keep it for clarity

    # multiprocessing
    parser.add_argument('-c', '--chunk_size', type=int, default=1000, help='The number of samples to produce filters for before appending to disk.')
    parser.add_argument('-w', '--workers', type=int, default=12, help='The number of workers to use for Python multiprocessing.')


    # random seed
    # parser.add_argument('--seed', type=int")  # to do

    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode.")

    args = parser.parse_args()

    assert args.data_dir is not None, "Output data directory must be provided with -d or --data_dir."
    assert args.static_args_ini is not None, "Static arguments .ini file must be provided with -s or --static_args."

    generate_filter_parameters(**args.__dict__)