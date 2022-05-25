import argparse
from pathlib import Path

from typing import Optional

import logging

import pandas as pd
import scipy.stats as st

import numpy as np
from numpy.lib.format import open_memmap

import multiprocessing
import concurrent.futures

from functools import partial

from datetime import datetime

from tqdm import tqdm

from gwpe.utils import read_ini_config


def chirp_mass(m1, m2):
    return ((m1 * m2)**0.6)/ (m1 + m2)**0.2


def sample_filter_params(
    chirp_masses,
    cm_range: float,
    templates,
    n: int,
    best_filter: bool=True,
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

    filters = []

    filter_idxs = np.zeros(shape=(len(chirp_masses), n), dtype=int)

    if best_filter:
        n -= 1

    for key, cm in enumerate(chirp_masses):
    # cm = chirp_masses

        # filter_idxs = np.zeros(shape=n, dtype=int)

        best_idx = np.searchsorted([i[0] for i in templates], cm)
        low_idx = np.searchsorted([i[0] for i in templates], cm*(1 - cm_range))
        high_idx = np.searchsorted([i[0] for i in templates], cm*(1 + cm_range))

        # print("best")
        # print(best_idx)
        # print("low")
        # print(low_idx)
        # print("high")
        # print(high_idx)

        if n > high_idx - low_idx:
            filters.append(np.zeros(shape=(1, n, 4)))
            # return np.zeros(shape=(n, 4))

        if best_filter:
            filter_idxs[key][-1] = best_idx

        sigma_low = int((best_idx - low_idx) / 2)
        sigma_high = int((high_idx - best_idx) / 2)

        pdf_range = np.arange(low_idx, high_idx)
        pdf_low = st.truncnorm.pdf(pdf_range, (low_idx - best_idx) / sigma_low, (high_idx - best_idx) / sigma_low, loc=best_idx, scale=sigma_low)
        pdf_high = st.truncnorm.pdf(pdf_range, (low_idx - best_idx) / sigma_high, (high_idx - best_idx) / sigma_high, loc=best_idx, scale=sigma_high)

        diff = best_idx - low_idx  # This is the index of the array (not template idx) of the best template
        scale_low = 0.5 / pdf_low[:diff].sum()
        scale_high = 0.5 / pdf_high[diff:].sum()
        pdf_split = np.concatenate((scale_low * pdf_low[:diff], scale_high * pdf_high[diff:]))

        best_idx_in_samples = True  # Faking it to start with
        while best_idx_in_samples:
            samples = np.random.choice(pdf_range, size=n, p=pdf_split)
            samples = sorted(samples)
            if best_idx not in samples:
                best_idx_in_samples = False

        for i in range(len(samples)):
            filter_idxs[key][i] = int(samples[i])

        # Now we need to get the filter parameters for each index
        temp = []
        for idx in filter_idxs[key]:
            temp.append(templates[idx][1:])  # The `[1:]` cuts out the chirp mass parameter that we don't care about
        filters.append(temp)
        # filters = temp

    # return np.zeros(shape=(2, 3))
    # print(filters)
    # print(np.shape(filters))
    # print(np.stack(filters))
    return filters


def generate_filter_parameters(
        n: int,
        cm_range: float,
        static_args_ini: str,
        template_file: str,
        best_filter: bool=False,
        data_dir: str='data',
        out_dir: Optional[str]=None,
        params_file: str='parameters.csv',
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
    parameters = pd.read_csv(data_dir / params_file, index_col=0)  # assumes index column
    n_samples = len(parameters)
    chunks = int(np.ceil(n_samples/chunk_size))

    samples = parameters.iloc[:].to_records()
    chirp_masses = []
    for i in range(len(samples)):
        chirp_masses.append(chirp_mass(samples[i][1], samples[i][2]))
    # print(chirp_masses)

    # load static argument file
    _, static_args = read_ini_config(static_args_ini)

    # load template parameters file
    file = open(template_file, 'r')
    templates = file.readlines()
    for key, i in enumerate(templates):
        templates[key] = i.strip().split(',')
        for key2, j in enumerate(templates[key]):
            templates[key][key2] = float(j)
    file.close()

    # specify output directory
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)

    # check output numpy arrays
    filters_file = out_dir / 'filters.npy'
    if not overwrite:
        if filters_file.is_file():
            logging.debug(f'Aborting - {filters_file} exists but overwrite is False.')
            return

    # specify precision of output waveforms
    # dtype = np.float64 if not downcast else np.float32
    dtype = np.float64

    filters_memmap = open_memmap(
        filename=filters_file,
        mode='w+',  # create or overwrite file
        dtype=dtype,
        shape=(n_samples, n, 4)  # 4 corresponds to number of parameters (mass1, mass2, spin1z, spin2z)
    )

    # multiprocessing generation
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:

        # create buffer in memory to temporarily store before writing to desk
        filters = np.empty((chunk_size, n, 4), dtype=dtype)

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

                    filters = np.empty((end - start, n, 4), dtype=dtype)

                # get a chunk of samples
                samples_per_job = 10
                saved += end - start
                samples = chirp_masses[start:end]
                samples = [samples[i:i+samples_per_job] for i in range(0, len(samples), samples_per_job)]  # This splits it up into each multiprocessed job having multiple samples to get filters for
                # print(samples)

                # submit waveform generation jobs to multiple processes while tracking source parameters by index
                filter_sampling_job = partial(
                    sample_filter_params,
                    cm_range=cm_range,
                    best_filter=best_filter,
                    templates=templates,
                    n=n,
                )

                # store waveform polarisations in correct order while updating progress bar as futures complete
                ordered_futures = {executor.submit(filter_sampling_job, params): i for i, params in enumerate(samples)} # Looks like this line submits a new filter_sampling_job for every item in the current chunk of samples
                # print(ordered_futures)
                for future in concurrent.futures.as_completed(ordered_futures):
                    # print(np.stack(future.result()))
                    # print(np.shape(filters))
                    idx = ordered_futures[future]
                    first = idx * samples_per_job
                    last = (idx + 1) * samples_per_job
                    # print(first)
                    # print(last)
                    if last > len(filters):
                        last = len(filters)
                    # print(last)
                    filters[first:last] = np.stack(future.result())  # assign (n, 4) to array idx
                    progress.update(1 * samples_per_job)
                progress.refresh()

                filters_memmap[start:end, :, :] = filters

                # notify timer that batch has been saved
                progress.set_postfix(saved=saved)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sampling matched filtering templates.')

    parser.add_argument('-n', default=10, type=int, help="Number of templates per sample.")
    parser.add_argument('-cm', '--cm_range', default=0.05, type=float, help="Percentage range in chirp mass to sample templates from (0.005 = 0.5%).")
    parser.add_argument('-b', '--best_filter', default=False, action="store_true", help="Whether to add the best matched filter (closest chirp mass to the sample), highly recommended.")  # Best filter gets stored as the last in the array, just so it is consistent and accessible for all samples
    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')
    parser.add_argument('-t', '--template_file', dest='template_file', default='template_params.txt', action='store', type=str, help='The file path of the template list .txt file.')
    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='The input directory to load sample parameter files.')
    parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, help='The output directory to save generated waveform files.')
    parser.add_argument('-p', '--params_file', dest='params_file', default='parameters.csv', type=str, help='The input .csv file of generated parameters to load.')
    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")  # Technically 'default=False' is not needed as the action takes care of this, but I keep it for clarity

    # multiprocessing
    parser.add_argument('-c', '--chunk_size', type=int, default=500, help='The number of samples to produce filters for before appending to disk.')
    parser.add_argument('-w', '--workers', type=int, default=12, help='The number of workers to use for Python multiprocessing.')


    # random seed
    # parser.add_argument('--seed', type=int")  # to do

    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode.")

    args = parser.parse_args()

    assert args.data_dir is not None, "Output data directory must be provided with -d or --data_dir."
    assert args.static_args_ini is not None, "Static arguments .ini file must be provided with -s or --static_args."

    generate_filter_parameters(**args.__dict__)