# local imports
from pycbc.filter import highpass as highpass_func
from pycbc.noise import noise_from_psd

from .utils import read_ini_config, match_precision
from .noise import (
    load_psd_from_file, frequency_noise_from_psd,
    get_noise_std_from_static_args,
)

import os
import shutil
import argparse

# TO DO: Implement logging over print statements
import logging

from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
from functools import partial
from tqdm import tqdm
from datetime import datetime

import multiprocessing
import concurrent.futures

import numpy as np
from numpy.lib.format import open_memmap

import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal.windows import tukey

from lal import MSUN_SI#, LIGOTimeGPs
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

import pycbc.psd
from pycbc.detector import Detector
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.waveform import (
    get_td_waveform, get_fd_waveform,
    td_approximants, fd_approximants,
)

# prevent download as compute node may not have internet
from astropy.utils import iers
iers.conf.auto_download = False


def source_frame_to_radiation(
    mass_1: float, mass_2: float, phase: float, theta_jn: float, phi_jl: float,
    tilt_1: float, tilt_2: float, phi_12: float, a_1: float, a_2: float, f_ref: float
):
    """Simulates a precessing inspiral of a Binary Black Hole (BBH) system given
    the specified input parameters. Spins are given in the source frame co-ordinates
    and returned in the (x,y,z) radiation frame co-ordinates.
    
    """
    # convert masses from Mpc to SI units
    mass_1_SI = mass_1 * MSUN_SI
    mass_2_SI = mass_2 * MSUN_SI

    # Following bilby code
    if (
        (a_1 == 0.0 or tilt_1 in [0, np.pi])
        and (a_2 == 0.0 or tilt_2 in [0, np.pi])
    ):
        spin_1x, spin_1y, spin_1z = 0.0, 0.0, a_1 * np.cos(tilt_1)
        spin_2x, spin_2y, spin_2z, = 0.0, 0.0, a_2 * np.cos(tilt_2)
        iota = theta_jn
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
            SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                a_1, a_2, mass_1_SI, mass_2_SI, f_ref, phase
            )
        )
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z


def fade_on(
        timeseries,
        alpha=0.25
):
    """
    Take a PyCBC time-series and use a one-sided Tukey window to "fade
    on" the waveform (to reduce discontinuities in the amplitude).

    Arguments:
        timeseries: pycbc.types.timeseries.TimeSeries
            TimeSeries object to be faded on.
        alpha: float
            The alpha parameter for the Tukey window.

    Returns:
        The `timeseries` which has been faded on.
    """

    # Save the parameters from the time-series we are about to fade on
    delta_t = timeseries.delta_t
    epoch = timeseries.start_time
    duration = timeseries.duration
    sample_rate = timeseries.sample_rate

    # Create a one-sided Tukey window for the turn on
    window = tukey(M=int(duration * sample_rate), alpha=alpha)
    window[int(0.5*len(window)):] = 1

    # Apply the one-sided Tukey window for the fade-on
    ts = window * np.array(timeseries)

    # Create and return a TimeSeries object again from the resulting array
    # using the original parameters (delta_t and epoch) of the time-series
    return TimeSeries(initial_array=ts,
                      delta_t=delta_t,
                      epoch=epoch)


def extend(
    timeseries,
    static_args
):
    """
    Take a PyCBC time-series and the static arguments dictionary and
    prepend and append the time-series with zeros so that the duration
    is as required, and the merger position is correct.

    Arguments:
        timeseries: pycbc.types.timeseries.TimeSeries
            TimeSeries object to be faded on.
        static_args: Dict[str, Union[str, float]]
            A dictionary of type-casted (from str to floats or str) arguments loaded from an .ini file.
            We expect keys in this dictionary to specify the approximant, domain, frequency bands, etc.

    Returns:
        The `timeseries` which has been extended.
    """

    sample_length = int(
        (
                static_args['seconds_before_event'] + static_args['seconds_after_event']
        ) * timeseries.sample_rate)

    padding_before_merger = static_args['seconds_before_event'] - abs(min(timeseries.sample_times))

    if padding_before_merger < 0:
        raise IndexError('due to f_lower or mass combination, the inspiral is longer than the seconds_before_merger.')

    points_before_merger = int(padding_before_merger * timeseries.sample_rate)
    timeseries.prepend_zeros(points_before_merger)

    points_after_merger = sample_length - len(timeseries)

    if points_after_merger < 0:
        raise IndexError('due to seconds_after_merger being shorter than the ringdown.')

    timeseries.append_zeros(points_after_merger)

    # Now we need to extend the whole thing evenly to get total sample length

    total_sample_length = int(
        static_args['waveform_length'] * static_args['target_sampling_rate']
    )

    extend_amount = (total_sample_length - len(timeseries.data)) // 2
    timeseries.prepend_zeros(extend_amount)
    timeseries.append_zeros(extend_amount)


    return timeseries


def generate_intrinsic_waveform(
    sample: Union[np.record, Dict[str, float]],
    static_args: Dict[str, Union[str, float]],
    spins: bool=True,
    spins_aligned: bool=False,
    # inclination: bool=True,
    downcast: bool=True,
    as_pycbc: bool=False,
) -> Union[Tuple[TimeSeries], Tuple[FrequencySeries]]:
    """Function generates a waveform in either time or frequency domain using PyCBC methods.
    
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
            E.g. np.complex128 > np.complex64 for frequency, np.float64 > np.float32 for time.
        
    Returns:
        (hp, hc)
            A tuple of the plus and cross polarizations as pycbc Array types
            depending on the specified waveform domain (i.e. time or frequency).
    """
    
    # type checking inputs
    if isinstance(sample, np.record):
        assert sample.shape == (), "input array be a 1 dimensional record array. (PyCBC compatibility)"
        
    for param in ('mass_1', 'mass_2', 'phase'):
        if isinstance(sample, np.record):
            assert param in sample.dtype.names, f"{param} not provided in sample."
        if isinstance(sample, dict):
            assert param in sample.keys(), f"{param} not provided in sample."

    for arg in ('approximant', 'domain','f_lower', 'f_final', 'f_ref'):
        assert arg in static_args, f"{arg} not provided in static_args."
        if arg == 'domain':
            assert static_args['domain'] in ('time', 'frequency'), (
                f"{static_args['domain']} is not a valid domain."
            )

    # determine parameters conditional on spin and inclination of CBC
    if spins:
        if spins_aligned:
            iota = sample['theta_jn']  # we don't want to overwrite theta_jn if spins not aligned
            # spin_1x, spin_1y, spin_1z = 0., 0., sample['chi_1']
            # spin_2x, spin_2y, spin_2z = 0., 0., sample['chi_2']
            spin_1x, spin_1y, spin_1z = 0., 0., sample['a_1']
            spin_2x, spin_2y, spin_2z = 0., 0., sample['a_2']
        else:
            # convert from event frame to radiation frame
            iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = source_frame_to_radiation(
                sample['mass_1'], sample['mass_2'], sample['phase'], sample['theta_jn'], sample['phi_jl'],
                sample['tilt_1'], sample['tilt_2'], sample['phi_12'], sample['a_1'], sample['a_2'], static_args['f_ref']
            )
    else:
        iota = sample['theta_jn']
        spin_1x, spin_1y, spin_1z = 0., 0., 0.
        spin_2x, spin_2y, spin_2z = 0., 0., 0.
        
    if static_args['domain'] in ('time',):
        # generate time domain waveform
        assert 'delta_t' in static_args, "delta_t not provided in static_args."
        assert static_args['approximant'] in td_approximants(), (
            f"{static_args['approximant']} is not a valid time domain waveform"
        )

        # TO DO: Handle variable length td_waveform generation.
        # ValueError: Time series does not contain a time as early as -8.
        # raise NotImplementedError('time domain waveforms not yet implemented.')

        # Make sure f_min is low enough
        # if static_args['waveform_length'] > get_waveform_filter_length_in_time(
        #     mass1=sample['mass_1'], mass2=sample['mass_2'],
        #     spin1x=spin_1x, spin2x=spin_2x,
        #     spin1y=spin_1y, spin2y=spin_2y,
        #     spin1z=spin_1z, spin2z=spin_2z,
        #     inclination=iota,
        #     f_lower=static_args['f_lower'],
        #     f_ref=static_args['f_ref'],
        #     approximant=static_args['approximant'],
        #     distance=distance,
        # ):
        #     print('Warning: f_min not low enough for given waveform duration')

        # get plus polarisation (hp) and cross polarisation (hc) as time domain waveforms
        hp, hc = get_td_waveform(
            mass1=sample['mass_1'], mass2=sample['mass_2'],
            spin1x=spin_1x, spin2x=spin_2x,
            spin1y=spin_1y, spin2y=spin_2y,
            spin1z=spin_1z, spin2z=spin_2z,
            coa_phase=sample['phase'],
            inclination=iota,  #  "Check this!" - Stephen Green
            delta_t=static_args['delta_t'],
            f_lower=static_args['f_lower'],
            f_ref=static_args['f_ref'],
            approximant=static_args['approximant'],
            distance=sample['distance'],
        )
        
        # Apply the fade-on filter to them - should be timeseries only?
        hp = fade_on(hp, alpha=static_args['tukey_alpha'])
        hc = fade_on(hc, alpha=static_args['tukey_alpha'])

        # Extend the waveform to the desired duration before and after merger
        hp = extend(hp, static_args=static_args)
        hc = extend(hc, static_args=static_args)

        if downcast:
            hp = hp.astype(np.float32)
            hc = hc.astype(np.float32)

    elif static_args['domain'] in ('frequency',):
        # generate frequency domain waveform
        assert 'delta_t' in static_args, "delta_t not provided in static_args."
        assert static_args['approximant'] in fd_approximants(), (
            f"{static_args['approximant']} is not a valid frequency domain waveform"
        )
        
        hp, hc = get_fd_waveform(
            mass1=sample['mass_1'], mass2=sample['mass_2'],
            spin1x=spin_1x, spin2x=spin_2x,
            spin1y=spin_1y, spin2y=spin_2y,
            spin1z=spin_1z, spin2z=spin_2z,
            coa_phase=sample['phase'],
            inclination=iota,  #  "Check this!" - Stephen Green
            delta_f=static_args['delta_f'],
            f_lower=static_args['f_lower'],
            f_final=static_args['f_final'],
            f_ref=static_args['f_ref'],
            approximant=static_args['approximant'],
            distance=sample['distance'],
        )

        # time shift - should we do this when we create or project the waveform?
        hp = hp.cyclic_time_shift(static_args['seconds_before_event'])
        hc = hc.cyclic_time_shift(static_args['seconds_before_event'])
        hp.start_time += static_args['seconds_before_event']
        hc.start_time += static_args['seconds_before_event']
        
        if downcast:
            hp = hp.astype(np.complex64)
            hc = hc.astype(np.complex64)
            
    if as_pycbc:
        return hp, hc

    return hp.data, hc.data


# https://pycbc.org/pycbc/latest/html/_modules/pycbc/detector.html
def project_onto_detector(
    detector: Detector,
    sample: Union[np.recarray, Dict[str, float]],
    hp: Union[np.ndarray, FrequencySeries],
    hc: Union[np.ndarray, FrequencySeries],
    static_args: Dict[str, Union[str, float]],
    sample_frequencies: Optional[np.ndarray]=None,
    as_pycbc: bool=True,
    time_shift: bool=False,
) -> Union[np.ndarray, FrequencySeries]:
    """Takes a plus and cross waveform polarization (i.e. generated by intrinsic parameters)
    and projects them onto a specified interferometer using a PyCBC.detector.Detector.
    """
    # input handling
    assert type(hp) == type(hc), "Plus and cross waveform types must match."
    if isinstance(hp, FrequencySeries):
        assert np.all(hp.sample_frequencies == hc.sample_frequencies), "FrequencySeries.sample_frequencies do not match."
        sample_frequencies = hp.sample_frequencies
    assert sample_frequencies is not None, "Waveforms not FrequencySeries type or frequency series array not provided."
    
    # project intrinsic waveform onto detector
    fp, fc = detector.antenna_pattern(sample['ra'], sample['dec'], sample['psi'], static_args.get('ref_time', 0.))
    h = fp*hp + fc*hc

    # scale waveform amplitude according to ratio to reference distance
    h *= static_args.get('distance', 1)  / sample['distance']  # default d_L = 1

    if time_shift:        
        # Calculate time shift at detector and add to geocentric time
        dt = detector.time_delay_from_earth_center(sample['ra'], sample['dec'], static_args.get('ref_time', 0.))
        dt += sample['time'] - static_args.get('ref_time', 0.)  # default ref t_c = 0
        dt = dt.astype(match_precision(sample_frequencies))
        h *= np.exp(- 2j * np.pi * dt * sample_frequencies).astype(match_precision(h, real=False))
        
    # output desired type
    if isinstance(h, FrequencySeries):
        if as_pycbc:
            return h
        else:
            return h.data
    else:
        if as_pycbc:
            return FrequencySeries(h, delta_f=static_args['delta_f'], copy=False)
        else:
            return h


def project_td(
        data: np.ndarray,
        idx: int,
        ifo: str,
        params: pd.DataFrame,
        static_args: Dict[str, Union[str, float]],
):
    # get the two polarisations as PyCBC TimeSeries objects
    hp = TimeSeries(data[0], delta_t=1.0/static_args['target_sampling_rate'])
    hc = TimeSeries(data[1], delta_t=1.0/static_args['target_sampling_rate'])

    # print(len(hp.data))

    # extract source location parameters
    ra = params.iloc[idx]['ra']
    dec = params.iloc[idx]['dec']
    pol = params.iloc[idx]['psi']

    # load detector object
    det = Detector(ifo)

    # project onto the detector
    strain = det.project_wave(hp=hp, hc=hc, ra=ra, dec=dec, polarization=pol, method='vary_polarization')

    # print(len(strain.data))

    return strain.data


def get_sample_frequencies(f_lower: float=0, f_final: float=1024, delta_f: float=0.125) -> np.ndarray:
    """Utility function to construct sample frequency bins for frequency domain data."""
    return np.linspace(f_lower, f_final, int((f_final - f_lower) / delta_f) + 1)


def batch_project(
    detector: Detector,
    extrinsics: Union[np.recarray, Dict[str, np.ndarray]],
    waveforms: np.ndarray,
    static_args: Dict[str, Union[str,float]],
    sample_frequencies: Optional[np.ndarray]=None,
    distance_scale: bool=False,
    time_shift: bool=False,
):
    # get frequency bins (we handle numpy arrays rather than PyCBC arrays with .sample_frequencies attributes)
    if sample_frequencies is None:
        sample_frequencies = get_sample_frequencies(
            f_final=static_args['f_final'],
            delta_f=static_args['delta_f']
        )
    
    # check waveform matrix inputs
    assert waveforms.shape[1] == 2, "2nd dim in waveforms must be plus and cross polarizations."
    assert waveforms.shape[0] == len(extrinsics), "waveform batch and extrinsics length must match."

    # if distance_scale:
    #     # scale waveform amplitude according to sample d_L parameter
    #     scale = static_args.get('distance', 1000) / extrinsics['distance']  # default d_L = 1
    #     waveforms = np.array(waveforms) * scale[:, None, None]

    # calculate antenna pattern given arrays of extrinsic parameters
    fp, fc = detector.antenna_pattern(
        extrinsics['ra'], extrinsics['dec'], extrinsics['psi'],
        static_args.get('ref_time', 0.)
    )

    # batch project
    projections = fp[:, None]*waveforms[:, 0, :] + fc[:, None]*waveforms[:, 1, :]

    # if distance_scale:
    #     # scale waveform amplitude according to sample d_L parameter
    #     scale = static_args.get('distance', 1000)  / extrinsics['distance']  # default d_L = 1
    #     projections *= scale[:, None]
    
    if time_shift:    
        assert waveforms.shape[-1] == sample_frequencies.shape[0], "delta_f and fd_length do not match."
        
        # calculate geocentric time for the given detector
        dt = detector.time_delay_from_earth_center(extrinsics['ra'], extrinsics['dec'], static_args.get('ref_time', 0.))
    
        # Calculate time shift due to sampled t_c parameters
        dt += extrinsics['time'] - static_args.get('ref_time', 0.)  # default ref t_c = 0
        
        dt = dt.astype(match_precision(sample_frequencies, real=True))
        shift = np.exp(- 2j * np.pi * dt[:, None] * sample_frequencies[None, :])
        projections *= shift

    return projections


def validate_waveform(
    waveform: np.ndarray,
    sample_length: int,
    static_args: Dict[str, float],
    out_dir: Union[str, os.PathLike],
    name: Optional[str]=None,
):
    # create out dir if it does not exist
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Generate waveform
    assert waveform.shape[0] == 2, "First dimension of waveform sample must be 2 (+ and x polarizations)."
    assert waveform.shape[1] == sample_length, "Waveform length not expected given provided static_args."

    if static_args['domain'] in ('frequency',):
        hp, hc = waveform
        plus = FrequencySeries(hp, delta_f=static_args['delta_f'], copy=True)
        cross = FrequencySeries(hc, delta_f=static_args['delta_f'], copy=True)

        # ifft to time domain
        sp = plus.to_timeseries(delta_t=static_args['delta_t'])
        sc = cross.to_timeseries(delta_t=static_args['delta_t'])

        # plot plus / cross polarizations
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
        ax.plot(sp.sample_times, sp, label='Plus', color='tab:pink')  # pink for plus
        ax.plot(sc.sample_times, sc, label='Cross', color='tab:cyan')  # cyan for cross
        ax.set_ylabel('Strain')
        ax.set_xlabel('Time (s)')
        ax.grid('both')

        aux_desc='' if name is None else f' ({name})'
        ax.set_title(f"IFFT of Simulated {static_args['approximant']} Waveform Polarizations{aux_desc}", fontsize=14)
        ax.legend(loc='upper left')

        fig.tight_layout()
        fig.savefig(out_dir / 'intrinsic_polarizations.png')

    elif static_args['domain'] in ('time',):
        hp, hc = waveform
        plus = TimeSeries(hp, delta_t=1.0/static_args['target_sampling_rate'], copy=True)
        cross = TimeSeries(hc, delta_t=1.0/static_args['target_sampling_rate'], copy=True)

        # plot plus / cross polarizations
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
        ax.plot(plus.sample_times, plus, label='Plus', color='tab:pink')  # pink for plus
        ax.plot(cross.sample_times, cross, label='Cross', color='tab:cyan')  # cyan for cross
        ax.set_ylabel('Strain')
        ax.set_xlabel('Time (s)')
        ax.grid('both')

        aux_desc='' if name is None else f' ({name})'
        ax.set_title(f"Plot of Simulated {static_args['approximant']} Waveform Polarizations{aux_desc}", fontsize=14)
        ax.legend(loc='upper left')

        fig.tight_layout()
        fig.savefig(out_dir / 'intrinsic_waveforms.png')


def validate_projections(
    projections: np.ndarray,
    sample_length: int,
    static_args: Dict[str, float],
    out_dir: Optional[Union[str, os.PathLike]]=None,
    ifos: Optional[List[str]]=None,
    name: Optional[str]=None,
    save: bool=True,
    show: bool=False,
):
    # full names for plotting
    interferometers = {'H1': 'Hanford', 'L1': 'Livingston', 'V1': 'Virgo', 'K1': 'KAGRA'}

    # Generate waveform
    if ifos is not None:
        assert projections.shape[0] == len(ifos), "First dimension of waveform sample must be 2 (+ and x polarizations)."
    assert projections.shape[1] == sample_length, "Waveform length not expected given provided static_args."

    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4*nrows))

    if static_args['domain'] in ('frequency',):
        # ifft to time domain
        for i in range(projections.shape[0]):
            label = None if ifos is None else interferometers[ifos[i]]
            h = FrequencySeries(projections[i, :], delta_f=static_args['delta_f'], copy=True)
            strain = h.to_timeseries(delta_t=static_args['delta_t'])
            ax.plot(strain.sample_times, strain, label=label, alpha=0.6)

        ax.set_ylabel('Strain')
        ax.set_xlabel('Time (s)')
        ax.grid('both')
        ax.legend(loc='upper left')

        aux_desc='' if name is None else f' ({name})'
        fig.suptitle(f"IFFT of Simulated {static_args['approximant']} Projected Waveforms at Interferometers{aux_desc}", fontsize=16)
        fig.tight_layout()

        if save:
            # create out dir if it does not exist
            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            fig.savefig(out_dir / 'projection_examples.png')

        if show: fig.show()

    elif static_args['domain'] in ('time',):
        for i in range(projections.shape[0]):
            label = None if ifos is None else interferometers[ifos[i]]
            strain = TimeSeries(projections[i, :], delta_t=static_args['target_sampling_rate'], copy=True)
            ax.plot(strain.sample_times, strain, label=label, alpha=0.6)

        ax.set_ylabel('Strain')
        ax.set_xlabel('Time (s)')
        ax.grid('both')
        ax.legend(loc='upper left')

        aux_desc='' if name is None else f' ({name})'
        fig.suptitle(f"Plot of Simulated {static_args['approximant']} Projected Waveforms at Interferometers{aux_desc}", fontsize=16)
        fig.tight_layout()

        if save:
            # create out dir if it does not exist
            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            fig.savefig(out_dir / 'projection_examples.png')

        if show: fig.show()


def generate_waveform_dataset(
    static_args_ini: str,
    data_dir: str,
    out_dir: Optional[str]=None,
    params_file: str='parameters.csv',
    ifos: Optional[List[str]]=None,
    add_noise: bool=False,
    gaussian: bool=True,
    psd_dir: Optional[str]=None,
    whiten: bool=True,
    ref_ifo: Optional[str]=None,
    highpass: bool=True,
    projections_only: bool=False,
    overwrite: bool=False,
    metadata: bool=True,
    downcast: bool=False,
    verbose: bool=False,
    validate: bool=False,
    distance_scale: bool=False,
    time_shift: bool=False,
    workers: int=1,
    chunk_size: int=2500,
):
    """Function to generate a dataset of intrinsic waveform parameters using Python multiprocessing.
    
    Arguments:
        static_args: Dict[str, float]
            A dictionary of static args....? But we load this
        data_dir: str
            The dataset directory to load configuration files and save waveforms.
        params_file: str
            The .csv file to load params from - TO DO : should this be "parameters" or "parameters.csv"?  
        file_name: str
            The base name of the intrinsic waveform numpy file.

    """
    # load parameters
    data_dir = Path(data_dir)
    parameters = pd.read_csv(data_dir / params_file, index_col=0)  # assumes index column
    n_samples = len(parameters)
    chunks = int(np.ceil(n_samples/chunk_size))

    # TO DO: if parameters are not provided - do we sample / call generate_parameters automatically?

    # load static argument file
    _, static_args = read_ini_config(static_args_ini)

    # specify output directory
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    assert not out_dir.is_file(), f"{out_dir} is a file. It should either not exist or be a directory."
    out_dir.mkdir(parents=True, exist_ok=True)

    # check output numpy arrays
    waveform_file = out_dir / 'waveforms.npy'
    projections_file = out_dir / 'projections.npy'
    if not overwrite:
        if waveform_file.is_file():
            logging.debug(f'Aborting - {waveform_file} exists but overwrite is False.')
            return
        if projections_file.is_file():
            logging.debug(f'Aborting - {projections_file} exists but overwrite is False.')
            return

    if metadata:
        # load static_args.ini and save as .json in dataset
        in_args_path = Path(static_args_ini)
        config_dir = out_dir / 'config_files'
        config_dir.mkdir(exist_ok=True)
        out_args_path = config_dir / in_args_path.name
        shutil.copy(in_args_path, out_args_path)

    # Due to issues whitening the waveforms on their own, we cannot do the gaussian option for timeseries
    if gaussian:
        if static_args['domain'] in ('time',):
            print("Cannot use gaussian option for timeseries due to pure waveform whitening issue.")
            return
    if whiten and not add_noise:
        if static_args['domain'] in ('time',):
            print("Will not whiten the pure waveform in the either waveforms or projections file due to pure waveform whitening issue.")
            return

    # Calculate sample length for time-series waveforms
    if static_args['domain'] in ('time',):
        total_sample_length = int(
            static_args['waveform_length'] * static_args['target_sampling_rate']
        )
        final_sample_length = int(
            (
                    static_args['seconds_before_event'] + static_args['seconds_after_event']
            ) * static_args['target_sampling_rate'])

        first = (total_sample_length // 2) - (final_sample_length // 2)
        last = first + final_sample_length

        # specify precision of output waveforms
        dtype = np.float64 if not downcast else np.float32
    elif static_args['domain'] in ('frequency',):
        total_sample_length = static_args['fd_length']
        final_sample_length = static_args['fd_length']

        # specify precision of output waveforms
        dtype = np.complex128 if not downcast else np.complex64
    else:
        raise ValueError('invalid domain in static arguments file, please choose one of `frequency` and `time`.')



    # whether to project waveforms onto detectors or keep as intrinsic
    psds = {}
    if ifos is not None:
        required_extrinsics = ('ra', 'dec', 'psi')
        for extrinsic in required_extrinsics:
            assert extrinsic in parameters.columns, f"{extrinsic} not in {required_extrinsics}."

        detectors = {ifo: Detector(ifo) for ifo in ifos}
        waveform_desc = f'({ifos})'
        sample_frequencies = get_sample_frequencies(
            f_final=static_args['f_final'],
            delta_f=static_args['delta_f']
        )

        # load PSD files for noise generation or whitening
        if psd_dir is not None:
            for ifo in ifos:
                # coloured noise from psd
                psd_file = Path(psd_dir) / f'{ifo}_PSD.npy'
                assert psd_file.is_file(), f"{psd_file} does not exist."
                psds[ifo] = load_psd_from_file(psd_file)

        # https://numpy.org/devdocs/reference/generated/numpy.lib.format.open_memmap.html#numpy.lib.format.open_memmap
        projection_memmap = open_memmap(
            filename=projections_file,
            mode='w+',  # create or overwrite file
            dtype=dtype,
            shape=(n_samples, len(ifos), final_sample_length)
        )

        if add_noise:
            # noise standardization factor
            noise_std = get_noise_std_from_static_args(static_args)
            noise_file = out_dir / 'noise.npy'
            noise_memmap = open_memmap(
                filename=noise_file,
                mode='w+',  # create or overwrite file
                dtype=dtype,
                shape=(n_samples, len(ifos), final_sample_length)
            )

    else:
        # plus and cross polarizations
        detectors = None
        waveform_desc = '(intrinsic)'
        sample_frequencies = None  # only used for projected time shifts
        projections_only = False

    # intrinsic waveforms
    if not projections_only:
        waveform_memmap = open_memmap(
            filename=waveform_file,
            mode='w+',  # create or overwrite file
            dtype=dtype,
            shape=(n_samples, 2, final_sample_length)
        )

    # reference PSD at specified ifo (optional)
    if ref_ifo is not None:
        if ref_ifo not in psds:
            if psd_dir is None:
                psd_dir = data_dir / 'PSD'  # assumption
                psd_dir.is_dir(), (
                    f"Assumed psd_dir {psd_dir} does not exist. \
                    Consider passing in the psd_dir argument or checking PSD generation."
                )
            psd_file = Path(psd_dir) / f'{ref_ifo}_PSD.npy'
            assert psd_file.is_file(), f"{psd_file} does not exist."
            psds[ref_ifo] = load_psd_from_file(psd_file)


    # multiprocessing generation
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # create buffer in memory to temporarily store before writing to desk
        waveforms = np.empty((chunk_size, 2, total_sample_length), dtype=dtype)

        # loop through samples at approx. 10GB at a time and append to numpy array
        progress_desc = f"[{datetime.now().strftime('%H:%M:%S')}] Generating {static_args['approximant']} {waveform_desc} waveforms"
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

                    waveforms = np.empty((end - start, 2, total_sample_length), dtype=dtype)
                    if static_args['domain'] in ('time',):
                        final_waveforms = np.empty((end - start, 2, final_sample_length), dtype=dtype)
                
                # get a chunk of samples
                saved += end - start
                samples = parameters.iloc[start:end].to_records()

                # submit waveform generation jobs to multiple processes while tracking source parameters by index
                waveform_generation_job = partial(
                    generate_intrinsic_waveform,
                    spins=True,  # hard-coded
                    spins_aligned=True,  # hard-coded
                    static_args=static_args,
                    downcast=downcast,
                )

                # store waveform polarisations in correct order while updating progress bar as futures complete
                ordered_futures = {executor.submit(waveform_generation_job, params): i for i, params in enumerate(samples)}
                for future in concurrent.futures.as_completed(ordered_futures):
                    waveforms[ordered_futures[future]] = np.stack(future.result())  # assign (2, sample_length) to array idx
                    progress.update(1)
                progress.refresh()

                if detectors is not None:
                    # output should be (batch, ifo, length)
                    projections = np.empty((end - start, len(ifos), total_sample_length), dtype=dtype)
                    if static_args['domain'] in ('time',):
                        final_projections = np.empty((end - start, len(ifos), final_sample_length), dtype=dtype)
                    if add_noise:
                        noise = np.empty((end - start, len(ifos), total_sample_length), dtype=dtype)
                        if static_args['domain'] in ('time',):
                            final_noise = np.empty((end - start, len(ifos), final_sample_length), dtype=dtype)

                    # validate_waveform(waveforms[-1], sample_length, static_args, out_dir / 'figures1', name=f'sample_idx={n_samples}')
                    # validate_projections(projections[-1], sample_length, static_args, out_dir / 'figures1', ifos=ifos, name=f'sample_idx={n_samples}')
                    for i, ifo in enumerate(ifos):
                        # batch project for each detector
                        if static_args['domain'] in ('frequency',):
                            projections[:, i, :] = batch_project(
                                detectors[ifo],
                                samples,
                                waveforms,
                                static_args,
                                sample_frequencies,
                                distance_scale=distance_scale,
                                time_shift=time_shift,
                            )
                        elif static_args['domain'] in ('time',):
                            for key, elem in enumerate(waveforms):
                                projections[key, i, :] = project_td(
                                    data=elem, idx=key, ifo=ifo, params=parameters, static_args=static_args
                                )

                        if add_noise:
                            # to do: check if the following approaches to generating noise are equivalent
                            if gaussian or psd_dir is None:
                                if static_args['domain'] in ('frequency',):
                                    # gaussian white noise in frequency domain
                                    size = (end-start, static_args['fd_length'])  # gaussian for each batch for each freq bin
                                    noise[:, i, :] = (np.random.normal(0., noise_std, size) + 1j*np.random.normal(0., noise_std, size)).astype(dtype)
                                elif static_args['domain'] in ('time',):
                                    # Since there will be no whitening for white gaussian noise, we skip generating the longer samples
                                    size = (end-start, final_sample_length)
                                    final_noise[:, i, :] = np.random.normal(0., noise_std, size).astype(dtype)
                            else:
                                if static_args['domain'] in ('frequency',):
                                    # coloured noise from psd -- cut to fd_length (bandpass filter for higher frequencies)
                                    noise[:, i, :] = frequency_noise_from_psd(psds[ifo], n=end-start)[:, :static_args['fd_length']]
                                elif static_args['domain'] in ('time',):
                                    for j, sample in enumerate(noise[:, i, :]):
                                        noise[j, i, :] = noise_from_psd(psd=psds[ifo], length=total_sample_length, delta_t=static_args['delta_t']).data

                                if whiten:
                                    if static_args['domain'] in ('frequency',):
                                        noise[:, i, :] /= psds[ifo][:static_args['fd_length']] ** 0.5
                                    elif static_args['domain'] in ('time',):
                                        for j, sample in enumerate(noise[:, i, :]):
                                            final_noise[j, i, :] = noise[j, i, first:last]
                                            # timeseries = TimeSeries(sample, delta_t=static_args['delta_t'], copy=True)
                                            # final_noise[j, i, :] = timeseries.whiten(
                                            #     static_args['whitening_segment_duration'],
                                            #     static_args['whitening_max_filter_duration'],
                                            #     remove_corrupted=False
                                            # ).data[first:last]

                        if whiten:
                            if static_args['domain'] in ('frequency',):
                                projections[:, i, :] /= psds[ifo][:static_args['fd_length']] ** 0.5

                            # I have decided to only whiten the combination of waveform and noise (projection),
                            # and they will be individually saved without whitening

                        if highpass:
                            # filter out values less than f_lower (e.g. 20Hz) - to do: check truncation vs. zeroing
                            if static_args['domain'] in ('frequency',):
                                projections[:, i, :int(static_args['f_lower'] / static_args['delta_f'])] = 0.0

                            if add_noise:
                                if static_args['domain'] in ('frequency',):
                                    noise[:, i, :int(static_args['f_lower'] / static_args['delta_f'])] = 0.0

                            # I have decided to only highpass the combination of waveform and noise (projection),
                            # and they will be individually saved without highpassing

                    if add_noise:

                        if static_args['domain'] in ('frequency',):
                            projections += noise

                        if not gaussian:
                            projections += noise

                            if whiten:
                                if static_args['domain'] in ('time',):
                                    for i, ifo in enumerate(ifos):
                                        for j, sample in enumerate(projections[:, i, :]):
                                            timeseries = TimeSeries(sample, delta_t=static_args['delta_t'], copy=True)
                                            projections[j, i, :] = timeseries.whiten(
                                                static_args['whitening_segment_duration'],
                                                static_args['whitening_max_filter_duration'],
                                                remove_corrupted=False
                                            ).data

                            if highpass:
                                if static_args['domain'] in ('time',):
                                    for i, ifo in enumerate(ifos):
                                        for j, sample in enumerate(projections[:, i, :]):
                                            timeseries = TimeSeries(sample, delta_t=static_args['delta_t'], copy=True)
                                            projections[j, i, :] = highpass_func(timeseries, static_args['f_lower']).data

                        if static_args['domain'] in ('frequency',):
                            noise_memmap[start:end, :, :] = noise
                        elif static_args['domain'] in ('time',):
                            if gaussian:
                                noise_memmap[start:end, :, :] = final_noise
                            else:
                                if whiten:
                                    noise_memmap[start:end, :, :] = final_noise
                                else:
                                    noise_memmap[start:end, :, :] = noise[:, :, first:last]

                    if static_args['domain'] in ('frequency',):
                        projection_memmap[start:end, :, :] = projections
                    elif static_args['domain'] in ('time',):
                        if gaussian:
                            projection_memmap[start:end, :, :] = final_projections
                        else:
                            projection_memmap[start:end, :, :] = projections[:, :, first:last]

                if not projections_only:
                    if whiten and (ref_ifo is not None):
                        if static_args['domain'] in ('frequency',):
                            waveforms /= psds[ref_ifo][:static_args['fd_length']] ** 0.5

                    if static_args['domain'] in ('frequency',):
                        waveform_memmap[start:end, :, :] = waveforms
                    elif static_args['domain'] in ('time',):
                        waveform_memmap[start:end, :, :] = waveforms[:, :, first:last]

                # notify timer that batch has been saved
                progress.set_postfix(saved=saved)

    if validate:
        # this should get the final sample in the full dataset

        if static_args['domain'] in ('frequency',):
            validate_waveform(waveforms[-1], final_sample_length, static_args, out_dir / 'figures', name=f'sample_idx={n_samples}')
            if detectors is not None:
                validate_projections(projections[-1], final_sample_length, static_args, out_dir / 'figures', ifos=ifos, name=f'sample_idx={n_samples}')
        elif static_args['domain'] in ('time',):
            validate_waveform(waveforms[-1, :, first:last], final_sample_length, static_args, out_dir / 'figures', name=f'sample_idx={n_samples}')
            if detectors is not None:
                validate_projections(projections[-1, :, first:last], final_sample_length, static_args, out_dir / 'figures', ifos=ifos, name=f'sample_idx={n_samples}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for waveform generation code.')

    # configuration
    parser.add_argument('-i', '--ifos', type=str, nargs='+', help='The interferometers to project data onto - assumes extrinsic parameters are present.')
    parser.add_argument('--ref_ifo', type=str,  help='A reference interferometer if we wish to whiten waveforms only one kind of PSD. Useful for pre-whitening and memory efficiency.')
    parser.add_argument('-s', '--static_args', dest='static_args_ini', action='store', type=str, help='The file path of the static arguments configuration .ini file.')
    parser.add_argument('-p', '--params_file', dest='params_file', default='parameters.csv', type=str, help='The input .csv file of generated parameters to load.')
    
    # data directories
    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='The input directory to load parameter files.')
    parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, help='The output directory to save generated waveform files.')
    parser.add_argument('--psd_dir', dest='psd_dir', type=str, help='The output directory to save generated waveform files.')

    parser.add_argument('--overwrite', default=False, action="store_true", help="Whether to overwrite files if data_dir already exists.")
    parser.add_argument('--metadata', default=False, action="store_true", help="Whether to copy config file metadata to data_dir with parameters.")
    parser.add_argument('--projections_only', default=False, action="store_true", help="Whether to only save projections.npy files and ignore intrinsic waveforms.npy.")

    # signal processing
    parser.add_argument('--add_noise', default=False, action="store_true", help="Whether to add frequency noise - if PSDs are provided we add coloured noise, else Gaussian.")
    parser.add_argument('--gaussian', default=False, action="store_true", help="Whether to generate white gaussian nois when add_noise is True. If False, coloured noise is generated from a PSD.")
    parser.add_argument('--whiten', default=False, action="store_true", help="Whether to whiten the data with the provided PSD before fitting a reduced basis.")
    parser.add_argument('--highpass', default=False, action="store_true", help="Whether to truncate the frequency domain data below 'f_lower' specified in the static args.")
    parser.add_argument('--distance_scale', default=False, action="store_true", help="Whether to scale the waveform by applying the sampled distance.")
    parser.add_argument('--time_shift', default=False, action="store_true", help="Whether to shift the waveform by applying the sampled time shift.")

    # logging
    # parser.add_argument("-l", "--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="Sets verbose mode to display progress bars.")
    parser.add_argument('--validate', default=False, action="store_true", help='Whether to validate a sample of the data to check for correctness.')

    # multiprocessing
    parser.add_argument('-c', '--chunk_size', type=int, default=2500, help='The number of samples to work on before appending to disk, we recommend a lower number for timeseries and for BNS samples.')
    parser.add_argument('-w', '--workers', type=int, default=8, help='The number of workers to use for Python multiprocessing.')
    
    # # generation
    # parser.add_argument('--seed', type=int, help="Random seed.")  # to do

    args = parser.parse_args()
    # logging.basicConfig(
    #     format='%(process)d-%(levelname)s-%(message)s',
    #     level=getattr(logging, args.logging)
    # )
    
    # if args.workers == -1: args.workers = multiprocessing.cpu_count()
    assert 1 <= args.workers <= multiprocessing.cpu_count(), f"{args.workers} workers are not available."

    # kwargs = {key: val for key, val in args.__dict__.items() if key not in 'logging'}
    generate_waveform_dataset(**args.__dict__)