# This Fork

The intention of this forked project is to implement the use of real data as the 
background and also provide an efficient implementation for generating SNR time-
series as the final output.

### Main tasks:
* Implement time domain outputs instead of frequency domain (due to real noise)
* Edit the batch_project function because it is currently only built for frequency domain. Be sure that it sowkrs for time domain as well or change
* Implement use of real noise from LIGO datasets on OzStar
* Add SNR time-series generation
    * Using SPIIR bank parameters and the OMF and sampling close chirp masses
* Implement use of SPIIR/LIGO injection sets (for astrophysical testing)
* Implement 3 (or arbitrary) detectors

## ozstar module loads + virtualenv
`module load python/3.8.5`

We could replace later `pip install`s with `module load`, e.g.

`module load wheel/0.37.0-python-3.8.5`

We do load wheel as above, but the rest is downloaded via pip.

## Virtualenv install commands

    python -m venv <name>
    . <name>/bin/activate

## Conda install commands

    conda create --name <name> python=3.8.5 -y
    conda activate <name>  # source activate <name> works for containers

## Shared install commands

Upgrade pip inside conda (or just in regular virtual environment)

`python -m pip install --upgrade pip`

Install main packages

`pip install tqdm numpy scipy pandas scikit-learn matplotlib seaborn pycbc lalsuite ligo.skymap`  # core
`pip install wheel black pytest mypy ipykernel`

OR

`pip install -r requirements.txt`

## Miscellaneous notes

Sometimes `astropy.utils.iers` will call something to auto-download data.
As compute nodes have no internet access, we have disabled this in `gwpe/waveforms.py` with:

    from astropy.utils import iers
    iers.conf.auto_download = False