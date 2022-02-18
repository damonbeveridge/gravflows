# This Fork

The intention of this forked project is to implement the use of real data as the background and also provide an efficient implementation for generating SNR time-series as the final output.

### Two main tasks:
* Implement time domain outputs instead of frequency domain
* Implement use of real noise
* Add SNR time-series generation
  * Using SPIIR bank parameters and using the OMF and sampling close chirp masses
* Implement use of SPIIR/LIGO injection sets (for astrophysical testing)

# Master of Data Science Research Project

This repository is for the research work of Daniel Tang (21140852) for work on data engineering and training of flow-based models for gravitational wave parameter estimation.

We have forked code written by Stephen R. Green and Jonathan Gair [arXiv:2008.03312](https://arxiv.org/abs/2008.03312) to leverage their previous work on training a neural spline flow model for gravitational wave parameter estimation.

We have also forked code from https://github.com/damonbeveridge/samplegen - a repository that adds additional features to the sample generation code by Timothy Gebhard and Niki Kilbertus [arXiv:1904.08693](https://arxiv.org/abs/1904.08693) for generating realistic synthetic gravitational-wave data. Our main use of this code is to leverage pre-existing code loading bit-masks and data from .hdf5 files to obtain valid gravitational wave strains as well as PyCBC config file handling.
## Current Goals

We have been investigating how to refactor the waveform generation code for potential run-time speedups using multi-threading (e.g. for I/O bound fetching open source data - concurrent multi-tasking was eventually tried rather multi-threading in Python) and multi-processing (e.g. for CPU bound simluating of waveform models given samples from our priors).

Additionally, we intend to rewrite a reproducible distributed training framework with PyTorch and DataDistributedParallel (e.g. GPU bound memory constraints during training) help accelerate research and development time for training deep learning models.

