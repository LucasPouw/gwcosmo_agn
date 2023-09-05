
Data required for running gwcosmo
=================================

This section describes the data required in order to run gwcosmo.

## Gravitational wave data
All of the analyses outlined below require some form of gravitational wave data as input. This comes in two forms: posterior samples (saved as .hdf5, .h5 or .dat file format), and skymaps (saved in .fits or .fits.gz format). These are publicly released following LIGO/Virgo/KAGRA observing runs and can be found on [zenodo](https://zenodo.org/record/5546663).

## Injections
Gravitational wave injections record the sensitivity of the detector(s) and are used to compute GW selection effects, necessary for unbiased population inference. It may be possible to use a pre-existing set of injections for your analysis, but they are not included in gwcosmo by default. If no relevant set exists, see this section on [generating your own injections using gwcosmo](#injections). 

## Galaxy catalogues
The galaxy catalogue (or "dark siren") analyses require that a pre-computed line-of-sight redshift prior (.hdf5 file format) be passed to gwcosmo. You may be able to use a pre-existing LOS redshift prior, but due to file size this data is not stored as part of the gwcosmo package. If an appropriate LOS redshift prior doesn't exist, see this section on [generating your own LOS redshift prior using gwcosmo](#los-prior). 


