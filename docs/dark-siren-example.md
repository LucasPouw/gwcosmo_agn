
# Dark siren example

We provide here the complete command line for the dark siren analysis of the 42 BBHs detected and selected during O1-O2-O3. The information used for this analysis is the mass distribution of the set of black holes together with their 3D location on the sky (right ascension, declination and luminosity distance).
We will assume you have a login at the CIT cluster of the form `albert.einstein`. All paths are relative to this cluster. The analysis runs the code `gwcosmo_dark_siren_posterior` and it assumes that `gwcosmo` was installed inside a `conda` environment in `/home/albert.einstein/.conda/envs/gwcosmo/`.

The script `job.sub` below must be given to `condor` with the command `condor_submit job.sub`. In the script we define the variable `environment`: careful, its definition can vary according to the cluster on which you are running the analysis. The version here runs well on CIT as of 20231010.

```
# contents of the file job.sub:

environment = "HOME='/home/albert.einstein/' OMP_NUM_THREADS=1 PATH='/home/albert.einstein/.conda/envs/gwcosmo/bin'"

executable = /home/albert.einstein/gwcosmo.test/bin/gwcosmo_dark_siren_posterior
arguments = --method sampling
--posterior_samples /home/albert.einstein/posterior_samples.json
--skymap /home/albert.einstein/skymaps_dict.json
--LOS_catalog /home/albert.einstein/GLADE+_LOS_redshift_prior_K_band_luminosity_weighted_nside_32_pixel_index_None.hdf5
--injections /home/albert.einstein/inj_SNR9_det_frame_2e6.h5
--parameter_dict /home/albert.einstein/parameter_dict_BBH_powerlaw_peak.json
--redshift_evolution Madau
--mass_model BBH-powerlaw-gaussian
--snr 10
--sampler dynesty
--nlive 1000
--dlogz 0.1
--npool 24
--outputfile MyAnalysis
--min_pixels 30 

output = /home/albert.einstein/out.txt
error = /home/albert.einstein/err.txt
log = /home/albert.einstein/log.txt

accounting_group = ligo.dev.o4.cbc.hubble.gwcosmo
request_cpus   = 24
request_memory = 16 gb
request_disk   = 8 gb

queue
```

Several input files are needed for the analysis.

## The posteriors of the GW events


The file `/home/albert.einstein/posterior_samples.json` contains the paths to the posteriors of all events you want to include in your analysis. A posterior file contains the arrays of the reconstructed parameters of the binary system ($m_1, m_2, \text{luminosity_distance}...$). The format of this json file is of the form:

```
{"GW150914_095045": "/full/path/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5",
"GW151226_033853": "/full/path/IGWN-GWTC2p1-v2-GW151226_033853_PEDataRelease_mixed_nocosmo.h5",
"GW170104_101158": "/full/path/IGWN-GWTC2p1-v2-GW170104_101158_PEDataRelease_mixed_nocosmo.h5"
...
}
```

## The skymaps of the GW events

The file `/home/albert.einstein/skymaps_dict.json` contains the paths to the skymaps of the same events you specified in the posterior files. The format of this file is of the form:

```
{"GW150914_095045": "/full/path/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_cosmo_reweight_C01:Mixed.fits",
"GW151226_033853": "/full/path/IGWN-GWTC2p1-v2-GW151226_033853_PEDataRelease_cosmo_reweight_C01:Mixed.fits",
"GW170104_101158": "/full/path/IGWN-GWTC2p1-v2-GW170104_101158_PEDataRelease_cosmo_reweight_C01:Mixed.fits",
...
}
```

## The Line-of-Sight (LOS) redshift prior

The file: `/home/albert.einstein/GLADE+_LOS_redshift_prior_K_band_luminosity_weighted_nside_32_pixel_index_None.hdf5` contains the line-of-sight redshift prior. The data stored in this hdf5 file are the values of $p(z|Ωi, \Lambda, I)$ (see Eq. 2.22 of arXiv:2308.02281) for each pixel and the summation over all pixels.


## The injection file for the selection effect

The denominator of Eq. 

## The parameters you want to estimate