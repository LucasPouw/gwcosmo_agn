
Running gwcosmo
===============

Gwcosmo contains two main methods for cosmological inference: the EM counterpart method (also termed the "bright siren" method), which uses the command line `gwcosmo_bright_siren_posterior`, and the galaxy catalogue method (also termed the "dark siren" method) which uses the command line `gwcosmo_dark_siren_posterior`.

Analyses can either be run over a single GW event, in which case the path to a specific posterior samples/skymap file can be provided for the `--posterior_samples` and `--skymap` options, or for a set of GW events, in which case the path to a json file containing a dictionary of all the event names and paths must be provided. An entry in the posterior samples dictionary should be of the following format:
```
{"GW150914": "/path/to/GW150914_posterior_samples", 
"GW151226": "/path/to/GW151226_posterior_samples"} 
```
The same format should be used for the skymaps dictionary. If both posterior samples and skymaps are required, the keys from the posterior samples dictionary must be present in the skymap dictionary.

Many posterior samples files will contain samples from multiple parameter estimation runs. If you do not want to use the default, provide `--posterior_samples_field` with either the name of the required field or a dictionary of names.

Both the bright and dark siren methods require GW selection effects to be estimated. Injections to do so are provided to gwcosmo using the `--injections_path` argument.

To specify which parameters to infer, and over which prior ranges, a dictionary must be provided to the `--parameter_dictionary` argument. Each parameter should be provided with a list (if the parameter is to be inferred) or a float (which the value for that parameter will be fixed to). The list should take the format `[start, stop [, bins]]`. The first two values dictate the prior range. The third value is only used when gwcosmo is run in "gridded" mode, and dictates the resolution of the grid. It defaults to 10 bins if only the first two values are provided. An example dictionary containing all the currently inferrable parameters in gwcosmo is provided in the gwcosmo/data/ folder.


## The bright siren method
GW data can be provided in the form of **LOS posterior samples** (samples conditioned on the line-of-sight of the EM counterpart), **posterior samples** (which are *not* conditioned on the line-of-sight of the EM counterpart, in which case the arguement `--post_los False` should be provided), or a **skymap**.

The counterpart information can be provided in terms of a recession velocity (using `--counterpart_v` and `--counterpart_sigmav`) or a redshift (using `--counterpart_z` and `--counterpart_sigmaz`). If not providing the LOS posterior samples then `--counterpart_ra` and `--counterpart_dec` must also be specified (in radians).


## The dark siren method
GW data must be provided in the form of **posterior samples** and a **skymap**.

A pre-computed line-of-sight redshift prior must be provided for the dark siren method. The `--LOS_catalog` argument should provide the path to the relevant hdf5 file.

## Analysis options

In general, when using gwcosmo for parameter estimation, there are two options, which must be specified using the `--method` argument:

 - Use a `sampling` method, which returns posterior samples for the parameters of interest (best suited for analyses which are carried out over a large number of parameters). Gwcosmo carries out sampling through [Bilby](https://lscsoft.docs.ligo.org/bilby/), and so can make use of any samplers included in the Bilby package. The default sampler is Dynesty, but other options are available: see the bilby documentation for details.
 - Use a `gridded` method, which computes the posterior on an n-dimensional grid with a user-chosen resolution (best suited for analyses constraining one or two parameters).
 
### Sampling method

The output from running the code in "sampling" mode will depend on the chosen sampler. The default sampler is [Dynesty](https://dynesty.readthedocs.io/en/stable/), a nested sampler. Dynesty allows for multiprocessing through the `--npool` argument. When submitting a job on the cluster, it is recommended to set npool to the number of CPUs requested. In this scenario, it is recommended to also set `OMP_NUM_THREADS=1`. For **Condor** this can be acheived by including `environment = "OMP_NUM_THREADS=1"` in your submission file. For **Univa Grid Engine** include `export OMP_NUM_THREADS=1` in your submission file.

### Gridded method

The gridded method will produce two files: **filename.npz** and **filename.png**. The format of the npz file is a list which contains: `[names, param_values, likelihood, opts, parameter_dict]`. "Names" refers to the parameters which have been gridded over (eg H0), "param\_values" is the array of parameter values. "Likelihood" is the *unnormalised* likelihood evaluated at each of the parameter values. "Opts" are the command line arguments which were provided, and "parameter\_dict" is the originally provided parameter dictionary which contains details of the fixed parameters.


### Other arguments of interest

There are also several optional arguments which are common between the bright and dark siren analyses. Some of the most important are summarised here. 

`--mass_model` Choose between 'BBH-powerlaw', 'NSBH-powerlaw', 'BBH-powerlaw-gaussian', 'NSBH-powerlaw-gaussian', ‘BBH-broken-powerlaw', 'NSBH-broken-powerlaw' and 'BNS'.

`--redshift_evolution`  Choose between ‘PowerLaw’, ‘Madau’ and ‘None’.

`--gravity_model` Choose between “GR” and “Xi0_n”.

`--snr_threshold` Select the SNR threshold which matches the subset of events you’ve chosen (must be compatible with provided injections).


