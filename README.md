# Welcome to gwcosmo

A package to estimate cosmological parameters using gravitational-wave observations. 


If you use **gwcosmo** in a scientific publication, please cite [R. Gray et al. Phys. Rev. D 101, 122001](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.122001) and [R. Gray et al. arXiv:2111.04629](https://arxiv.org/abs/2111.04629), and include the following statement in your manuscript: "This work makes use of gwcosmo which is available at https://git.ligo.org/lscsoft/gwcosmo".

# How-to install

* Clone the **gwcosmo** repository with 
    ```
    git clone <repository>
    ```
    The name of the repository can be copied from the git interface (top right button). If you do not have ssh key on git, please use the `https` protocol
* Complete the install by following one of the options below. Note that `gwcosmo` requires Python version 3.7-3.9 to run.

## Installing with Anaconda

You will need an [Anaconda distribution](https://www.anaconda.com/). The conda distribution is correctly initialized when, if you open your terminal, you will see the name of the python environment used. The default name is `(base)`.

Once the conda distribution is installed and activated on your machine, please follow these steps:

* Enter the cloned **gwcosmo** directory.

* Create a conda virtual environment to host gwcosmo. Use
```
conda create -n gwcosmo
```
* When the virtual environment is ready, activate it with (your python distribution will change to `gwcosmo`)
```
conda activate gwcosmo
```
* Install **gwcosmo** by running 
```
python setup.py install
```
* You are ready to use **gwcosmo**. Note that, if you modify the code, you can easily reinstall it by using
```
python setup.py install --force
```

## Installing with pip and venv

`venv` is included in Python for versions >=3.3.

* Create a virtual environment to host **gwcosmo**. Use
```
python -m venv env
```
* When the virtual environment is ready, activate it with
```
source env/bin/activate
```
* Enter the cloned gwcosmo directory.
* Install **gwcosmo** by running 
```
pip install .
```
* Alternatively, if you are planning to modify **gwcosmo** run the following instead:
```
pip install -e .
```
The `-e` stands for "editable" and means that your installation will automatically update when you make changes to the code.


# Data required for running **gwcosmo**
This section describes the data required in order to run **gwcosmo**.

## Gravitational wave data
All of the analyses outlined below require some form of gravitational wave data as input. This comes in two forms: posterior samples (saved as hdf5 file format), and skymaps (saved in fits format). These are publicly released following LIGO/Virgo/KAGRA observing runs. The data from the first gravitational wave transient catalogue (GWTC-1) is linked below:

[GWTC-1 posterior samples](https://dcc.ligo.org/LIGO-P1800370/public)

[GWTC-1 skymaps](https://dcc.ligo.org/LIGO-P1800381/public)

## Probability of detection
Another necessary input of all the **gwcosmo** analyses is the gravitational wave probability of detection ("Pdet" for short), as a function of redshift and $`H_0`$, which allows **gwcosmo** to account for gravitational wave selection effects. Pdet must be precomputed. For convenience, several precomputed values of Pdet are included within the **gwcosmo** installation, under gwcosmo/data/FILENAME.p, where FILENAME describes the detector sensitivity and mass distribution used to generate it.

For example, `O1PSD_BBH-powerlaw_alpha_1.6_Mmin_5.0_Mmax_100.0_Nsamps20000_full_waveform_snr_12.0.p` corresponds to O1-like sensitivity for a population of binary black holes with a source-frame mass distribution described by a powerlaw with slope -1.6, a minimum mass cutoff of 5 solar masses, and a maximum mass cutoff of 100 solar masses, assuming a network SNR threshold of 12. The options that were used to generate each pickle are stored within the pickle, and can be accessed by running the following:
```
$ import pickle
$ file = '/path/to/precomputed/pdet'
$ pdet = pickle.load(open(file, 'rb'))
$ print(pdet.__dict__)
```

If none of the precomputed Pdets are of use, the section "Precomputing probability of detection" details how to generate your own probability of detection curves using **gwcosmo**. Please note that the choice of compact binary mass distribution and matter fraction of the universe ($`\Omega_M`$) must be consistent throughout the **gwcosmo** analysis, and so will be read in from the provided Pdet.

## Galaxy catalogues
The galaxy catalogue analyses require that a pre-processed galaxy catalogue be provided to **gwcosmo**. At the moment, the compatible catalogues are the GLADE 2.4 galaxy catalogue and the GLADE+ galaxy catalogue (see http://glade.elte.hu/ for details of what these catalogues contain, as well as citation instructions if you use them in an analysis).

Scripts and instructions for generating pre-processed versions of these catalogues can be found under scripts_galaxy_catalogs/.

# Computing the posterior on $`H_0`$ for a single gravitational wave event
The main executable you will be using is called `gwcosmo_single_posterior`. It calculates the posterior on $`H_0`$ for a single gravitational wave event.

There are 4 main options for running gwcosmo with gravitational wave data:
1. **The counterpart method:** the GW data is used in conjunction with a known associated host galaxy to measure $`H_0`$.
1. **The population method:** the empty catalogue method, where no EM information is assumed, and all information on $`H_0`$ comes from chosen priors, and the mass and distance information of the GW event.
1. **The statistical method:** the GW data is used in conjunction with a galaxy catalogue which updates the redshift prior using known galaxies, and assumes uniform galaxy catalogue incompleteness across the sky-area of the GW event.
1. **The pixel method:** an improvement to the statistical method, where its applied on a pixel-by-pixel basis.

The default settings for all methods will use a uniform in comoving volume prior on redshift for unknown galaxies, assume no GW rate evolution with redshift, and will reweight GW posterior samples to use the source-frame mass prior that the GW selection effects were calculated assuming.  If using one of the galaxy catalogue methods, the Schechter function parameters (used to calculate the out-of-catalog contribution) will be determined from the `--band` provided, and luminosity weighting is assumed by default.

Running `$ /path/to/executable/gwcosmo_single_posterior --help` provides information on all possible command-line arguments.

```
Options:
  --method=METHOD       counterpart, statistical, population, pixel (required)
  --min_H0=MIN_H0       Minimum value of H0
  --max_H0=MAX_H0       Maximum value of H0
  --bins_H0=BINS_H0     Number of H0 bins
  --posterior_samples=POSTERIOR_SAMPLES
                        Path to LALinference posterior samples file in format
                        (.dat or hdf5)
  --posterior_samples_field=POSTERIOR_SAMPLES_FIELD
                        Internal field of the posterior samples file, e.g. h5
                        or json field
  --skymap=SKYMAP       Path to LALinference 3D skymap file in format (.fits
                        or fits.gz)
  --Pdet=PDET           Path to precomputed probability of detection pickle
  --redshift_uncertainty=REDSHIFT_UNCERTAINTY
                        Marginalise over redshift uncertainties (default=True)
  --counterpart_ra=COUNTERPART_RA
                        Right ascension of counterpart
  --counterpart_dec=COUNTERPART_DEC
                        Declination of counterpart
  --counterpart_z=COUNTERPART_Z
                        Redshift of counterpart (in CMB frame)
  --counterpart_sigmaz=COUNTERPART_SIGMAZ
                        Uncertainty of counterpart in redshift
  --counterpart_v=COUNTERPART_V
                        Recessional velocity of counterpart in km/sec (in CMB
                        frame)
  --counterpart_sigmav=COUNTERPART_SIGMAV
                        Uncertainty of counterpart in km/sec
  --redshift_evolution=REDSHIFT_EVOLUTION
                        Allow GW host probability to evolve with redshift.
                        Select between None, PowerLaw or Madau (Default=None)
  --Lambda=LAMBDA       Set rate evolution parameter Lambda for redshift
                        evolution (For Madau model this is equal to alpha)
  --Madau_beta=MADAU_BETA
                        Set Beta for Madau model. (Not used if
                        redshift_evolution=None or PowerLaw)
  --Madau_zp=MADAU_ZP   Set zp for Madau model. (Not used if
                        redshift_evolution=None or PowerLaw)
  --Kcorrections=KCORRECTIONS
                        Apply K-corrections.
  --reweight_posterior_samples=REWEIGHT_POSTERIOR_SAMPLES
                        Reweight posterior samples with the same priors used
                        to calculate the selection effects.
  --zmax=ZMAX           Upper redshift limit for integrals (default=10)
  --galaxy_weighting=GALAXY_WEIGHTING
                        Weight potential host galaxies by luminosity?
                        (Default=True)
  --assume_complete_catalog=ASSUME_COMPLETE_CATALOG
                        Assume a complete catalog? (Default=False)
  --zcut=ZCUT           Hard redshift cut to apply to the galaxy catalogue
                        (default=none)
  --mth=MTH             Override the apparent magnitude threshold of the
                        catalogue, if provided (default=None)
  --schech_alpha=SCHECH_ALPHA
                        Override the default value for slope of schechter
                        function for given band, if provided (default=None)
  --schech_Mstar=SCHECH_MSTAR
                        Override the default value for Mstar of schechter
                        function for given band, if provided (default=None)
  --schech_Mmin=SCHECH_MMIN
                        Override the default value for Mmin of schechter
                        function for given band, if provided (default=None)
  --schech_Mmax=SCHECH_MMAX
                        Override the default value for Mmax of schechter
                        function for given band, if provided (default=None)
  --nside=NSIDE         skymap nside choice for reading in galaxies from the
                        overlap of catalogue and skymap (default=32)
  --sky_area=SKY_AREA   contour boundary for galaxy catalogue method
                        (default=0.999)
  --pixel_index=PIXEL_INDEX
                        index of the skymap pixel to analyse (for use with
                        pixel method only)
  --min_pixels=MIN_PIXELS
                        minimum number of pixels desired to cover sky area of
                        event (for use with pixel method only)
  --return_skymap_indices=RETURN_SKYMAP_INDICES
                        Return the skymap indices needed to run the pixelated
                        method (for use with pixel method only)
  --combine_pixels=COMBINE_PIXELS
                        combine multiple pixels to make the full likelihood
                        for an event. Folder must contain pixel likelihoods
                        and pixel indices file. (for use with pixel method
                        only)
  --outputfile=OUTPUTFILE
                        Name of output file
  --seed=SEED           Random seed
  --numerical=NUMERICAL
                        If set to true numerical integration will be used for
                        the calculation of integrals
  -h, --help            show this help message and exit

  Galaxy Catalog Options:
     Use these options to control the galaxy catalog input

    --catalog=NAME      Specify a galaxy catalog by name. Known catalogs are:
                        DESI, DES, GLADE, GLADE+
    --catalog_band=CATALOG_BAND
                        Observation band of galaxy catalog
                        (B,K,W1,bJ,u,g,r,i,z) (must be compatible with the
                        catalogue provided)
```

When running the counterpart, population or statistical methods, the output from **gwcosmo** comes in the form of files called **eventname.npz** and **eventname_likelihood_breakdown.npz**, and a figure **eventname.pdf**.

**eventname.npz** contains the following: [H0,likelihood,posterior_uniform_norm,posterior_log_norm,opts]

While **eventname_likelihood_breakdown.npz** contains [H0, likelihood, pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO], where `likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO`.

These files can be read by running
```
import numpy as np
file = 'path/to/data/file.npz'
data = np.load(file, allow_pickle=True)['arr_0']
```

The output of running the pixelated method is slightly more complicated and is explained in detail in the Pixelated section. In general, however, each pixel will have an associated file called **eventname_pixel_i.npz**, which is in the same format as **eventname_likelihood_breakdown.npz**. However in the pixelated case, the likelihood is only for the pixel under consideration.

## The EM counterpart method
When passed a set of GW samples and the redshift or recessional velocity of an EM counterpart, and its uncertainty, computes a posterior on $`H_0`$. Currently assumes that the GW posterior samples have already been conditioned on the line-of-sight of the EM counterpart.

### Example run through

```
$ /path/to/executable/gwcosmo_single_posterior \
--method counterpart \
--posterior_samples /path/to/posterior/samples \
--Pdet /path/to/precomputed/pdet \
--counterpart_v 3017 \
--counterpart_sigmav 166 \
--min_H0 20 \
--max_H0 140 \
--bins_H0 50 \
--reweight_posterior_samples True \
--outputfile eventname
```

The arguments `--counterpart_v` and `--counterpart_sigmav` are for the recession velocity of the counterpart (in km s-1 Mpc-1) and its uncertainty (also in km s-1 Mpc-1). Alternative, `--counterpart_z` and `--counterpart_sigmaz` can be used to provide the counterpart redshift and redshift uncertainty. **gwcosmo** will assume that the counterpart recession velocity or redshift is has already been corrected for peculiar motion.

Running the command line above should take a matter of seconds. The printed output will look something like the following: 

```
Selected method is: counterpart
Loading precomputed pdet with a BNS mass distribution at O2 sensitivity assuming an SNR threshold of 12.0.
Setting up a cosmology with Omega_m=0.308
Loading posterior samples
Setting up p(x|z,H0)
Uniform Prior
H0 = 70 + 22 - 9 (MAP and 68.3 percent HDI)
Log Prior
H0 = 69 + 18 - 8 (MAP and 68.3 percent HDI)
```

## The fixed population method
When passed a set of GW samples, computes a posterior on $`H_0`$ using purely population information (also known as the empty-catalogue method). This is a relatively uninformative but quick analysis, and can be used to assess to what extent GW population information is driving an analysis, as opposed to EM information in the form of catalogues or counterparts.

### Example run through

```
$ /path/to/executable/gwcosmo_single_posterior \
--method population \
--posterior_samples /path/to/posterior/samples \
--Pdet /path/to/precomputed/pdet \
--min_H0 20 \
--max_H0 140 \
--bins_H0 50 \
--reweight_posterior_samples True \
--outputfile eventname
```

Running the command line above should take a matter of seconds. The printed output will look something like the following: 

```
Selected method is: population
Loading precomputed pdet with a BBH-powerlaw mass distribution at O1 sensitivity assuming an SNR threshold of 12.0.
BBH-powerlaw mass distribution with parameters: alpha=-1.6, beta=0.0, Mmin=5.0, Mmax=100.0
Setting up a cosmology with Omega_m=0.308
Loading posterior samples
Setting up p(x|z,H0)
Uniform Prior
H0 = 140 + -140 - 79 (MAP and 68.3 percent HDI)
Log Prior
H0 = 20 + -20 - -19 (MAP and 68.3 percent HDI)
```

## The galaxy catalogue method (quick approximation)

When passed a set of GW samples, a GW skymap, and a galaxy catalogue, computes a quick posterior on $`H_0`$. Assumes uniform catalogue completeness in the overlap between GW sky area and catalogue. Assumes GW distance and sky are separable, leading to a slightly less informative result.

### Example run through

```
$ export GWCOSMO_CATALOG_PATH=/path/to/catalog/directory/
$ /path/to/executable/gwcosmo_single_posterior \
--method statistical \
--posterior_samples /path/to/posterior/samples \
--skymap /path/to/skymap \
--Pdet /path/to/precomputed/pdet \
--galaxy_catalog catalogue_name \
--catalog_band band \
--redshift_uncertainty True \
--galaxy_weighting True \
--assume_complete_catalog False \
--min_H0 20 \
--max_H0 140 \
--bins_H0 50 \
--reweight_posterior_samples True \
--nside 64 \
--sky_area 0.9999 \
--outputfile eventname
```

Here, the choice of `--nside` determines the resolution at which to crossmatch the GW event and the galaxy catalogue in order to determine 1) the fraction of the sky covered by both GW event and catalogue and 2) the amount of GW sky area probability covered by the galaxy catalogue. A value of `--nside` which is "too high" will cause some pixels to incorrectly evaluate as empty, and artificially inflate the "out-of-catalogue" contribution, leading to a less informative result. A value that is "too low" will not accurately capture any hard edges to the catalogue within the GW sky area, and may artificially inflate or deflate the relative "in-catalogue" and "out-of-catalogue" contributions.

The choice of `--sky_area` determines the fraction of the GW event sky area used in the analysis for finding the overlap between galaxy catalogue and GW event. A smaller value for `--sky_area` will lead to a smaller patch of sky, and therefore a more accurate estimate of $`m_{th}`$. It is important to note that any GW sky area not included here will be automatically added on as a relatively uninformative "outside-the-catalogue" contribution.

The command line above should take a minute or two to run. The printed output will look something like the following:

```
Selected method is: statistical
Loading precomputed pdet with a BBH-powerlaw mass distribution at O1 sensitivity assuming an SNR threshold of 12.0.
BBH-powerlaw mass distribution with parameters: alpha=-1.6, beta=0.0, Mmin=5.0, Mmax=100.0
Setting up a cosmology with Omega_m=0.308
Loading posterior samples
Setting up p(x|z,H0)
Schechter function with parameters: alpha=-1.21, Mstar=-19.7, Mmin=-22.96, Mmax=-12.96, 
Computing magnitude threshold for None,None
Ngal = 70188
Catalogue apparent magnitude threshold: 17.805
mth in this sky patch: 17.805
Ngal in this sky patch: 35101
Computing the in-catalogue part
4 galaxies are getting sampled finely
Using K-correction = 0
35097 galaxies are getting sampled coarsely
Using K-correction = 0
Computing the beyond catalogue part
97.77563839801141% of this event's sky area appears to have galaxy catalogue support
Computing the contribution outside the catalogue footprint
Uniform Prior
H0 = 140 + -140 - 76 (MAP and 68.3 percent HDI)
Log Prior
H0 = 20 + -20 - -23 (MAP and 68.3 percent HDI)
```

## The galaxy catalogue method (pixelated)

When passed a set of GW samples, a GW skymap, and a galaxy catalogue, computes the on $`H_0`$. Allows non-uniform catalogue completeness, and makes full use of the GW data.

When running the pixelated method, it is recommended to use a computer cluster in order to allow for the analysis to be parallelised, as it will be time consuming to run each pixel one after the other. The section "Generating a DAG" (below) outlines one easy way of doing so.

### Example run through

First set the galaxy catalogue path by running

```
$ export GWCOSMO_CATALOG_PATH=/path/to/catalog/directory/
```

Then follow the steps below.

#### Step 1: identify the pixel indices

```
$ /path/to/executable/gwcosmo_single_posterior \
--method pixel \
--posterior_samples /path/to/posterior/samples \
--skymap /path/to/skymap \
--Pdet /path/to/precomputed/pdet \
--galaxy_catalog catalogue_name \
--catalog_band band \
--min_H0 20 \
--max_H0 140 \
--bins_H0 50 \
--reweight_posterior_samples True \
--nside 32 \
--sky_area 0.999 \
--min_pixels 30 \
--return_skymap_indices True \
--outputfile eventname
```

This will return a message similar to the following:
```
Selected method is: pixel
Loading precomputed pdet with a BBH-powerlaw mass distribution at O1 sensitivity assuming an SNR threshold of 12.0.
BBH-powerlaw mass distribution with parameters: alpha=-1.6, beta=0.0, Mmin=5.0, Mmax=100.0
Setting up a cosmology with Omega_m=0.308
Loading posterior samples
Schechter function with parameters: alpha=-1.21, Mstar=-19.7, Mmin=-22.96, Mmax=-12.96, 
35 pixels to cover the 99.9% sky area (nside=8)
No pixel index found for nside 8, generating now.
```
This will produce a txt file called **eventname_indices.txt**, which will list the all the pixel indices which cover the event's sky area, assuming the printed value of nside, where this value of nside is chosen internally by gwcosmo to be the lowest resolution which satisfies the criteria given using the `--sky_area` and `--min_pixels` arguments.

Here, it is important to note that (unlike in the statistical method), only the GW probability defined by `--sky_area` is included in the final result, and so reducing the value for `--sky_area` can artificially inflate how informative an event is.

`--min_pixels` simply determines the minimum number of pixels desired to cover the GW sky area. If the number of pixels here is not high enough, increase `--min_pixels` to one more than the number of pixels printed above, and this will increase the internal resolution by one step (from e.g. nside =  8 to nside = 16).

#### Step 2: Run each pixel

This step needs to be run for each pixel, i, listed in **eventname_indices.txt**. Fortunately, these can be parallelised over a cluster.

```
$ /path/to/executable/gwcosmo_single_posterior \
--method pixel \
--posterior_samples /path/to/posterior/samples \
--skymap /path/to/skymap \
--Pdet /path/to/precomputed/pdet \
--galaxy_catalog catalogue_name \
--catalog_band band \
--min_H0 20 \
--max_H0 140 \
--bins_H0 50 \
--reweight_posterior_samples True \
--nside 32 \
--sky_area 0.999 \
--min_pixels 30 \
--pixel_index i \
--outputfile eventname
```

This will return files of the format **eventname_pixel_i.npz**.  Here the `--nside` argument becomes important, as it determines the resolution of the sub-pixels that each pixel will be split into, and so affects 1) how long each pixel will take to run and 2) how good the resolution of the galaxy catalogue is. A value of `--nside` which is "too high" will cause the estimate of $`m_{th}`$ to be unreliable, and cause some pixels to incorrectly evaluate as empty. A value that is "too low" will not accurately capture any hard edges to the catalogue, or abrupt changes in completeness. The printed output for a single pixel will look something like the following:

```
Selected method is: pixel
Loading precomputed pdet with a BBH-powerlaw mass distribution at O1 sensitivity assuming an SNR threshold of 12.0.
BBH-powerlaw mass distribution with parameters: alpha=-1.6, beta=0.0, Mmin=5.0, Mmax=100.0
Setting up a cosmology with Omega_m=0.308
Loading posterior samples
Schechter function with parameters: alpha=-1.21, Mstar=-19.7, Mmin=-22.96, Mmax=-12.96, 
35 pixels to cover the 99.9% sky area (nside=8)
angular radius: 0.06521271776725232 radians, No. samples: 7759
Setting up p(x|z,H0)
Pixel 584 at nside=8 contains pixels [9344 9345 9346 9347 9348 9349 9350 9351 9352 9353 9354 9355 9356 9357
 9358 9359] at nside=32
Computing magnitude threshold for 1.658062789394613,-1.3406490668906557
Ngal = 69
Computing magnitude threshold for 1.806415775814131,-1.3149438733849594
Ngal = 78
Computing magnitude threshold for 1.6493361431346414,-1.3149438733849594
Ngal = 59
Computing magnitude threshold for 1.7849958259032916,-1.2891961103650367
Ngal = 117
Computing magnitude threshold for 1.9277954919755549,-1.2891961103650367
Ngal = 82
Computing magnitude threshold for 2.028945255443408,-1.2634012757103932
Ngal = 106
Computing magnitude threshold for 1.8980455615438334,-1.2634012757103932
Ngal = 109
Computing magnitude threshold for 1.9937030301627534,-1.2375547945636298
Ngal = 76
Computing magnitude threshold for 1.6421961598310282,-1.2891961103650367
Ngal = 77
Computing magnitude threshold for 1.7671458676442586,-1.2634012757103932
Ngal = 93
Computing magnitude threshold for 1.636246173744684,-1.2634012757103932
Ngal = 101
Computing magnitude threshold for 1.7520420568096924,-1.2375547945636298
Ngal = 103
Computing magnitude threshold for 1.8728725434862228,-1.2375547945636298
Ngal = 122
Computing magnitude threshold for 1.9634954084936207,-1.2116520114493077
Ngal = 65
Computing magnitude threshold for 1.8512956708654138,-1.2116520114493077
Ngal = 101
Computing magnitude threshold for 1.9373154697137058,-1.1856881820323233
Ngal = 105
mth in this sub-pixel: 17.368
Ngal in this sub-pixel: 35
Computing the in-catalogue part with 35 galaxies
1 galaxies are getting sampled finely
Using K-correction = 0
34 galaxies are getting sampled coarsely
Using K-correction = 0
Computing the beyond catalogue part
mth in this sub-pixel: 17.465
Ngal in this sub-pixel: 39
Computing the in-catalogue part with 39 galaxies
1 galaxies are getting sampled finely
Using K-correction = 0
38 galaxies are getting sampled coarsely
Using K-correction = 0
Computing the beyond catalogue part
...
```
and so on for the rest of the sub-pixels.

For reference, a single pixel which is split into 16 sub-pixels and evaluated for 50 $`H_0`$ bins will take approximately 10 minutes to run, and scales approximately with the number of pixels. Increasing nside by one step (from e.g. 32 to 64) will take ~4 times longer to evaluate, while reducing nside by one step will make it ~4 times faster.

#### Step 3: Combine all pixels

```
$ /path/to/executable/gwcosmo_single_posterior \
--method pixel \
--combine_pixels True \
--outputfile eventname
```

This should be run in the same folder as **eventname_indices.txt** and all **eventname_pixel_i.npz** files and will combine them to a final result with the format **eventname.npz** and a figure **eventname.pdf**. The printed output will look something like the following:

```
Combining 35 pixels
Uniform Prior
H0 = 57 + -57 - -12 (MAP and 68.3 percent HDI)
Log Prior
H0 = 20 + -20 - 20 (MAP and 68.3 percent HDI)
```


### Generating a DAG

In order to simplify things for people running this analysis on a cluster, there is the option to generate a DAG which will automate the three steps above.  In this case you would run:

```
$ /path/to/executable/gwcosmo_pixel_dag \
--posterior_samples /path/to/posterior/samples \
--skymap /path/to/skymap \
--Pdet /path/to/precomputed/pdet \
--catalog catalogue_name \
--catalog_band band \
--min_H0 20 \
--max_H0 140 \
--bins_H0 50 \
--reweight_posterior_samples True \
--nside 32 \
--sky_area 0.999 \
--min_pixels 30 \
--outputfile eventname
```

When run correctly this will output a file called **dagfile.dag**, and two sub files called **single_pixel.sub** and **combine_pixel.sub**, as well as the initial **eventname_indices.txt** file. It will also create a folder called **log/** which will store all of the printed outputs from each stage of the analysis.  By running `$ condor_submit_dag dagfile.dag`, the pixels will be parallelised over the cluster and, once they have all finished running, the results will be automatically combined. **Note:** for those not running on a ligo cluster, make sure to include the argument `--run_on_ligo_cluster False` when generating the DAG. Every cluster is different, so you may need to taylor these files to your specific circumstances.

# Precomputing probability of detection

To precompute your custom pdets there is an executable you can call. Running `/path/to/executable/gwcosmo_compute_pdet --help` will show all the available flags that you can pass. To create a pdet you need to choose a mass distribution for the black holes. The implemented mass distributions are the single powerlaw, the broken-powerlaw and the powerlaw+Gaussian peak (see Appendix A3 from https://arxiv.org/pdf/2111.03604.pdf). The mass distribution for neutron stars is chosen by default to be uniform between 1 and 3 solar masses. 

```
Options:
  --mass_distribution=MASS_DISTRIBUTION
                        Choose between BNS or NSBH/BBH-powerlaw, NSBH/BBH-
                        powerlaw-gaussian, NSBH/BBH-broken-powerlaw mass
                        distributions for default Pdet calculations.
  --psd=PSD             Select between 'O1' and 'O2' and 'O3' PSDs, for
                        default Pdet calculations. By default we use aLIGO at
                        design sensitivity.
  --powerlaw_slope=POWERLAW_SLOPE
                        Set powerlaw slope for BBH powerlaw mass distribution.
  --powerlaw_slope_2=POWERLAW_SLOPE_2
                        Set second powerlaw slope for BBH with broken powerlaw
                        mass distribution.
  --beta=BETA           Set powerlaw slope for the second black hole.
  --minimum_mass=MINIMUM_MASS
                        Set minimum mass in the source frame for BBH (default
                        is 5).
  --maximum_mass=MAXIMUM_MASS
                        Set maximum mass in the source frame for BBH mass
                        distribution (default is 100).
  --mu_g=MU_G           Set the mu of the gaussian peak in case of BBH-
                        powerlaw-gaussian mass distribution.
  --lambda_peak=LAMBDA_PEAK
                        Set the lambda of the gaussian peak in case of BBH-
                        powerlaw-gaussian mass distribution.
  --sigma_g=SIGMA_G     Set the sigma of the gaussian peak in case of BBH-
                        powerlaw-gaussian mass distribution.
  --delta_m=DELTA_M     Set the smoothing parameter in case of BBH-powerlaw-
                        gaussian or BBH-broken-powerlaw mass distributions.
  --b=B                 Set the fraction at which the powerlaw breaks in case
                        of BBH-broken-powerlaw mass distribution.
  --linear_cosmology=LINEAR_COSMOLOGY
                        Assume a linear cosmology.
  --basic_pdet=BASIC_PDET
                        Allow for masses to be redshifted in Pdet using False.
  --full_waveform=FULL_WAVEFORM
                        Use the full waveform to calculate detection
                        probability, otherwise only use the inspiral part
                        (default is True).
  --Nsamps=NSAMPS       The number of samples to use to calculate detection
                        probability (default is 10000).
  --constant_H0=CONSTANT_H0
                        Compute at a fixed H0 value (default is False).
  --min_H0=MIN_H0       Set minimum value of H0 Posterior (default is 20).
  --max_H0=MAX_H0       Set maximum value of H0 Posterior (default is 200).
  --bins_H0=BINS_H0     Set number of H0 Posterior bins (default is 100)
  --H0=H0               Set H0 value when using constant_H0 = True (default is
                        70).
  --combine=COMBINE     Directory of constant_H0 Pdets to combine into single
                        Pdet pickle.
  --outputfile=OUTPUTFILE
                        Name of output pdet file.
  --Omega_m=OMEGA_M     Omega of matter.
  --snr=SNR             Network SNR threshold.
  --detected_masses=DETECTED_MASSES
                        Set to True if you want to keep track of the detected
                        masses.
  --detectors=DETECTORS
                        Set the detectors to use for the pickle (default=HLV).
  --det_combination=DET_COMBINATION
                        Set whether or not to consider all possible detectors
                        combinations (default=True).
  --seed=SEED           Set the random seed.

```
You can calcualte a pdet without using a condor, and by passing values for  `--min_H0, --max_H0` and `--bins_H0` but it will take quite some time (order of days) in a normal pc. It is highly recommended to run one in a cluster by passing values for a constant $`H_0`$ and running in parallel for many $`H_0`$ values. To do this you pass `--constant_H0 True` and `--H0 the_value_that_you_want`. Beware that you need to use different names for different $`H_0`$ outputfiles, otherwise they will be overwritten. You can do that by using the flag `--outputfile`. 

For example, if you want to calculate the pdet for $`H_0=60`$ and $`H_0=70`$, you will need two submit files with the only difference in the flags being the value of $`H_0`$. This will create two pickle files, each containing the pdet for the given $`H_0`$ value. To combine all the different $`H_0`$ pickle files to one you need to run `/path/to/executable/gwcosmo_compute_pdet --combine path_to_the_folder_of_the_constant_H0_files`.

If the flag `--det_combination`  is set to True, the code will create all the possible combinations between detectors based on the real duty factors from O1, O2 and O3 runs. With the flag `--detectors` you can select which detectors you want to take into account for the pdet (For example, `--detectors HLV` will take into account the LIGO Hanford, LIGO Livingston, and Virgo detectors.).
