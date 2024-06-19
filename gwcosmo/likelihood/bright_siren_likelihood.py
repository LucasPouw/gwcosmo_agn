"""
Multi-parameter likelihood Module for GW events with electromagnetic counterpart

Tathagata Ghosh
"""

import sys
from copy import deepcopy

import astropy.constants as const
import bilby
import gwcosmo
import numpy as np
from gwcosmo.likelihood.posterior_samples import *
from gwcosmo.likelihood.skymap import *
from gwcosmo.utilities.cosmology import standard_cosmology
from gwcosmo.utilities.posterior_utilities import str2bool
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from scipy.stats import truncnorm


class MultipleEventLikelihoodEM(bilby.Likelihood):

    def __init__(
        self,
        posterior_samples_dictionary,
        injections,
        zrates,
        cosmo,
        mass_priors,
        network_snr_threshold=12.0,
        ifar_cut=0.0,
    ):
        """
        Class to calculate log-likelihood on cosmological and population hyper-parameters.

        parameters
        ------------------

        posterior_samples_dictionary : dictionary
            Dictionary to store the GW events to be used in the analysis.
            The dictionary holds several informations related to the event such as the path to
            the posterior samples, the skymap or the counterpart information.
                Structure of the dictionary:
                 {"GW170817": {"use_event": "True",
                               "posterior_file_path": "GW170817_GWTC-1.hdf5",
                               "posterior_los": "True",
                               "counterpart_velocity": [3017, 166]/c,
                               "counterpart_ra_dec": [3.44602385, -0.40813555]}}
                with counterpart redshift being [3017, 166]/c=[mu/c, sigma/c] and
                mu, sigma from Gaussian distribution; c: speed of light in km/s.
        injections : injection object
            The injection object from gwcosmo to calculate selection effects.
                Pre-computed by using gwcosmo.gwcosmo.create_injections.py
        zrates : gwcosmo.utilities.host_galaxy_merger_relations object
            Object of merger rate evolution with redshift.
        cosmo : gwcosmo.utilities.cosmology object
            Object of cosmological model.
        mass_priors : gwcosmo.prior.priors object
            Object of mass model: For example, BNS, NSBH_powerlaw.
        network_snr_threshold : float
            Network SNR threshold of GW events which are used for analysis.
        ifar_cut : float

        """

        super().__init__(
            parameters={
                "H0": None,
                "Xi0": None,
                "n": None,
                "gamma": None,
                "Madau_k": None,
                "Madau_zp": None,
                "alpha": None,
                "delta_m": None,
                "mu_g": None,
                "sigma_g": None,
                "lambda_peak": None,
                "alpha_1": None,
                "alpha_2": None,
                "b": None,
                "mminbh": None,
                "mmaxbh": None,
                "beta": None,
                "alphans": None,
                "mminns": None,
                "mmaxns": None,
            }
        )

        # mass distribution
        self.mass_priors = mass_priors

        # cosmology
        self.cosmo = cosmo

        # prior redshift distribution: uniform in comoving volume
        self.zprior = self.cosmo.p_z

        # Event information
        self.events = {}

        for event_name, meta in posterior_samples_dictionary.items():
            if not str2bool(meta.get("use_event", "True")):
                print(f"Event '{event_name}' not used in the analysis")
                continue

            if meta.get("posterior_file_path") and meta.get("skymap_path"):
                raise ValueError(
                    "Posterior sample mode can not live together with skymap mode! "
                    + "Within your json file, choose either 'posterior_file_path' "
                    + f"or 'skymap_path' for '{event_name}'."
                )

            # Add new empty event
            event = self.events.setdefault(event_name, {})

            # Add posterior samples if any
            if meta.get("posterior_file_path"):
                event.update(
                    posterior_samples=load_posterior_samples(meta),
                    log_likelihood_numerator=self.log_likelihood_numerator_single_event_from_samples,
                )

            # Counterpart settings
            if counterpart_redshift := meta.get("counterpart_redshift"):
                redshift = counterpart_redshift
            elif counterpart_velocity := meta.get("counterpart_velocity"):
                redshift = counterpart_velocity / const.c.to("km/s").value
            else:
                raise ValueError(
                    f"Missing either 'counterpart_redshift' or 'counterpart_velocity' for '{event_name}' event!"
                )
            counterpart_muz, counterpart_sigmaz = redshift
            zmin = max(0.0, counterpart_muz - 5 * counterpart_sigmaz)
            zmax = counterpart_muz + 5 * counterpart_sigmaz
            a = (zmin - counterpart_muz) / counterpart_sigmaz
            b = (zmax - counterpart_muz) / counterpart_sigmaz
            event.update(counterpart_pdf=truncnorm(a, b, counterpart_muz, counterpart_sigmaz))
            event.update(counterpart_zmin_zmax=np.array([zmin, zmax]))

            posterior_los = str2bool(meta.get("posterior_los", "True"))
            event.update(posterior_los=posterior_los)
            if not posterior_los:
                if not (counterpart_ra_dec := meta.get("counterpart_ra_dec")):
                    raise ValueError(f"Missing 'counterpart_ra_dec' for '{event_name}' event!")
                ra_los, dec_los = counterpart_ra_dec

                if not (samples := event.get("posterior_samples")):
                    raise ValueError(
                        "Posterior los option set to False but no posterior samples found ! "
                        + "Make sure to add a 'posterior_file_path' field in the json file."
                    )
                nsamp_event = int(meta.get("nsamps", 1000))
                sample_index, ang_rad_max = identify_samples_from_posterior(
                    ra_los, dec_los, samples.ra, samples.dec, nsamp_event
                )
                event.update(sample_index=sample_index)
                print(
                    f"Considering {nsamp_event} samples around line of sight for '{event_name}' event"
                )

            if skymap_path := meta.get("skymap_path"):
                if not (counterpart_ra_dec := meta.get("counterpart_ra_dec")):
                    raise ValueError(f"Missing 'counterpart_ra_dec' for '{event_name}' event!")
                skymap = gwcosmo.likelihood.skymap.skymap(skymap_path)
                ra_los, dec_los = counterpart_ra_dec
                dlmin, dlmax, dlpost = skymap.lineofsight_posterior_dl(ra_los, dec_los)
                dl_array = np.linspace(dlmin if dlmin > 0 else 0, dlmax, 10000)
                event.update(
                    posterior_dl_skymap=dlpost,
                    dlarray=dl_array,
                    log_likelihood_numerator=self.log_likelihood_numerator_single_event_from_skymap,
                )

                skymap_prior_distance = meta.get("skymap_prior_distance", "dlSquare") # search for "skymap_prior_distance", if not found set it to "dlSquare"
                if skymap_prior_distance not in ["Uniform", "UniformComoving", "dlSquare"]:
                    raise ValueError(
                        f"Unkown '{skymap_prior_distance}' skymap prior distance for event '{event_name}'! "
                        + "Must be either ['Uniform', 'UniformComoving', 'dlSquare']"
                    )
                event.update(skymap_prior_distance=skymap_prior_distance)
                if skymap_prior_distance == "UniformComoving":
                    cosmo_skymap = standard_cosmology(
                        meta.get("skymap_H0", 67.90), meta.get("skymap_Omega_m", 0.3065) # see default values in https://dcc.ligo.org/DocDB/0167/T2000185/005/LVC_symbol_convention.pdf
                    )
                    zmin, zmax = 0, 10
                    z_array = np.linspace(zmin, zmax, 10000)
                    dl_array = cosmo_skymap.dgw_z(z_array)
                    z_prior_skymap = cosmo_skymap.p_z(z_array)
                    event.update(dl_prior_skymap=interp1d(dl_array, z_prior_skymap))

            # Sanity check
            if not event.get("log_likelihood_numerator"):
                raise ValueError(
                    f"Something is mis-configured for event '{event_name}'! "
                    + "Missing either posterior samples or skymap."
                )

        # redshift evolution model
        self.zrates = zrates

        # selection effect
        self.injections = deepcopy(injections)
        self.injections.update_cut(snr_cut=network_snr_threshold, ifar_cut=ifar_cut)
        # it's the number of GW events entering the analysis, used for the check Neff >= 4Nobs inside the injection class
        self.injections.Nobs = len(self.events)

        print(f"Bright siren likelihood runs with the following event settings: {self.events}")

    def log_likelihood_numerator_single_event_from_samples(self, event_name):

        current_event = self.events[event_name]

        samples = current_event["posterior_samples"]
        z_samps, m1_samps, m2_samps = self.reweight_samps.compute_source_frame_samples(
            samples.distance, samples.mass_1, samples.mass_2
        )
        PEprior = samples.pe_priors
        if current_event["posterior_los"]:
            kde, norm = self.reweight_samps.marginalized_redshift_reweight(
                z_samps, m1_samps, m2_samps, PEprior
            )
        else:
            sample_index = current_event["sample_index"]
            kde, norm = self.reweight_samps.marginalized_redshift_reweight(
                z_samps[sample_index],
                m1_samps[sample_index],
                m2_samps[sample_index],
                PEprior[sample_index],
            )

        redshift_bins = 1000
        zmin = self.cosmo.z_dgw(np.amin(samples.distance)) * 0.5
        zmax = self.cosmo.z_dgw(np.amax(samples.distance)) * 2.0
        z_array_temp = np.linspace(zmin, zmax, redshift_bins)
        px_zOmegaH0_interp = interp1d(
            z_array_temp, kde(z_array_temp), kind="linear", bounds_error=False, fill_value=0
        )  # interpolation may produce some -ve values when kind='cubic'
        num_x = (
            lambda x: px_zOmegaH0_interp(x)
            * self.zrates(x)
            * current_event["counterpart_pdf"].pdf(x)
        )
        zmin, zmax = current_event["counterpart_zmin_zmax"]
        num, _ = quad(num_x, zmin, zmax)

        return np.log(num * norm)

    def log_likelihood_numerator_single_event_from_skymap(self, event_name):

        current_event = self.events[event_name]

        zmin = self.cosmo.z_dgw(current_event["dlarray"][0]) * 0.5
        zmax = self.cosmo.z_dgw(current_event["dlarray"][-1]) * 2.0
        redshift_bins = 10000
        z_array_temp = np.linspace(zmin, zmax, redshift_bins)
        dlarr_given_H0 = self.cosmo.dgw_z(z_array_temp)

        skymap_prior_distance = current_event["skymap_prior_distance"]
        posterior_dl_skymap = current_event["posterior_dl_skymap"]
        if skymap_prior_distance == "dlSquare":
            likelihood_x_z_H0 = posterior_dl_skymap.pdf(dlarr_given_H0) / dlarr_given_H0**2
        elif skymap_prior_distance == "Uniform":
            likelihood_x_z_H0 = posterior_dl_skymap.pdf(dlarr_given_H0)
        elif skymap_prior_distance == "UniformComoving":
            likelihood_x_z_H0 = posterior_dl_skymap.pdf(dlarr_given_H0) / current_event[
                "dl_prior_skymap"
            ](dlarr_given_H0)
        likelihood_x_z_H0 /= simpson(likelihood_x_z_H0, z_array_temp)

        px_zOmegaH0_interp = interp1d(
            z_array_temp, likelihood_x_z_H0, kind="linear", bounds_error=False, fill_value=0
        )

        num_x = (
            lambda x: px_zOmegaH0_interp(x)
            * self.zrates(x)
            * current_event["counterpart_pdf"].pdf(x)
        )
        zmin, zmax = current_event["counterpart_zmin_zmax"]
        num, _ = quad(num_x, zmin, zmax)

        return np.log(num)

    def log_likelihood_denominator_single_event(self):

        zmin = 0
        zmax = 10
        z_array = np.linspace(zmin, zmax, 10000)
        z_prior = interp1d(z_array, self.zprior(z_array) * self.zrates(z_array))
        dz = np.diff(z_array)
        z_prior_norm = np.sum((z_prior(z_array)[:-1] + z_prior(z_array)[1:]) * (dz) / 2)
        # no need for deepcopy (mattermost bilby.help channel, 20240619), Colm Talbot wrote:
        # "Each thread has it's own copy of the likelihood object, so there's no need for copying."
        #injections = deepcopy(self.injections)

        # Update the sensitivity estimation with the new model
        self.injections.update_VT(self.cosmo, self.mass_priors, z_prior, z_prior_norm)
        Neff, Neff_is_ok, var = self.injections.calculate_Neff()
        if Neff_is_ok:  # Neff >= 4*Nobs
            log_den = np.log(injections.gw_only_selection_effect())
        else:
            print(
                f"Not enough Neff ({Neff}) compared to Nobs ({self.injections.Nobs}) "
                + f"for current mass-model {self.mass_priors} and z-model {z_prior}"
            )
            print(
                f"mass prior dict: {self.mass_priors_param_dict}, "
                + f"cosmo_prior_dict: {self.cosmo_param_dict}"
            )
            print("returning infinite denominator")
            log_den = np.inf

        return log_den, np.log(z_prior_norm)

    def log_combined_event_likelihood(self):

        den_single, zprior_norm_log = self.log_likelihood_denominator_single_event()
        den = den_single * len(self.events)

        num = 0.0
        for event_name, meta in self.events.items():
            log_likelihood_numerator = meta.get("log_likelihood_numerator")
            num += log_likelihood_numerator(event_name) - zprior_norm_log

        return num - den

    def log_likelihood(self):

        self.cosmo_param_dict = {par: self.parameters[par] for par in ["H0", "Xi0", "n"]}
        self.cosmo.update_parameters(self.cosmo_param_dict)

        self.zrates.gamma = self.parameters["gamma"]
        self.zrates.k = self.parameters["Madau_k"]
        self.zrates.zp = self.parameters["Madau_zp"]

        self.mass_priors_param_dict = {
            par: self.parameters[par]
            for par in [
                "alpha",
                "delta_m",
                "mu_g",
                "sigma_g",
                "lambda_peak",
                "alpha_1",
                "alpha_2",
                "b",
                "mminbh",
                "mmaxbh",
                "beta",
                "alphans",
                "mminns",
                "mmaxns",
            ]
        }
        self.mass_priors.update_parameters(self.mass_priors_param_dict)

        # This is only needed by posterior samples mode. The reweight_posterior_samples class
        # initialization is just a reference assignation of self.cosmo and self.mass_prior: since
        # self.cosmo and self.mass_priors remains the same during all the likelihood computation, we
        # can create the self.reweight_samps object at the beginning once and for all. For clarity
        # reasons, we create it here every time just to reflect the change of cosmology and mass
        # prior contents.
        self.reweight_samps = reweight_posterior_samples(self.cosmo, self.mass_priors)

        return self.log_combined_event_likelihood()

    def __call__(self):
        return np.exp(self.log_likelihood())
