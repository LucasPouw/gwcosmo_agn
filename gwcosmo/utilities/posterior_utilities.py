# Global Imports
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import fmin
from scipy.interpolate import interp1d, UnivariateSpline


class confidence_interval(object):
    def __init__(self, posterior, param, level=0.683, verbose=False):
        self.posterior = posterior
        self.param = param
        self.level = level
        self.verbose = verbose
        self.lower_level, self.upper_level = self.HDI()
        self.interval = self.upper_level - self.lower_level
        self.map = self.MAP()

    def HDI(self):
        cdfvals = cumtrapz(self.posterior, self.param)
        sel = cdfvals > 0.
        x = self.param[1:][sel]
        cdfvals = cdfvals[sel]
        ppf = interp1d(cdfvals, x, fill_value=0., bounds_error=False)

        def intervalWidth(lowTailPr):
            ret = ppf(self.level + lowTailPr) - ppf(lowTailPr)
            if (ret > 0.):
                return ret
            else:
                return 1e4
        HDI_lowTailPr = fmin(intervalWidth, 1. - self.level, disp=self.verbose)[0]
        return ppf(HDI_lowTailPr), ppf(HDI_lowTailPr + self.level)


    def MAP(self):
        sp = UnivariateSpline(self.param, self.posterior, s=0.)
        x_highres = np.linspace(self.param[0], self.param[-1], 100000)
        y_highres = sp(x_highres)
        return x_highres[np.argmax(y_highres)]

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# define the keys for posterior samples in json file and other keys when running gwcosmo_explore_priors

PE_file_key = "posterior_file_path" # path to the posteriors file (needed)
PE_skymap_file_key = "skymap_path" # path to the event skymap (needed)
PE_samples_field_key = "samples_field" # name of the approximant (online, CO1:...) (optional)
PE_prior_file_key = "PEprior_file_path" # path to the PE prior file (optional)
PE_prior_kind_key = "PEprior_kind" # to use the PE priors internally defined in posterior_samples.py (optional)
PE_use_event_key = "use_event" # to consider or skip the current event in the analysis (optional)
PE_min_pixels = "min_pixels"
PE_prior_class_name = "PE_priors"
PE_analysis_type = "analysis_type"
PE_sampling_vars = "sampling_variables"
PE_multi_analysis = "multi"
PE_single_analysis = "single"
PE_approximants_available = "approximants_available"
PE_approximant_requested = "approximant_requested"
PE_approximant_selected = "approximant_selected"
PE_has_analytic_priors = "has_analytic_priors"
PE_search_analytic_priors_str = "search_analytic_priors"
PE_could_read_with_pesummary = "could_read_with_pesummary"
PE_user_defined_PE = "user_defined_PE_priors"
PE_existing_PE_kinds = ["m1d_m2d_uniform_dL_square_PE_priors",
                        "chirp_det_frame_q_uniform_dL_square_PE_priors",
                        "m1d_m2d_uniform_dL_uniform_merger_rate_in_source_comoving_frame_PE_priors"]
