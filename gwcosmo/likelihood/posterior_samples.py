"""
LALinference posterior samples class and methods
Ignacio Magana, Ankan Sur
"""
import numpy as np
from scipy.stats import gaussian_kde
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import bilby
import h5py
from gwcosmo.likelihood.skymap import ra_dec_from_ipix
from ..prior.priors import distance_distribution
import json
import healpy as hp
import copy
import sys
import importlib.util
import pesummary
from pesummary.io import read
from bilby.core.prior.analytical import *
from bilby.gw.prior import *

from scipy.interpolate import RegularGridInterpolator

class m1d_m2d_uniform_dL_square_PE_priors(object):
    """
    This is the class handling the default PE priors, being uniform in pi(m1d, m2d) and \propto dL^2 for the luminosity distance

    Reminder: 
    For other PE priors, the user must create another file user_prior.py but 
    the name of the class must always be "PE_priors" with a member function 
    called def get_prior(self,m1d,m2d,dL) that returns a floating value (or an array)    
    """
    
    def __init__(self):
        
        self.name = "m1d_m2d:uniform --- dL:square"

    def get_prior_m1d_m2d_dL(self,m1d,m2d,dL):
        """
        This function returns something proportional to p(m1d,m2d,dL)
        the default case considers a uniform 2D distribution for p(m1d,m2d)
        which is the case in general for actual events
        """
        return dL**2

def check_sampling(prior_dict, samp_vars):
    """
    This function detects if the variables in argument samp_vars are sampled in the prior_dict
    returns a dict with keys=samp_vars and the values are True/False if the variables are sampled or not
    """
    is_sampled = {}
    for sv in samp_vars:
        is_sampled[sv] = False
        if sv in prior_dict.keys():
            sd = prior_dict.sample_subset([sv])
            if sd: # actual sampling
                is_sampled[sv] = True
    return is_sampled
                
class analytic_PE_priors(object):
    """
    This is the class handling the analytic PE priors
    these analytic priors were found in the posteriors file of events, when Bilby has been used to sample them

    Reminder: 
    For other PE priors, the user must create another file user_prior.py but 
    the name of the class must always be "PE_priors" with a member function 
    called def get_prior_m1d_m2d_dL(self,m1d,m2d,dL) that returns a floating value (or an array)

    Example prior file: (CIT:/home/cbc.cosmology/MDC/population_only_MDC_gwsim_2023/Events/event_99/priors.priors)

    mass_1 = Constraint(minimum=1, maximum=500, name='mass_1', latex_label='$m_1$', unit=None)
    mass_2 = Constraint(minimum=1, maximum=500, name='mass_2', latex_label='$m_2$', unit=None)
    mass_ratio = Uniform(minimum=0.01, maximum=1, name='mass_ratio', latex_label='$\\mathcal{M}$', unit=None, boundary=None)
    chirp_mass = Uniform(minimum=14.091282825125148, maximum=56.36513130050059, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)
    luminosity_distance = PowerLaw(alpha=2, minimum=0, maximum=15000, name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None)
    dec = Cosine(minimum=-1.5707963267948966, maximum=1.5707963267948966, name='dec', latex_label='$\\mathrm{DEC}$', unit=None, boundary=None)
    ra = Uniform(minimum=0, maximum=6.283185307179586, name='ra', latex_label='$\\mathrm{RA}$', unit=None, boundary='periodic')
    theta_jn = Sine(minimum=0, maximum=3.141592653589793, name='theta_jn', latex_label='$\\theta_{JN}$', unit=None, boundary=None)
    psi = Uniform(minimum=0, maximum=3.141592653589793, name='psi', latex_label='$\\psi$', unit=None, boundary='periodic')
    phase = Uniform(minimum=0, maximum=6.283185307179586, name='phase', latex_label='$\\phi$', unit=None, boundary='periodic')
    a_1 = 0.0 
    a_2 = 0.0 
    tilt_1 = 0.0 
    tilt_2 = 0.0 
    phi_12 = 0.0 
    phi_jl = 0.0 
    geocent_time = Uniform(minimum=62928.38062513391, maximum=62928.58062513391, name='mass_2', latex_label='$m_2$', unit=None, boundary=None)
    seed = 1988
    """
    
    def __init__(self,analytic_dict):
        
        self.name = "analytic_PE_prior_from_PE_file"
        # analytic_dict must be an objet having a .prob() function, like Bilby prior dicts
        self.prior = analytic_dict
        # determine if the prior is on m1d, m2d or Mc, q
        # we expect the sampling to be done either or (Mc,q) or (m1d,m2d)
        sampling_OK = False
        self.sampling_vars = []
        
        # check if Mc and q are in the keys
        svars = ['chirp_mass','mass_ratio']
        is_sampled = check_sampling(self.prior,svars)
        if is_sampled[svars[0]] and is_sampled[svars[1]]: # then the sampling is done on ['chirp_mass','mass_ratio']
            # check if it's UniformInComponents, i.e. sampling in Mc, q with U(m1d,m2d)
            if 'UniformInComponents' in str(type(self.prior[svars[0]])) and 'UniformInComponents' in str(type(self.prior[svars[1]])):
                print("Sampled vars are Mc and q but setting m1d,m2d,dL prior to dL as it's UniformInComponents for Mc and q.")
                self.get_prior_m1d_m2d_dL = self.get_prior_dL # uniform 2D pdf pi(m1d,m2d)
                sampling_OK = True
            else:
                print("The sampling is on Mc, q with no 'UniformInComponents' option. Adding the jacobian.")
                self.get_prior_m1d_m2d_dL = self.get_prior_actual_Mc_q_dL_to_m1d_m2d_dL
                sampling_OK = True
            self.sampling_vars = svars
        if not sampling_OK: # the true sampling must be on m1d, m2d as we did not succeed to set it for Mc, q
            # double check that the sampling is on m1d, m2d
            svars = ['mass_1','mass_2']
            is_sampled = check_sampling(self.prior,svars)
            if is_sampled[svars[0]] and is_sampled[svars[1]]: # then the sampling is done on ['mass_1','mass_2']
                print("Setting m1d, m2d, dL prior to the analytic prior of PE file.")
                self.get_prior_m1d_m2d_dL = self.get_prior_actual_m1d_m2d_dL
                sampling_OK = True
            else:
                raise ValueError("Weird... no correct sampling on ['mass_1','mass_2'] or ['chirp_mass','mass_ratio'] in the dict. Exiting.")
            self.sampling_vars = svars
                
    def get_prior_actual_Mc_q_dL_to_m1d_m2d_dL(self,m1d,m2d,dL):
        """
        This function returns something proportional to p(m1d,m2d,dL)
        it must be used when the PE posterior were obtained after a sampling on Mc, q
        the function is called with args m1d, m2d, dL so that we must first compute Mc, q (det frame),
        compute the PE prior probability on Mc, q and convert it into the PE prior probability on m1d, m2d and add the jacobian
        """
        mcdet = (m1d*m2d)**(3./5)/(m1d+m2d)**(1./5) # chirp mass det frame
        q = m2d/m1d
        nvals = len(q)
        prior_mcdet_q_dL = np.zeros(nvals)
        for i in range(nvals): # did not find how to compute all probs at once!
            prior_mcdet_q_dL[i] = self.prior.prob({'chirp_mass':mcdet[i],'mass_ratio':q[i],'luminosity_distance':dL[i]})
        jacobian = mcdet/m1d**2
        return prior_mcdet_q_dL*jacobian

    def get_prior_actual_m1d_m2d_dL(self,m1d,m2d,dL):
        """
        This function returns something proportional to p(m1d,m2d,dL)
        the analytic case returns the probability p(m1d,m2d,dL) after marginalizing over all other parameters
        """
        return self.prior.prob({'mass_1':m1d,'mass_2':m2d,'luminosity_distance':dL})

    def get_prior_dL(self,m1d,m2d,dL):
        """
        This function returns something proportional to p(dL) assuming p(m1d,m2d) is uniform
        """
        return self.prior['luminosity_distance'].prob(dL)
    



class load_posterior_samples(object):
    """
    Posterior samples class and methods.

    Parameters
    ----------
    posterior_samples : dict for each GW event, the dict can contain the following fields:
    "posterior_file_path" => path to the PE posteriors, h5, hdf, json, dat...
    "samples_field" => name of the waveform approximant (CO1:Mixed, C01:IMRPhenomXPHM...)
    "PEprior_file_path" => path to the PE prior file (optional)
    "skymap_path" => path to the GW event skymap (fits file)
    """
    
    def __init__(self,posterior_samples):

        self.PE_file_key = "posterior_file_path"
        self.samples_field_key = "samples_field"
        self.PE_prior_file_key = "PEprior_file_path"
        self.PE_prior_class_name = "PE_priors"
        self.PE_skymap_file_key = "skymap_path"
        # we will add some new fields to the 'posterior_samples' dict
        self.analysis_type = "analysis_type"
        self.sampling_vars = "sampling_variables"
        self.multi_analysis = "multi"
        self.single_analysis = "single"
        self.approximants_available = "approximants_available"
        self.approximant_selected = "approximant_selected"
        self.has_analytic_priors = "has_analytic_priors"

        # define the default approximant to consider if the user did not specify one
        # the approximants will be search for in this order
        self.default_approximants = ['PublicationSamples',
                                     'C01:Mixed',
                                     'C01:PhenomPNRT-HS', 
                                     'C01:NRSur7dq4',
                                     'C01:IMRPhenomPv3HM',
                                     'C01:IMRPhenomPv2',
                                     'C01:IMRPhenomD',
                                     'C01:IMRPhenomPv2_NRTidal:LowSpin', 
                                     'C01:IMRPhenomPv2_NRTidal:HighSpin']
                
        self.posterior_samples = posterior_samples
        print("\n\nTreating event: {}".format(posterior_samples))
        # deal with the PE priors:
        if self.PE_prior_file_key in self.posterior_samples.keys():
            print("PE prior file provided: {}".format(self.posterior_samples[self.PE_prior_file_key]))
            try:
                spec = importlib.util.spec_from_file_location(self.PE_prior_class_name,self.posterior_samples[self.PE_prior_file_key])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.pe_priors_object = module.PE_priors()
                print("PE priors loaded: prior name = {}".format(self.pe_priors_object.name))
            except:
                raise ValueError("Could not find class named \"PE_priors\" in file {}. Exiting.".format(self.posterior_samples[self.PE_prior_file_key]))
        else:
            print("NO PE prior file user-provided.")
            self.pe_priors_object = None # the object self.pe_priors_object will be initialized later

        # deal with the field for the PE analysis (C01:Mixed etc.):
        self.field = None
        if self.samples_field_key in self.posterior_samples.keys(): # i.e. if the user specified the 'C01:...' waveform model
            if self.posterior_samples[self.samples_field_key].lower() != "none":
                self.field = self.posterior_samples[self.samples_field_key]

        # deal with the skymap:
        # for dark_siren: the key 'skymap_path' must exists in the dictionary => already check for that in bin/gwcosmo_dark...
        # but the bright siren case can have no skymap
        if self.PE_skymap_file_key in self.posterior_samples.keys():
            self.skymap_path = self.posterior_samples[self.PE_skymap_file_key]
                
        self.load_posterior_samples()


    def extract_data_and_PEprior(self,posterior_file):
        '''
        This function is called when dealing with .h5 files and O4-.hdf5 files.
        '''
        print(posterior_file)
        try:
            pes = read(posterior_file,package="core")
            print("Posterior file {} correctly read with pesummary.".format(posterior_file))
        except:
            raise ValueError("Could not read posterior file {} with pesummary. Check the file. Exiting.".format(posterior_file))

        if isinstance(pes.samples_dict,pesummary.utils.samples_dict.MultiAnalysisSamplesDict): # check if we have a multianalysis file
            self.posterior_samples[self.analysis_type] = self.multi_analysis
            self.posterior_samples[self.approximants_available] = list(pes.samples_dict.keys())
            if self.field is None:
                for approximant in self.default_approximants:
                    try:
                        data = pes.samples_dict[approximant]
                        print("No waveform field provided -> setting model: "+approximant)
                        self.field = approximant # record the approximant
                        break
                    except KeyError:
                        continue
                self.posterior_samples[self.approximant_selected] = self.field
            else:
                if self.field in  pes.samples_dict.keys(): # check if required key exists
                    data = pes.samples_dict[self.field]
                else:
                    raise ValueError("The required analysis key {} does not exist in file. Available keys are: {}. Exiting."
                          .format(self.field,pes.samples_dict.keys()))

        else: # single analysis in file
            self.posterior_samples[self.analysis_type] = self.single_analysis
            self.posterior_samples[self.approximant_selected] = None
            data = pes.samples_dict
            
        self.distance = data['luminosity_distance']
        self.ra = data['ra']
        self.dec = data['dec']
        self.mass_1 = data['mass_1']
        self.mass_2 = data['mass_2']
        self.nsamples = len(self.distance)
        show_keys = ['mass_1','mass_2','chirp_mass','mass_ratio','luminosity_distance']
        print("Searching for sample field: {}".format(self.field))
        # deal with PE prior values for each sample
        if self.pe_priors_object is None: # no prior file provided by the user so the prior object may be stored in the posterior file
            status, subdict, pdicts = get_priors(pes) # try to find a prior in the posterior file
            if status:
                self.posterior_samples[self.has_analytic_priors] = True
                print("Analytic priors found in file...")                    
                #print(pdicts)
                non_empty_dicts_keys = []
                if subdict: # this is the multianalysis case, {'C01:Mixed':{}, 'C01:other':{}...}
                    print("Multianalysis case.")
                    for k in pdicts.keys():
                        if len(pdicts[k]) == 0: # this case should not happen, just in case
                            print("Empty dict")
                        else:
                            non_empty_dicts_keys.append(k) # record existing dict keys
                    if len(non_empty_dicts_keys) == 0:
                        print("Problem: no dict with active keys available! Using m1d_m2d_uniform_dL_square PE priors <=== CHECK IF THIS IS OK FOR YOUR ANALYSIS.")
                        self.pe_priors_object = m1d_m2d_uniform_dL_square_PE_priors()
                    elif len(non_empty_dicts_keys) == 1:
                        ldict = pdicts[non_empty_dicts_keys[0]]
                        self.pe_priors_object = analytic_PE_priors(ldict)
                        print("Found a single analytic prior dict with field: {} , will use it for the analysis <=== CHECK IF THIS IS OK FOR YOUR ANALYSIS.".format(self.field))
                        print("Dict characteristics for masses and distance:")
                        for k in show_keys:
                            if k in ldict.keys():
                                print("\t {}:{}".format(k,ldict[k]))
                    else:
                        print("WARNING!!!!!!!!! Several prior dicts are available.")
                        print("Required sample field: {}".format(self.field))
                        print("Available keys: {}".format(non_empty_dicts_keys.keys()))
                        if self.field in non_empty_dicts_keys.keys():
                            print("Found analytic prior dict with same field name: {}, using this one for the analysis".format(self.field))
                            self.pe_priors_object = analytic_PE_priors(pdicts[self.field])
                        else:
                            print("No analytic prior dict with field name {}. Using m1d_m2d_uniform_dL_square PE priors <=== CHECK IF THIS IS OK FOR YOUR ANALYSIS."
                                  .format(self.field))
                            self.pe_priors_object = m1d_m2d_uniform_dL_square_PE_priors()
                else: # this is the single analysis case {'mass_1':....}
                    print("Single analysis case.")
                    self.pe_priors_object = analytic_PE_priors(pdicts)
                    print("Found a single analytic prior dict, will use it for the analysis <=== CHECK IF THIS IS OK FOR YOUR ANALYSIS!.")
                    print("Dict characteristics for masses and distance:")
                    for k in show_keys:
                        if k in pdicts.keys():
                            print("\t {}:{}".format(k,pdicts[k]))
                self.posterior_samples[self.sampling_vars] = self.pe_priors_object.sampling_vars
            else:
                print("No analytic priors in file, using U(m1d, m2d) and p(dL) \propto dL^2")
                self.posterior_samples[self.has_analytic_priors] = False
                self.pe_priors_object = m1d_m2d_uniform_dL_square_PE_priors()

                    
        
    def load_posterior_samples(self):
        """
        Method to handle different types of posterior samples file formats.
        Currently it supports .dat (LALinference), .hdf5 (GWTC-1),
        .h5 (PESummary) and .hdf (pycbcinference) formats.
        """
        posterior_file = self.posterior_samples[self.PE_file_key]
        if posterior_file[-3:] == 'dat':
            samples = np.genfromtxt(self.posterior_samples, names = True)
           
            self.distance = np.array([var for var in samples['luminosity_distance']])
            self.ra =  np.array([var for var in samples['ra']])
            self.dec =  np.array([var for var in samples['dec']])
            self.mass_1 =  np.array([var for var in samples['mass_1']])
            self.mass_2 =  np.array([var for var in samples['mass_2']])
            self.nsamples = len(self.distance)

        if posterior_file[-4:] == 'hdf5':
            if posterior_file[-11:] == 'GWTC-1.hdf5':
                if posterior_file[-20:] == 'GW170817_GWTC-1.hdf5':
                    dataset_name = 'IMRPhenomPv2NRT_lowSpin_posterior'
                else:
                    dataset_name = 'IMRPhenomPv2_posterior'
                file = h5py.File(posterior_file,'r')
                data = file[dataset_name]
                self.distance = data['luminosity_distance_Mpc']
                self.ra = data['right_ascension']
                self.dec = data['declination']
                self.mass_1 = data['m1_detector_frame_Msun']
                self.mass_2 = data['m2_detector_frame_Msun']
                self.nsamples = len(self.distance)
                file.close()
            else: # O4-like events, hdf5 files
                self.extract_data_and_PEprior(posterior_file)


        if posterior_file.endswith('.json'):
            with open(posterior_file) as f:
                data = json.load(f)

            PE_struct=data['posterior_samples'][self.field]

            m1_ind=PE_struct['parameter_names'].index('mass_1')
            m2_ind=PE_struct['parameter_names'].index('mass_2')
            dl_ind=PE_struct['parameter_names'].index('luminosity_distance')
            ra_ind=PE_struct['parameter_names'].index('ra')
            dec_ind=PE_struct['parameter_names'].index('dec')
                        
            nsamp=len(PE_struct['samples'])
            
            self.distance = np.array(PE_struct['samples'])[:,dl_ind].reshape(-1)
            self.ra = np.array(PE_struct['samples'])[:,ra_ind].reshape(-1)
            self.dec = np.array(PE_struct['samples'])[:,dec_ind].reshape(-1)
            self.mass_1 = np.array(PE_struct['samples'])[:,m1_ind].reshape(-1)
            self.mass_2 = np.array(PE_struct['samples'])[:,m2_ind].reshape(-1)
            self.nsamples = len(self.distance)

        if posterior_file[-2:] == 'h5':
            self.extract_data_and_PEprior(posterior_file)

        if posterior_file == 'hdf':
            file = h5py.File(posterior_file,'r')
            self.distance = file['samples/distance'][:]
            self.ra = file['samples/ra'][:]
            self.dec = file['samples/dec'][:]
            self.mass_1 = file['samples/mass_1'][:]
            self.mass_2 = file['samples/mass_2'][:]
            self.nsamples = len(self.distance)
            file.close()

        if self.pe_priors_object is None:
            # case where no prior has been found: neither user-provided nor in the posterior file
            raise ValueError("WARNING !!!!!!!!!!! No PE-prior has been set. Cannot run the analysis.")
        
        print("Computing PE prior(m1d,m2d,dL) using object: {} with name: {}"
              .format(self.pe_priors_object,self.pe_priors_object.name))
        self.pe_priors = self.pe_priors_object.get_prior_m1d_m2d_dL(self.mass_1,self.mass_2,self.distance) # we compute all PE priors values -> pi(m1d,m2d,dL)
        print("PE priors values for posterior samples are computed.")

            
    def marginalized_sky(self):
        """
        Computes the marginalized sky localization posterior KDE.
        """
        return gaussian_kde(np.vstack((self.ra, self.dec)))

def get_priors(pes):
    '''
    This function searches for prior dictionnaries in a posterior file (.hdf5 or .h5).
    '''
    status = False
    subdict = False
    pdicts = {}
    print("labels in posterior samples file: {}".format(pes.labels))
    if pes.priors == None:
        raise ValueError("No PE prior object in posterior samples file. Cannot perform the analysis.")

    print("prior keys in posterior samples file: {}".format(pes.priors.keys()))
    dict_count = 0
    if isinstance(pes.priors,dict):
        if 'analytic' in pes.priors.keys():
            status = True
            if all('C01' in key for key in pes.priors['analytic'].keys()): # it's a multianalysis prior dict
                print("multianalysis file: keys of priors['analytic'] = ",pes.priors['analytic'].keys())
                subdict = True
                print("\tsubdict!")
                for key in pes.priors['analytic'].keys():
                    print("\tChecking key {}".format(key))                    
                    if len(pes.priors['analytic'][key].keys())>0: # it's a non-empty dict
                        dict_count += 1
                        ndict = copy.deepcopy(pes.priors['analytic'][key])
                        for k in pes.priors['analytic'][key].keys():
                            #print("Getting key {}, {}".format(k,pes.priors['analytic'][key][k]))
                            if isinstance(pes.priors['analytic'][key][k],str): # some old true GW events have priors written in terms of str
                                # take care of the luminosity distance prior
                                if 'luminosity_distance' in k:
                                    dLprior, PEcosmo = get_dL_prior(pes.priors['analytic'][key][k])
                                    ndict[k] = dLprior
                                else:
                                    ndict[k] = eval(pes.priors['analytic'][key][k])
                            else: # get the object directly, no need to convert str into object
                                ndict[k] = pes.priors['analytic'][key][k]
                        pdicts[key] = BBHPriorDict() # deal with the NSBH of BNS cases!
                        allkeys = list(pdicts[key].keys())
                        for k in allkeys:
                            pdicts[key].pop(k) # remove all keys, in order to keep only those of the PE file
                        pdicts[key].from_dictionary(ndict)
                    else:
                        print("\tKey {} is an empty dict. Ignoring.".format(key))
                        
            else: # it's a single analysis h5 file
                print("Single analysis posterior file. Getting analytic prior data.")
                ndict = copy.deepcopy(pes.priors['analytic'])
                print(ndict['luminosity_distance'])
                dLprior, PEcosmo = get_dL_prior(str(ndict['luminosity_distance']))
                #print("main::dLprior: {}".format(dLprior))
                #print("main::dLprior: type = {}".format(type(dLprior)))
                #print("main::PEcosmo: {}".format(PEcosmo))
                #print("main::PEcosmo: {}".format(type(PEcosmo)))
                ndict['luminosity_distance'] = dLprior
                pdicts = bilby.gw.prior.BBHPriorDict() # just to have the correct type for pdicts; have to deal with the NSBH of BNS cases!
                allkeys = list(pdicts.keys())
                for k in allkeys:
                    pdicts.pop(k) # remove all keys, in order to keep only those of the PE file
                pdicts.from_dictionary(ndict)
                
                
    return status, subdict, pdicts


def get_dL_prior(dl_prior):
    """
    input: dl_prior is the name of the luminosity prior, it's a string
    it is needed when the dL prior is something similar to
    bilby.gw.prior.UniformSourceFrame(minimum=100.0, maximum=5000.0,
    cosmology=FlatLambdaCDM(H0=67.74 km / (Mpc s), Om0=0.3075, Tcmb0=2.7255 K,
    Neff=3.046, m_nu=[0.   0.   0.06] eV, Ob0=0.0486),
    name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None)
    as the FlatLambdaCDM part must be extracted and modified to recreate this astropy object
    once it is done, we can recreate the dL prior

    output: returns the name of the dL prior and the astropy cosmo object, needed to define the dL prior
    """
    keep = ""
    #print("func: dl_prior type= {}".format(type(dl_prior)))
    thestr = copy.deepcopy(dl_prior)
    cosmostr = "cosmology="
    par_count = 0
    first = True
    fc = thestr.find(cosmostr)
    if fc == -1: # it's not a prior using an astropy object, no need to go further
        print("\tdL prior is not an astropy object, no special treatment.")
        return eval(dl_prior),None
    
    for ic, c in enumerate(thestr[fc+len(cosmostr):]):
        keep += c
        if c == '(': # first parenthesis
            par_count += 1
            first = False
        if c == ')':
            par_count -= 1
        if not first and par_count == 0:
            break
        if c == '\'' or c == '\"': # check if the astropy object is using an alias for cosmology, such as 'Planck15'
            #print("found quote! {},{}".format(ic,c))
            fc2 = thestr[fc+len(cosmostr)+ic+1:].find(c) # find the closing ' or "
            keep = thestr[fc+len(cosmostr)+ic:fc+len(cosmostr)+fc2+2]
            break
    cmod = copy.deepcopy(keep)
    cmod = cmod.replace(" km / (Mpc s)","")
    cmod = cmod.replace(" K","")
    cmod = cmod.replace(" eV","")
    cmod = cmod.replace("[0.   0.   0.06]","[0.,   0.,   0.06]")
    # Fix for astropy >= 6.0
    cmod = cmod.replace("<Quantity","")
    cmod = cmod.replace(">","")
    PE_cosmo = eval(cmod)
    #print("cmod: {}".format(cmod))
    #print("PE_cosmo: {}".format(PE_cosmo))
    #print("keep: {}".format(keep))
    PE_dl = thestr.replace(keep,"PE_cosmo")
    #print("PEDL: {}".format(PE_dl))
    #print("func: PE_dl and PE_cosmo types= {}, {}".format(type(dl_prior),type(PE_cosmo)))
    #print("func: obj and PE_cosmo types= {}, {}".format(type(PE_dl),type(PE_cosmo)))
    #print(PE_dl)
    PE_dl = eval(str(PE_dl))
    return PE_dl, PE_cosmo


class reweight_posterior_samples(object):
    """
    Posterior samples class and methods.

    Parameters
    ----------
    cosmo : Fast cosmology class
    mass_priors: Fast mass_distributions class
    """
    
    def __init__(self,cosmo,mass_priors):
        self.cosmo = cosmo
        # Prior distribution used in this work
        self.source_frame_mass_prior = mass_priors

    def jacobian(self,z):
        """
        (1+z)^2 * ddL/dz
        """
        return np.power(1+z,2)*self.cosmo.ddgw_dz(z)    
        
    def compute_source_frame_samples(self, GW_distance, det_mass_1, det_mass_2):
        """
        Posterior samples class and methods.

        Parameters
        ----------
        GW_distance: GW distance samples in Mpc
        det_mass_1, det_mass_2 : detector frame mass samples in Msolar
        H0 : Hubble constant value in kms-1Mpc-1
        """
        redshift = self.cosmo.z_dgw(GW_distance)

        mass_1_source = det_mass_1/(1+redshift)
        mass_2_source = det_mass_2/(1+redshift)
        return redshift, mass_1_source, mass_2_source

    def get_kde(self, data, weights):
        # deal first with the weights
        weights, norm, neff = self.check_weights(weights)
        if norm != 0:
            try:
                kde = gaussian_kde(data, weights=weights)
            except:
                print("KDE problem! create a default KDE with norm=0")
                print("norm: {} -> 0, neff: {}".format(norm,neff))
                norm = 0
                kde = gaussian_kde(data)
        else:
            kde = gaussian_kde(data)
            
        return kde, norm
                    
    def ignore_weights(self, weights):

        weights = np.ones(len(weights))
        norm = 0
        return weights, norm

    def check_weights(self, weights):
        """
        Check the weights values to prevent gaussian_kde crash when Neff <= 1,
        where Neff is an internal variable of gaussian_kde
        defined by Neff = sum(weights)^2/sum(weights^2)
        careful, cases with Neff = 1+2e-16 = 1.0000000000000002
        have been seen and give crash: set Neff limit to >= 2
        """
        neff = 0
        if np.isclose(max(weights),0,atol=1e-50):
            weights, norm = self.ignore_weights(weights)
        else:
            neff = sum(weights)**2/sum(weights**2)
            if neff<2:
                weights, norm = self.ignore_weights(weights)
            else:
                norm = np.sum(weights)/len(weights)
        return weights, norm, neff
    
    def marginalized_redshift_reweight(self, redshift, mass_1_source, mass_2_source, PEpriors_detframe):
        """
        Computes the marginalized distance posterior KDE.
        it uses the prior object PEpriors_detframe that provides p_PE(m1det,m2det,dL) used in the PE step (detector frame)
        """
        # Re-weight
        PEpriors_source_frame = PEpriors_detframe * self.jacobian(redshift) # this is pPE(m1d,m2d,dL) (1+z)^2 |ddL/dz|
        weights = self.source_frame_mass_prior.joint_prob(mass_1_source,mass_2_source)/PEpriors_source_frame

        return self.get_kde(redshift,weights)

    def marginalized_redshift(self, redshift):
        """
        Computes the marginalized distance posterior KDE.
        """
        # remove dgw^2 prior and include dz/ddgw jacobian
        weights = 1/(self.cosmo.ddgw_dz(redshift)*self.cosmo.dgw(redshift)**2)
        return self.get_kde(redshift,weights)


class make_pixel_px_function(object):
    """
    Identify the posterior samples which lie within some angular radius
    (depends on skymap pixel size) of the centre of each pixel
    """
    
    def __init__(self, samples, skymap, npixels=30, thresh=0.999):
        """
        Parameters
        ----------
        samples : posterior_samples object
            The GW samples
        skymap : object
            The GW skymap
        npixels : int, optional
            The minimum number of pixels desired to cover given sky area of
            the GW event (default=30)
        thresh : float, optional
            The sky area threshold (default=0.999)
        """
        
        self.skymap = skymap
        self.samples = samples
        nside=1
        indices,prob = skymap.above_percentile(thresh, nside=nside)
    
        while len(indices) < npixels:
            nside = nside*2
            indices,prob = skymap.above_percentile(thresh, nside=nside)
        
        self.nside = nside
        print('{} pixels to cover the {}% sky area (nside={})'.format(len(indices),thresh*100,nside))
        
        dicts = {}
        for i,idx in enumerate(indices):
            dicts[idx] = prob[i]
        self.indices = indices
        self.prob = dicts # dictionary - given a pixel index, returns skymap prob

        
    def identify_samples(self, idx, minsamps=100):
        """
        Find the samples required 
        
        Parameters
        ----------
        idx : int
            The pixel index
        minsamps : int, optional
            The threshold number of samples to reach per pixel
            
        Return
        ------
        sel : array of ints
            The indices of posterior samples for pixel idx
        """
    
        racent,deccent = ra_dec_from_ipix(self.nside, idx, nest=self.skymap.nested)
    
        separations = angular_sep(racent,deccent,self.samples.ra,self.samples.dec)
        sep = hp.pixelfunc.max_pixrad(self.nside)/2. # choose initial separation
        step = sep/2. # choose step size for increasing radius
        
        sel = np.where(separations<sep)[0] # find all the samples within the angular radius sep from the pixel centre
        nsamps = len(sel)
        while nsamps < minsamps:
            sep += step
            sel = np.where(separations<sep)[0]
            nsamps = len(sel)
            if sep > np.pi:
                raise ValueError("Problem with the number of posterior samples.")
        print('angular radius: {} radians, No. samples: {}'.format(sep,len(sel)))
            
        return sel

        

def identify_samples_from_posterior(ra_los, dec_los, ra, dec, nsamps=1000):
    """
    Find the angular separation between all posterior samples and a specific
    LOS. Return the indices of the nsamps closest samples, as well as the 
    maxiumum separation of those samples.
    
    Parameters
    ----------
    ra_los : float
        right ascension of the line-of-sight (radians)
    dec_los : float
        declination of the line-of-sight (radians)
    ra : array of floats
        right ascensions of a set of samples (radians)
    dec : array of floats
        declinations of a set of samples (radians)
    nsamps : int, optional
        The number of samples to select (default=1000)
        
    Return
    ------
    index : array of ints
        The indices of of the nsamps samples
    ang_rad_max : float
        The maximum angular radius between the selected samples and the LOS
    """

    separations = angular_sep(ra_los,dec_los,ra,dec)
    sep_argsort = np.argsort(separations)
    index = sep_argsort [:nsamps]
    ang_rad_max = separations [nsamps-1]

    return index, ang_rad_max

        
    
def angular_sep(ra1,dec1,ra2,dec2):
    """Find the angular separation between two points, (ra1,dec1)
    and (ra2,dec2), in radians."""
    
    cos_angle = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    angle = np.arccos(cos_angle)
    return angle
