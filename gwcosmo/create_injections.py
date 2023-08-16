"""
Injections module

Benoit Revenu
Christos Karathanasis
"""

import bilby as bl
import numpy as np
import pickle 
from tqdm import tqdm
import lalsimulation as lalsim
import os
import logging
from gwcosmo.injections import injections_at_detector, injections_at_source
from bilby.gw import utils as gwutils
import multiprocess as mp
import sys
import time
import copy
import tempfile
from gwcosmo.utilities.cosmology import standard_cosmology
import random
import string
import scipy.integrate as integrate
import h5py

class Counter(object):
    '''
    This class defines a counter in shared memory for usage in the children processes.
    In the current version of the code, where now each child process has its dedicated number of injections
    to compute, the counter object is used only for information; it's not part anymore of the process
    of controlling the number of injections to compute.
    The lock mechanism avoid problems when 2 processes want access at the same time to the same variable.

    Parameters
    ----------
    initval: int
    this is the initial value of the counter (default = -1)

    maxval: int
    this is the maximum value of the counter (default = -1)
    '''

    def __init__(self, initval = -1, maxval = -1):
        self.val = mp.Value('i',initval)
        self.maxval = mp.Value('i',maxval)
        self.lock = mp.Lock()

    def set_value(self, value):
        with self.lock:
            self.val = mp.Value('i',value)
            
    def increment(self,to_add=1):
        with self.lock:
            self.val.value += to_add
            return self.val.value

    def get_value(self):
        with self.lock:
            return self.val.value

    def max_value(self):
        with self.lock:
            return self.maxval.value

def get_dLmax_params(snr):
    '''
    This function returns the values of the parameters of the function dLmax(m1) for snrs 9, 10, 11, 12.
    Parameters are: a[0], a[1], a[2], a[3] with dLmax(m1) = (a[0]+a[1]*m1)*exp(-(m1+a[2])**2/(2*a[3]**2))
    '''
    dLmax_m1 = {}
    if snr == 9:
        dLmax_m1['O1'] = [ 84.31804818, 192.53468201, 182.93373141, 299.50877694]
        dLmax_m1['O2'] = [147.91817378, 175.75506129,  35.93831714, 263.5630744 ]
        dLmax_m1['O3'] = [213.84817574, 244.51138966,  69.23433534, 271.4468494 ]
        dLmax_m1['O4high'] = [436.18835656, 366.88443749,  93.34706107, 343.16794722]
        dLmax_m1['O4low'] = [489.22000474, 834.02779856, 633.62937608, 496.61871247]
    elif snr == 10:
        dLmax_m1['O1'] = [ 86.99423443, 159.58449504, 178.84598956, 300.5293686 ]
        dLmax_m1['O2'] = [128.05686148, 154.41689787,  64.73827621, 280.0737362 ]
        dLmax_m1['O3'] = [160.81115278, 234.71142174, 130.4708518,  294.36698715]
        dLmax_m1['O4high'] = [359.47412722, 313.11598171,  70.02679103, 325.7867842 ]
        dLmax_m1['O4low'] = [353.02830031, 521.1612163,  471.41373238, 457.18316462]
    elif snr == 11:
        dLmax_m1['O1'] = [ 80.00135878, 129.34914561, 140.42625102, 286.23857801]
        dLmax_m1['O2'] = [114.27477259, 130.78112051,  48.1232669,  277.0573754 ]
        dLmax_m1['O3'] = [137.31083208, 204.85159489, 125.53551322, 291.99014915]
        dLmax_m1['O4high'] = [310.99831975, 273.55408815,  73.49104502, 328.08351715]
        dLmax_m1['O4low'] = [245.81416194, 391.50469287, 373.16419088, 424.40083527]
    elif snr == 12:
        dLmax_m1['O1'] = [ 67.88545626, 109.03198868, 111.82572526, 274.54170286]
        dLmax_m1['O2'] = [102.23016472, 113.16162086,  24.72815689, 267.31760986]
        dLmax_m1['O3'] = [119.23589567, 182.81133974, 129.51557969, 294.25833901]
        dLmax_m1['O4high'] = [274.3325264,  268.83221473, 165.03056849, 371.29597527]
        dLmax_m1['O4low'] = [180.99809457, 273.19457205, 200.29022794, 348.78256775]
    else:
        print("dLmax(m1) parameters not available for snr {}.".format(snr))
    return dLmax_m1

        
class Create_injections(object):
    '''
    This is the class dealing with injections in the detector frame only. The source frame case will be considered later.

    Parameters
    ----------
    
    asd_path: str
    path of the strain files (2 columns: frequency - amplitude), this is NOT the psd

    psd_opts: str
    is either 'low' or 'high' corresponding to the pessimistic or optimistic (respectively) sentivity for O4

    days_of_run: dict
    number of days of run for O1, O2, O3
    default = {'O3':330,'O2':268,'O1':129}

    duty_factors: dict
    duty factors for O1, O2, O3
    default = {'O4':{'H1':0.75,'L1':0.75,'V1':0.75},
               'O3':{'H1':0.7459,'L1':0.7698,'V1':0.7597},
               'O2':{'H1':0.653,'L1':0.618,'V1':0.0777},
               'O1':{'H1':0.646,'L1':0.574,'V1':-1}}
    in https://arxiv.org/pdf/1304.0670.pdf
    number of days of O1: 12 September 2015 - 19 January 2016 with duty factors 64.6% (H) and 57.4% (L), i.e. 129 days
    number of days of O2: 30 November 2016 - 25 August 2017 with duty factors 65.3% (H) and 61.8% (L), i.e. 268 days
    Virgo/O2 was in data taking mode during 24.5 days (2017-08-01:10:00:00 - 2017-08-25:22:00:00) with a duty cycle of 85.08%.
    So that during O2, Virgo must be considered online with a probability of 85.08% * 24.5 / 268 = 7.77%
    Reference for Virgo/O2: https://wiki.virgo-gw.eu/DataAnalysis/DetChar/DetCharO2Summary and https://logbook.virgo-gw.eu/virgo/?r=39301
    for O3, number of days = 330 days (28519200/86400) values are here: https://wiki.ligo.org/Operations/O3OfficialNumbers

    frame: str
    reference frame for masses, either 'detectors_frame' or 'source_frame'
    default = 'detectors_frame'

    nprocs: int
    this is the number of child processes started by the main process to compute injections

    tmp_to_dict: str
    name of a temporary file
    the temporary file will be converted into a dict
    no injections are computed if activated

    tmp_to_stdout: str
    name of a temporary file
    the contents of the temporary file is dumped on stdout

    tmpfile_to_dict: str
    name of a file containing the list of temporary files (full path) containing injections
    this is used when using a large number of temporary files coming from separate computing centers

    approx: str
    model used for the theoretical signal in injections
    default: 'IMRPhenomPv2'

    priors: str
    this is the path to a file containing a prior for Bilby
    this prior file is sent to the function bilby.core.prior.dict.PriorDict
    default: None i.e. we create the prior in the code

    fmin: float
    this is the value of the frequency in Hz used in gwutils.calculate_time_to_merger(frequency = fmin,...)
    and also in waveform_arguments = dict(.. reference_frequency = fmin)
    default: 20 (Hz)

    sampling frequency: float
    sampling frequency of the signal in Hz
    this value is used in bilby.gw.waveform_generator.WaveformGenerator(... sampling_frequency = self.fsamp...)
    default: 4096 Hz

    dLmax_depends_on_m1: bool
    activate or not the dependence of dLmax on m1
    if not activated, dLmax is constant with m1, with a constant value depending of O1, O2, O3
    default: True (i.e. use a dLmax(m1) with SNR_thres=9)

    SNR_thres: float
    the minimum value of the SNR for detected injections
    default: 9
    '''
    
    def __init__(self,detectors=['H1', 'L1', 'V1'],
                 psd=None,
                 psd_opts=None,
                 asd_path='./',
                 days_of_runs=None,
                 duty_factors=None,
                 frame='detectors_frame',
                 nprocs=1,
                 tmp_to_dict=None,
                 tmp_to_stdout=None,
                 tmpfile_to_dict=None,
                 approx='IMRPhenomPv2',
                 priors=None,
                 priors_limits=None,
                 fmin=20,
                 sampling_frequency=4096,
                 dLmax_depends_on_m1=False,
                 dLmax_m1_params=None,
                 SNR_thres=9,
                 dump_inj_period=200):


        ###########################################################################
        # define strings and data structure for the communication with the
        # communication process (write_injection function, using a queue)
        self.kill_msg = "CaN YoU terMinate YOURSELF PLEASE?"
        self.header_msg = "header"
        self.footer_msg = "footer"
        self.injection_msg = "NewInjection"
        # we define the list sent by the child injection processes to the write_injections process
        # a single item in the list: the injections
        self.empty = [0]
        ###########################################################################


        logging.disable(logging.INFO)
        if priors != None:
            print("The current version of the injection code does not treat priors from file. Exiting.")
            sys.exit()
                
        if tmp_to_dict != None:
            # convert a temporary injections file into a dict file
            print("Converting temp file {}".format(tmp_to_dict))
            if os.path.isfile(tmp_to_dict):
                self.dict_detected, ninj, toskip = convert_temp_file_to_dict(tmp_to_dict,
                                                                             self.header_msg,
                                                                             self.footer_msg,
                                                                             self.injection_msg)
                if not toskip:
                    dn = os.path.dirname(tmp_to_dict)
                    if dn == '': dn = "./"
                    self.output_file_dict = dn+"/dict_"+os.path.basename(tmp_to_dict)
                    write_object_to_file(self.dict_detected,self.output_file_dict)
                    print("Dict written to file {}".format(self.output_file_dict))
                else:
                    print("Tmp file {} skipped".format(tmp_to_dict))
            else:
                print("Could not open file {}. Exiting.".format(list_to_dict))
            sys.exit()

        if tmpfile_to_dict != None:
            # merge many temporary injections file into a single dict, used in the case of injections from different clusters
            print("Merging tmp files into a single dict.")
            if os.path.isfile(tmpfile_to_dict):
                self.dict_detected = merge_tmpfile(tmpfile_to_dict,
                                                   self.header_msg,
                                                   self.footer_msg,
                                                   self.injection_msg)
            else:
                print("Could not open file {}. Exiting.".format(tmpfile_to_dict))
            sys.exit()
            
        if tmp_to_stdout != None:
            # display on stdout the contents of a temporary injections file
            print("Displaying tmp file {} on stdout".format(tmp_to_stdout))            
            if os.path.isfile(tmp_to_stdout):
                try:
                    fd = open(tmp_to_stdout,"rb")
                except Exception as e:
                    print(e)
                    print("Could not open file {}. Exiting.".format(tmp_to_stdout))
                    sys.exit()
                try:
                    evt_list = pickle.load(fd)
                    for i in evt_list:
                        print(i)
                except Exception as e:
                    print(e)
                    print("Could not unpickle the file {}, file could be damaged, try to redownload it?\nSkipping file...".format(tmp_to_stdout))                
            else:
                print("Invalid file {}. Exiting.".format(tmp_to_stdout))
            sys.exit()
        
        if days_of_runs == None:
            # use the same values as the default ones in gwcosmo/utilities/arguments.py
            days_of_runs = {'O4':330,'O3':330,'O2':268,'O1':129 } # assume O4 days = O3 days
            
        self.total_days = 0
        self.frame = frame
        self.SNR_th = SNR_thres # SNR threshold

        if self.frame != "detectors_frame":
            print("Injections are computed only in the detector frame. Source frame not implemented yet. Exiting.")
            sys.exit()
            
        self.nprocs = nprocs
        print("Machine with {} cores. Will use {} processes.".format(mp.cpu_count(),self.nprocs))
        self.nclients = self.nprocs+1 # +1 for the main process
        self.manager = mp.Manager()
        self.shared_queue = mp.Queue() # prepare the communication pipe
        self.H0_ref = -1 # init, updated later on if frame = source
        self.Om0_ref = -1 # init, updated later on if frame = source

        # self.frame == 'detectors_frame':
        self.generate_priors_G = self.generate_priors_detector_G
        self.generate_priors_R = self.generate_priors_detector_R

        if psd == None:
            self.psd_dict = {'O1':{},'O2':{},'O3':{},'O4':{}}
        else:
            if isinstance(psd,list):
                self.psd_dict ={psd[0]:{}}
            else:
                self.psd_dict ={psd:{}}

        for key in self.psd_dict:
            self.total_days += days_of_runs[key]
        if self.total_days == 0:
            print("Observation time Tobs = 0. Check parameter 'days_of_runs'. Exiting.")
            sys.exit()

        if duty_factors == None:
            # use the same values as the default ones in gwcosmo/utilities/arguments.py
            duty_factors = {'O4':{'H1':0.75,'L1':0.75,'V1':0.75},
                            'O3':{'H1':0.7459,'L1':0.7698,'V1':0.7597},
                            'O2':{'H1':0.653,'L1':0.618,'V1':0.0777},
                            'O1':{'H1':0.646,'L1':0.574,'V1':-1}}
            print("No duty factors provided, using default values: {}".format(duty_factors))
            
        # get proba of drawing O1, O2, O3, O4
        self.prob_of_run = {}#'O4':0,'O3':0,'O2':0,'O1':0}
        ptot = 0
        for key in self.psd_dict:
            if days_of_runs[key] > 0:
                self.prob_of_run[key] = days_of_runs[key]/self.total_days
                ptot += self.prob_of_run[key]
        if abs(1-ptot) > 0.01:
            print("Anomaly: probabilities don't add-up to 1 (ptot = {}), check the days of activity of runs! Exiting.".format(ptot))
            sys.exit()
        else: # rescale prob of runs, to reach exactly 1
            print(self.prob_of_run)
            for i in self.prob_of_run:
                self.prob_of_run[i] /= ptot
            print(self.prob_of_run)

        self.active_runs = list(self.prob_of_run.keys()) # will be use at some places
        
        self.Tobs_year = self.total_days/365.25



        # create bilby PSDs once for all
        self.bilby_psds = {k:{} for k in self.active_runs}
        
        self.psd_opts = psd_opts # copy optional run keyword (psd_opts is 'low' or 'high' for O4)
        if ('O4' in self.active_runs) and (self.psd_opts == None):
            print("O4 activated but no sensitivity is specified. Selecting the 'low' configuration (pessimistic).")
            self.psd_opts = 'low'
            
        nfiles_ok = 0
        for LVCrun in self.active_runs:
            print(LVCrun)
            for ifo in detectors:
                try:
                    # load strain data, these files have 2 columns: frequency - ASD
                    if LVCrun == 'O4':
                        asd_file = asd_path+ifo+'_'+LVCrun+self.psd_opts+'_strain.txt'
                    else:
                        asd_file = asd_path+ifo+'_'+LVCrun+'_strain.txt'
                    print(asd_file)
                    data = np.genfromtxt(asd_file)
                    self.psd_dict[LVCrun][ifo] = {}
                    self.psd_dict[LVCrun][ifo]['frequency'] = data[:,0]
                    self.psd_dict[LVCrun][ifo]['psd'] = data[:,1]**2 # data[:,1] is asd, so **2 for psd
                    self.psd_dict[LVCrun][ifo]['duty_cycle'] = duty_factors[LVCrun][ifo]
                    self.bilby_psds[LVCrun][ifo] = bl.gw.detector.PowerSpectralDensity(frequency_array=self.psd_dict[LVCrun][ifo]['frequency'],
                                                                                       psd_array=self.psd_dict[LVCrun][ifo]['psd'])
                    nfiles_ok += 1
                except Exception as e:
                    print('Problem in loading asd file for  {:},{:}. Setting up without it.'.format(LVCrun,ifo))
                    print(e)

        if nfiles_ok == 0:
            print("No asd file loaded. Check paths! Exiting.")
            sys.exit()
        
        # create object detector_config
        # this object if used when randomly drawing the run (O1, O2, O3, O4) and the online interferometers
        self.detector_config = detector_config(self.prob_of_run,self.psd_dict)

        # copy data from function arguments
        self.dLmax_depends_on_m1 = dLmax_depends_on_m1
        self.dLmax_m1_params_args = dLmax_m1_params # copied from function arguments, obtained from simulations
        self.approx = approx
        self.priors = priors
        self.priors_limits = priors_limits
        self.fmin = fmin
        self.fsamp = sampling_frequency
        self.dump_inj_period = dump_inj_period
        if self.priors_limits == None:
            # use the same values as the default ones in gwcosmo/bin/create_injections
            self.priors_limits = {'Mmin_det':1.,
                                  'Mmax_det':500.,
                                  'power_index_m1_det':-2.,
                                  'power_index_m2_det':1.,
                                  'Mmin_source':2.,
                                  'Mmax_source':300.,
                                  'power_index_m1_source':-2.,
                                  'power_index_m2_source':1.,
                                  'zmin':0.001,
                                  'zmax':4,
                                  'z_alpha':2.,
                                  'power_index_dL':2.,
                                  'dLmin':0.1,
                                  'dLmax':80e3} # same value for all runs

        self.dl_m1_functions =  self.get_condition_func_dl_m1()

        # check if we use dLmax(m1), available for SNRth = 9, 10, 11, 12
        if self.dLmax_depends_on_m1 and ((self.SNR_th != 9) and (self.SNR_th != 10) and (self.SNR_th != 11) and (self.SNR_th != 12)):
            print("Using dLmax(m1) is OK for SNR_th = 9, 10, 11 or 12 only. Exiting.")
            sys.exit()
        
        self.set_priors() # here we set the priors in the global params space and in the reduced one if activated

        print("Using observation days: {}".format(days_of_runs))
        print("Using duty factors: {}".format(duty_factors))
        print("Using probability of run: {}".format(self.prob_of_run))
        print("Using detectors: {}".format(detectors))
        print("Using Tobs (years): {}".format(self.Tobs_year))
        if 'O4' in self.active_runs:
            print("Using O4 sensitivity: {}".format(self.psd_opts))
        print("Using priors limits: {}".format(self.priors_limits))
        if self.dLmax_depends_on_m1: print("Priors params dLmax(m1): {}".format(self.dLmax_m1_params))
        # define the range for the seed before generation noise in the injections
        self.seedmin = 0
        self.seedmax = 2**32-1
        
    def set_priors(self):

        # create priors once for all, in the global parameter space 'G'
        # this prior is the same for all runs O1, O2, O3, O4
        # random variables are m1, (m2|m1) and dL independent of m1
        # these priors will be used to compute ini_prob of individual injections
        priors_G = self.generate_priors_G()
        self.priors_dict_G = {}
        for k in self.active_runs:
            self.priors_dict_G[k] = priors_G # same priors for all runs in the global space

        if not self.dLmax_depends_on_m1:
            # there is no reduced parameter space, we draw in G
            self.priors_dict_R = self.priors_dict_G

        else:
            # set the priors in the reduced parameter space
            # random variables are m1, (m2|m1) and (dL|m1)
            # if dLmax depends on m1, we use the dLmax(m1,Oi) value, Oi = O1, O2, O3, O4low, O4high
            # empirical functions obtained with simulations using a configuration that maximizes the GW signal
            # m1 = m2, a1 = a2 = 1, inclination = 0...
            # dLmax(m1) = (a[0]+a[1]*m1)*exp(-(m1+a[2])**2/(2*a[3]))
            self.priors_dict_R = {}
            self.dLmax_m1_params = {}
            if self.dLmax_m1_params_args != None: # the user provided the values a[0], a[1], a[2], a[3] for each LVCrun
                for k in self.active_runs: # keep only those for our actual LVCruns
                    self.dLmax_m1_params[k] = self.dLmax_m1_params_args[k]
              
            else: # use the computed ones by simulation
                dLmax_m1 = get_dLmax_params(self.SNR_th) # get params for all runs (O1, O2, O3, O4low, O4high)
                if self.psd_opts == 'low': # rename the O4low or O4high as O4, keep the correct one
                    dLmax_m1['O4'] = dLmax_m1['O4low']
                else:
                    dLmax_m1['O4'] = dLmax_m1['O4high']
                for k in self.active_runs: # keep only active runs
                    self.dLmax_m1_params[k] = dLmax_m1[k]
                        
            for k in self.active_runs:
                self.priors_dict_R[k] = self.generate_priors_R(k)


    def get_condition_func_dl_m1(self):

        return dict(O1=self.condition_func_dlm1_O1,
                    O2=self.condition_func_dlm1_O2,
                    O3=self.condition_func_dlm1_O3,
                    O4=self.condition_func_dlm1_O4)
        
    def condition_func_dlm1_O1(self, reference_params, mass_1):
        
        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax('O1',mass_1))

    def condition_func_dlm1_O2(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax('O2',mass_1))

    def condition_func_dlm1_O3(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax('O3',mass_1))

    def condition_func_dlm1_O4(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax('O4',mass_1))

    def condition_func_m1m2(self,reference_params, mass_1):

        return dict(minimum=self.priors_limits['Mmin_det'], maximum=mass_1)

    def reduced_space_function(self,x,LVCrun):

        '''
        This function returns pG(m1) * int_lmin^lmax(m1) pG(l) dl which is proportional to pR(m1), 
        i.e. proportional to the marginal distribution of m1 in the reduced parameter space
        x is mass_1 in the detector frame and must be a scalar
        LVCrun is O1 or O2, or O3, or O4
        we use the object self.priors_dict_G that contains the pdf of m1 and luminosity_distance
        in the global parameter space
        '''
        dlmax_m1 = self.getdLmax(LVCrun,x)
        int_l = integrate.quad(self.priors_dict_G[LVCrun]['luminosity_distance'].prob,
                               self.priors_dict_G[LVCrun]['luminosity_distance'].minimum,
                               dlmax_m1)[0]
        return self.priors_dict_G[LVCrun]['mass_1'].prob(x)*int_l

    def getdLmax(self,LVCrun,m1):

        '''
        returns the value dLmax(m1) = (a[0]+a[1]*m1)*exp( -(m1+a[2])^2 / (2*a[3]^2) )
        '''
        dlp = self.dLmax_m1_params[LVCrun]
        dl = (dlp[0]+dlp[1]*m1)*np.exp( - (m1+dlp[2])**2 / (2 * dlp[3]**2) )
        #dlcst = {'O1':8e3,'O2':1e4,'O3':1.3e4}
        #return np.min([dl,dlcst[LVCrun]])
        return dl

    
    def generate_priors_detector_G(self):

        '''
        Define the priors in the detector frame, in the global parameter space
        '''
        
        prior_dict = bl.gw.prior.BBHPriorDict()
        prior_dict.pop('mass_1')
        prior_dict.pop('mass_2')
        prior_dict.pop('mass_ratio')
        prior_dict.pop('chirp_mass')
        prior_dict.pop('luminosity_distance')
        prior_dict['mass_1'] = bl.core.prior.PowerLaw(alpha=self.priors_limits['power_index_m1_det'],
                                                      minimum=self.priors_limits['Mmin_det'],
                                                      maximum=self.priors_limits['Mmax_det'],
                                                      name='mass_1')

        prior_dict['mass_2'] = bl.core.prior.ConditionalPowerLaw(name="mass_2",
                                                                 condition_func=self.condition_func_m1m2,
                                                                 alpha=self.priors_limits['power_index_m2_det'], 
                                                                 minimum=self.priors_limits['Mmin_det'],
                                                                 maximum=self.priors_limits['Mmax_det'],
                                                                 latex_label="mass_2")
        
        # default dL law with a constant dLmax (the same for all runs O1, O2, O3, O4)
        prior_dict['luminosity_distance'] = bl.core.prior.PowerLaw(alpha=self.priors_limits['power_index_dL'],
                                                                   minimum=self.priors_limits['dLmin'],
                                                                   maximum=self.priors_limits['dLmax'],
                                                                   name='luminosity_distance')

        prior_dict['geocent_time'] = bl.core.prior.Uniform(minimum=0.,
                                                           maximum=86400.,
                                                           name='geocent_time')

        return prior_dict
    
    
    def generate_priors_detector_R(self,LVCrun):

        '''
        Define the priors in the detector frame, in the reduced parameter space
        '''

        bdict = bl.gw.prior.BBHPriorDict()
        bdict.pop('mass_1')
        bdict.pop('mass_2')
        bdict.pop('mass_ratio')
        bdict.pop('chirp_mass')
        bdict.pop('luminosity_distance')
        prior_dict = bl.core.prior.ConditionalPriorDict()
        prior_dict.update(bdict.copy())
        prior_dict['geocent_time'] = bl.core.prior.Uniform(minimum=0.,
                                                           maximum=86400.,
                                                           name='geocent_time')
        # then prepare specific priors for m1, m2, dL

        npts = 1000
        vx = np.logspace(np.log10(self.priors_limits['Mmin_det']),np.log10(self.priors_limits['Mmax_det']),npts)
        #vx = np.linspace(self.priors_limits['Mmin_det'],self.priors_limits['Mmax_det'],npts)
        vy = np.zeros(npts)
        for i in np.arange(npts):
            vy[i] = self.reduced_space_function(vx[i],LVCrun) # will be normalized inside bl.core.prior.Interped
            
        # then create pR(m1) with the numerical values (vx,vy), pR(m1) is a pdf as bilby normalizes it
        prior_dict['mass_1'] = bl.core.prior.Interped(name="mass_1",
                                                      xx = vx,
                                                      yy = vy,
                                                      minimum=self.priors_limits['Mmin_det'],
                                                      maximum=self.priors_limits['Mmax_det'],
                                                      latex_label="mass_1")
            
        prior_dict['mass_2'] = bl.core.prior.ConditionalPowerLaw(name="mass_2",
                                                                 condition_func=self.condition_func_m1m2,
                                                                 alpha=self.priors_limits['power_index_m2_det'], 
                                                                 minimum=self.priors_limits['Mmin_det'],
                                                                 maximum=self.priors_limits['Mmax_det'],
                                                                 latex_label="mass_2")
        
        prior_dict['luminosity_distance'] = bl.core.prior.ConditionalPowerLaw(name="luminosity_distance",
                                                                              condition_func=self.dl_m1_functions[LVCrun],
                                                                              alpha=self.priors_limits['power_index_dL'],
                                                                              minimum=self.priors_limits['dLmin'],
                                                                              maximum=self.priors_limits['dLmax'],
                                                                              latex_label="luminosity_distance")
        return prior_dict
        
 
    
    def generate_priors_source(self): # not used in the current version

        '''
        Define the priors in the source frame
        '''
        
        if priors == None:
            prior_dict = bl.gw.prior.BBHPriorDict()
            prior_dict.pop('mass_1')
            prior_dict.pop('mass_2')
            prior_dict.pop('mass_ratio')
            prior_dict.pop('chirp_mass')
            prior_dict.pop('luminosity_distance')

            prior_dict['mass_1'] = bl.core.prior.PowerLaw(alpha=self.priors_limits['power_index_m1_source'],
                                                          minimum=self.priors_limits['Mmin_source'],
                                                          maximum=self.priors_limits['Mmax_source'],
                                                          name='mass_1')
            prior_dict['mass_2'] = bl.core.prior.PowerLaw(alpha=self.priors_limits['power_index_m2_source'],
                                                          minimum=self.priors_limits['Mmin_source'],
                                                          maximum=self.priors_limits['Mmax_source'],
                                                          name='mass_2')
            #prior_dict['redshift'] = bl.gw.prior.UniformComovingVolume(minimum=0.01, maximum=4, name="redshift")
            prior_dict['redshift'] = bl.core.prior.PowerLaw(alpha=self.priors_limits['z_alpha'],
                                                            minimum=self.priors_limits['zmin'],
                                                            maximum=self.priors_limits['zmax'],
                                                            name='redshift')
            prior_dict['luminosity_distance'] = 1

        
        return prior_dict

    
    def do_injections(self,
                      Nsamps=100,
                      output_dir='./injection_files',
                      run=1):


        '''
        This is the function computing the injections
        injections parameters are computed in the detector frame
        it splits the work between child processes
        each of them call the function run_bilby_sim to get the SNR
        the functions defines a shared object (using the class multiprocess/Manager) in the list format (it cannot be a dict)
        at the end of the process the shared object is converted into a dict which will be used afterwards to merge with many
        others to have a large number of injections
        parameters
        ----------
        Nsamps: int
        number of requested accepted injections
        default: 100

        output_dir: str
        name of the output directory where injections file will be stored
        default: 'injection_files'

        run: int
        run id to distinguish injection files when using dag
        default: 1
        '''
        
        if not os.path.isdir(output_dir): os.mkdir(output_dir)
        self.Nsamps = Nsamps # number of detected injections required
        if self.Nsamps > 1000 and self.dump_inj_period<self.Nsamps/50 :
            oldv = self.dump_inj_period
            self.dump_inj_period = int(self.Nsamps/50)
            print("Ask for {} injections: reduce the pickle dump period from: {} injections to {} injections".format(self.Nsamps,oldv,self.dump_inj_period))
        
        self.run = run # keyword, will be added to the output filename
        #self.start_time = int(time.time())
        #pattern = "detected_events_{:d}_{:d}_{:.2f}.p".format(self.start_time,self.run,self.SNR_th)
        pattern = "detected_events_{:d}_{:.2f}.p".format(self.run,self.SNR_th) # pattern of the final dict filename
        self.output_file = output_dir+"/tmp_"+pattern  # temporary file
        self.output_file_dict = output_dir+"/"+pattern # final file, containing the dictionnary
        self.Ndet = Counter(0,self.Nsamps) # min, max, to count the number of detected injections
        self.NsimTot = Counter(0) # min, to count the number of tries, whatever the considered run (O1, O2, O3, O4)
        self.Nsim = {k:Counter(0) for k in self.active_runs} # counter specific to the run id (O1, O2, O3, O4)
        self.ErrorInjections = Counter(0) # to count the number of failures in injections
        self.N_no_ifos = Counter(0) # to count the number of time we have no ifos online
        print("Ndet counter: {} -> {}".format(self.Ndet.get_value(),self.Ndet.max_value()))
        print("Nsim counter: {} -> {}".format(self.NsimTot.get_value(),self.NsimTot.max_value()))
        print("Nsim[Oi] counter:")
        for k in self.active_runs:
            print("\t{}:{}".format(k,self.Nsim[k].get_value()))
        print("Temporary events will be written in {}".format(self.output_file))
        print("Final events will be written in {}".format(self.output_file_dict))
                
        # prepare room for the results
        # results objects is a list, not a dict as the multiprocess class does NOT allow dict to be shared objects
        # the results shared object will be converted into a dict at the end of the execution
        self.results = self.manager.list([self.empty]*self.Nsamps)
        indexes = np.arange(self.Nsamps)
        if self.Nsamps < self.nprocs:
            print("Not enough samples to split among processes. Use only 1 process.")
            self.nprocs = 1
        process_indexes = np.array_split(indexes,self.nprocs) # divide the work between subprocesses

        wproc = mp.Process(target=self.write_injections,args=(self.output_file,self.shared_queue))
        procs = [mp.Process(target=self.run_bilby_sim,args=(process_indexes[i],self.results,self.shared_queue)) for i in range(self.nprocs)]

        # start first write process and write the header for this run
        wproc.start()
        time.sleep(0.1)
        self.header_dict = self.get_header()
        self.header = [self.header_msg,self.header_dict]
        self.shared_queue.put(self.header) # first item of the final list
        time.sleep(0.1)

        for p in procs: p.start()
        for p in procs: p.join() # wait for the requested number of detected events to finish

        print("Child processes terminated... Stopping write_injections process.")
        # write footer in pickle file, we need the final statistics
        self.shared_queue.put([self.footer_msg,self.get_footer()])
        time.sleep(1)
        # then kill the writer
        self.shared_queue.put([self.kill_msg])
        wproc.join()

        if self.Ndet.get_value() > 0:
            print("and dealing with results...")

            # convert the big self.results object into a dict
            self.dict_detected = write_result_to_dict(self.results)

            # add header and footer data
            self.dict_detected.update(self.header_dict.copy())
            self.dict_detected.update(self.get_footer().copy()) # final values of Nsim etc

            # add simulation information, final values
            self.dict_detected['Ndet'] = np.max(self.dict_detected['Ndet'])
            self.dict_detected['NsimTot'] = np.max(self.dict_detected['NsimTot'])
            
            # and write it to file. This is NOT the icarogw/gwcosmo dict at this level
            # the icarogw/gwcosmo dict is created when running the executable create_injection with option --combine: doing this,
            # you merge several dicts, needed to have a large number of injections
            write_object_to_file(self.dict_detected,self.output_file_dict)
            # last check:
            nvals = 1
            if self.Ndet.get_value() > 1: nvals = len(self.dict_detected['SNR'])
            print("NsimTot: {}, Ndet: {}, dict_size: {}, ErrorInjections: {}, #no ifos online: {}"
                  .format(self.NsimTot.get_value(),
                          self.Ndet.get_value(),
                          nvals,
                          self.ErrorInjections.get_value(),
                          self.N_no_ifos.get_value()))
            sim_values = 0
            for k in self.active_runs:
                print("Nsim[{}]: {}".format(k,self.Nsim[k].get_value()))
                sim_values += self.Nsim[k].get_value()
            isok = (sim_values == self.NsimTot.get_value())
            print("Check: NsimTot should be: {}: {} ===> {}".format(sim_values,self.NsimTot.get_value(),isok))
        else:
            print("No injections computed, no output file produced.")
        

        print("Done.")
 

    def get_Nsim(self):
        # return dict of real-time values
        return {k:self.Nsim[k].get_value() for k in self.active_runs}
        
    def get_header(self):

        '''
        This funtion returns a dictionnary containing the common data for all injections
        '''
        hdict  = {'SNR_th':self.SNR_th,
                  'Tobs':self.Tobs_year,
                  'frame':self.frame,
                  'prob_of_run':self.prob_of_run}

        if self.frame == "source":
            hdict['H0_ref'] = self.H0_ref
            hdict['Om0_ref'] = self.Om0_ref
            
        return hdict

    def get_footer(self):

        '''
        this function return the desired information in the footer
        it is set as the last line of temporary files and in the last line of the final dict
        we return the current values of the Nsims for the active runs in case we need to build an injection file before the end of the process
        '''
        tmpdict = dict(Nsim=self.get_Nsim())
        tmpdict['tot'] = np.sum(list(tmpdict['Nsim'].values()))
        return tmpdict # format 'Nsim': {'O1': 223, 'O2': 465, 'O3': 551, 'O4': 517, 'tot':1756}
    
    def write_injections(self,filename,queue):

        '''
        This is the function started by the main process as a child process
        it listens for new data on the socket (queue). New data can be new detected injections
        from the run_bilby_sim function, of a request for auto-kill if the number of requested injections is reached,
        or a header describing the conditions of the injections (SNRth, H0, Om...)
        '''
        writer_start = time.time()
        ngets = 0
        nevts = 0
        nhdr = 0
        nfoot = 0
        nkill = 0
        evt_list = []
        write_counter = 0
        write_done = 0
        while True:
            #print("Wait for something in the queue...")
            item = queue.get() # blocking, listens for new data in the socket
            ngets += 1 # new data arrived            
            if item[0] == self.injection_msg:
                # in these cases we record the data
                if item[0] == self.injection_msg:
                    nevts += 1
                    write_counter += 1
                    evt_list.append(item)
                    if write_counter == self.dump_inj_period:
                        write_counter = 0
                        try:
                            write_done += 1
                            start_0 = time.time()
                            # the values of Nsim Oi are at the time of a detected inj so it's OK
                            # careful: the sum of NOi can differ a bit from the NsimTot of the injection as there are some computed simulations
                            # between the time of the received injection and the time we get the counter in the line just below (get_footer()):
                            footer = [self.footer_msg,self.get_footer()]
                            #evt_list.append(footer) # add temporary the footer to ease further file reading
                            data_to_write = copy.deepcopy(evt_list)
                            data_to_write.append(footer)
                            write_object_to_file(data_to_write,self.output_file)
                            del data_to_write
                            #evt_list.pop() # remove the footer for the next injections
                            start_1 = time.time()
                            print("{} (+{} s) Dump injections in {}, iter #{} ({} injections). Dump time: {} s."
                                  .format(start_0,start_0-writer_start,self.output_file,write_done,nevts,start_1-start_0))
                            sys.stdout.flush()
                        except Exception as e:
                            print(e)
            elif item[0] == self.header_msg:
                evt_list.append(item) # record information in tmp file
                nhdr += 1
            elif item[0] == self.footer_msg: # last message, when normal termination
                evt_list.append(item) # record information in tmp file
                nfoot += 1
            elif item[0] == self.kill_msg:
                nkill += 1
                break # don't listen anymore
            else:
                print("Got unknown message type. Ignoring it.")            

        print("write_injections process terminated, got {} msgs:\n\t{} header\n\t{} injections\n\t{} footer\n\t{} kill".format(ngets,nhdr,nevts,nfoot,nkill))

        # last dump on disk to be sure to have all events
        try:
            print("Last dump of event list on file {}".format(self.output_file))
            write_object_to_file(evt_list,self.output_file)
            write_done += 1
            print("File written ({} times)".format(write_done))
        except Exception as e:
            print(e)

        return
        
    
    def run_bilby_sim(self,n,results,shared_queue):

        '''
        The actual injections are computed in this function
        this is a child process started by the main
        the idea is to draw a detector configuration (O1, O2, O3 and available interferometers)
        the presence of a given interferometer is considered independent on the others
        Parameters
        ----------
        n: is the list of indices attributed to the process to write in the list self.results
        results: is the shared object self.results
        shared_queue: is the communication pipe to the process "write_injections"
        '''
        
        mypid = os.getpid()
        np.random.seed(None) # to have independent samples in the different
        print("child process pid: {}, niter: {}, range: [{}-{}]".format(mypid,len(n),n[0],n[-1]))
        idx_n = 0
        dtloop = 0.
        dtinj = 0.
        ninj = 0
        nloop = 0
        nsim_child_tot = 0 # no reset
        nsim_child_tot_LVC = {k:0 for k in self.active_runs} # no reset
        n_nodets = 0
        while idx_n < len(n): # store results in self.results[idx_n], repeat until requested number is obtained

            start_0 = time.time()

            # increase the number of simulations, event if there are no available interferometers
            nsim_child_tot += 1            
            run_LVC, dets = self.detector_config.GetDetectorConfig() # draw a random detector configuration ([O1, O2, O3, O4], + [L, V, H])
            nsim_child_tot_LVC[run_LVC] += 1
            nsim_global_tot = self.NsimTot.increment() # return value after increment
            nsim_global_tot_LVC = self.Nsim[run_LVC].increment() # return value after increment
            
            if len(dets) == 0: # no need to go further, redraw a new configuration after increase of nsim
                n_nodets +=1
                self.N_no_ifos.increment()
                continue
            
            # load the correct prior dict according to O1, O2, O3, O4
            # self.priors_dict_R is self.priors_dict_G if dLmax independent of m1
            priors_dict = self.priors_dict_R[run_LVC]

            injection_parameters = priors_dict.sample(1)
            injection_parameters = {var: injection_parameters[var][0] for var in injection_parameters.keys()} # change array size 1 values into scalar

            if self.frame == 'source_frame': # compute values in the source_frame
                z = injection_parameters['redshift']
                injection_parameters['luminosity_distance'] = bl.gw.conversion.redshift_to_luminosity_distance(z,cosmology=self.cosmo.name)
                injection_parameters['mass_1'] *= (1+z)
                injection_parameters['mass_2'] *= (1+z)
                    
            duration = np.ceil(gwutils.calculate_time_to_merger(frequency = self.fmin,
                                                                mass_1 = injection_parameters['mass_1'],
                                                                mass_2 = injection_parameters['mass_2']))
            if duration<1: duration=1.

            start_1 = time.time()
                
            waveform_arguments = dict(waveform_approximant = self.approx, reference_frequency = self.fmin, minimum_frequency = self.fmin)
            waveform_generator = bl.gw.waveform_generator.WaveformGenerator(
                sampling_frequency = self.fsamp, duration=duration+1,
                frequency_domain_source_model = bl.gw.source.lal_binary_black_hole,
                parameter_conversion = bl.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments = waveform_arguments)
            
            ifos = bl.gw.detector.InterferometerList(dets)
            for i in range(len(ifos)):
                ifos[i].power_spectral_density = self.bilby_psds[run_LVC][ifos[i].name]
                
            # draw a random seed and record it if the injection is detected
            #seed_value = np.random.randint(self.seedmin,self.seedmax,size=1,dtype=np.uint32)[0]
            #np.random.seed(seed_value) # to have independent samples in the different processes
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency = self.fsamp, duration=duration+1,
                start_time = injection_parameters['geocent_time']-duration)
            try:
                ifos.inject_signal(waveform_generator = waveform_generator,
                                   parameters = injection_parameters)                
            except Exception as e:
                print(e)
                self.ErrorInjections.increment() # increment shared counter
                continue

            end_1 = time.time()
            dtinj += end_1-start_1
            ninj += 1

            sum_SNR_sq = 0
            for ifo_string in dets:
                sum_SNR_sq += np.real(ifos.meta_data[ifo_string]['matched_filter_SNR'])**2

            SNR = np.sqrt(sum_SNR_sq)

            if SNR >= self.SNR_th: # detected injection

                # we increment here by +nsims, as a new SNR is computed, avoid updating the shared counter at each injection
                # it means that in the temporary injection file we'll always have the correct Ndet and Nsim values
                ndet = self.Ndet.increment() # update Ndet counter
                idx_result = n[idx_n] # write in this place in the shared list

                # compute the draw probability in the global parameter space
                # self.priors_dict_G[run_LVC] are all the same, independently of O1, O2, O3, O4
                #if self.frame == 'source_frame': 
                #    injection_parameters['mass_1'] /= (1+z)
                #    injection_parameters['mass_2'] /= (1+z)
                #    ini_prob = self.priors_dict_G[run_LVC]['mass_1'].prob(injection_parameters['mass_1'])*\
                #        self.priors_dict_G[run_LVC]['mass_2'].prob(injection_parameters['mass_2'])*\
                #        self.priors_dict_G[run_LVC]['redshift'].prob(z)
                #else:

                ini_prob = self.priors_dict_R[run_LVC].prob({'mass_1':injection_parameters['mass_1'],
                                                             'mass_2':injection_parameters['mass_2'],
                                                             'luminosity_distance':injection_parameters['luminosity_distance']})
            
                # add new keys to injections
                injection_parameters['idx'] = idx_result
                injection_parameters['SNR'] = SNR
                injection_parameters['dt'] = duration
                injection_parameters['pi'] = ini_prob
                injection_parameters['run'] = run_LVC
                injection_parameters['Ndet'] = ndet # write the value recorded previously (at the time of detected injection)
                injection_parameters['NsimTot'] = nsim_global_tot # write the value recorded previously (at the time of detected injection)
                injection_parameters['seed'] = 0#seed_value

                # store data in the list
                results[idx_result] = injection_parameters
                
                idx_n += 1 # increment the index for next detected injection
                shared_queue.put([self.injection_msg,results[idx_result]]) # send the data to the write process which will also add the footer
                
            end_0 = time.time()
            dtloop += end_0-start_0
            nloop += 1
            if 0 and (nloop % 100) == 0:
                print("pid: {}, loop: {} ({}), inject: {} ({}), inj/tot: {} num_nodets: {}".format(mypid, dtloop/(1.*nloop),nloop,
                                                                                                   dtinj/(1.*ninj),ninj,
                                                                                                   (dtinj/(1.*ninj))/(dtloop/(1.*nloop)),n_nodets))
                print("\t-> ",mypid,self.Ndet.get_value(),self.Ndet.max_value(),self.NsimTot.get_value())
                #            if self.Ndet.get_value() >= self.Ndet.max_value():
                #                can_quit.set()
                #                break
                
        print("{}: child {}: process terminated... iter {}/{}, nsim_child_tot: {}, num_no_ifos: {}"
              .format(time.time(),os.getpid(),idx_n,len(n),nsim_child_tot,n_nodets))
        for k in nsim_child_tot_LVC.keys():
            print("\tchild Nsim[{}]:{}".format(k,nsim_child_tot_LVC[k]))
        print("\tchild check: \sum nsim[Oi]: {} == nsimtot: {}".format(np.sum(list(nsim_child_tot_LVC.values())),nsim_child_tot))
            
        return

    
    def combine(self,path='./injection_files',output=None,Tobs=None):

        '''
        This function is typically called after distinct injections are computed i.e.
        we have several dictionnaries corresponding to different runs (for instance with dag)
        the path in argument is the directory where all dictionnaries to be merged are located
        output is the filename where the merged dict will be stored
        '''
        
        files = []
        
        for file in os.listdir(path):
            if file.endswith('.p'): # select *.p files (the dict must end with .p)
                files.append(os.path.join(path, file))

        print("Got {} \".p\" file(s) in dir {}".format(len(files),path))
        first_dict = True
        dict_detected = {}
        ndicts = 0
        avg_ratio = 0.
        sq_ratio = 0.
        ndet_loop = 0
        nsim_loop = 0
        for i,path in enumerate(files):
            if os.path.getsize(path) == 0:
                print("Empty pickle file {}, skip it.".format(path))
                continue
            try:
                data = pickle.load(open(path,"rb"))
            except:
                print("Could not load pickle file {}, skip file.".format(path))
                continue
            if not isinstance(data,dict):
                print("Skip not dict file {}".format(path)) # deal with dict file only
                continue
            else:
                ndicts += 1 
                if first_dict: # first open dict, we create its structure
                    first_dict = False
                    for var in data.keys():
                        dict_detected[var] = data[var]
                else: # for the other dicts, we append their data
                    for var in data.keys():
                        dict_detected[var] = np.hstack((dict_detected[var],data[var]))
                ndet_loop += data['Ndet']
                nsim_loop += data['NsimTot']
                print("dict file {} data recorded (#{}), #injections: {}, #sims: {}, acc rate: {} %, Ndet_tot: {}, Nsim_tot: {}.".format(path,ndicts,data['Ndet'],
                    data['NsimTot'],100*data['Ndet']/data['NsimTot'],ndet_loop,nsim_loop))
                ar = 100.*data['Ndet']/data['NsimTot'] # average acceptance in %
                avg_ratio += ar
                sq_ratio += ar**2

        if ndicts == 0:
            print("No valid dict file found, nothing to merge. Exiting.")
            sys.exit()
            
        dict_detected['avg_ratio'] = avg_ratio/ndicts # average acceptance ratio
        if ndicts > 1:
            # stdev of the acceptance ratio
            dict_detected['std_ratio'] = np.sqrt((sq_ratio/ndicts-dict_detected['avg_ratio']**2)/(ndicts-1))
        else:
            dict_detected['std_ratio'] = 0
        print("average Ndet/Nsim in % over {} files: {} {}".format(ndicts,dict_detected['avg_ratio'],dict_detected['std_ratio']))

        # compute the combined number of detected injections and simulated
        dict_detected['Ndet_tot'] = np.sum(dict_detected['Ndet'])
        dict_detected['NsimTot_total'] = np.sum(dict_detected['NsimTot'])

        if ndicts > 1:
            dict_detected['SNR_th'] = dict_detected['SNR_th'][0]
            dict_detected['frame'] = dict_detected['frame'][0]        

        if (Tobs == None) and (ndicts > 1): 
            dict_detected['Tobs'] = dict_detected['Tobs'][0]
        else:
            dict_detected['Tobs'] = dict_detected['Tobs']

        if output == None:
            output = 'combined_pdet_SNR_'+str(dict_detected['SNR_th'])+'.h5' # hdf5 format by default

        # compute the rescaled ini_prob, needed for the injection object used for the analyses
        if ndicts > 1:
            dict_detected['NsimCombined'] = {k:0 for k in dict_detected['prob_of_run'][0].keys()} # combined values of Nsim['Oi']
            prob_of_run = dict_detected['prob_of_run'][0]
        else:
            dict_detected['NsimCombined'] = {k:0 for k in dict_detected['prob_of_run'].keys()} # combined values of Nsim['Oi']
            prob_of_run = dict_detected['prob_of_run']

        # get Nsim for each run Oi for rescaling    
        active_runs = list(dict_detected['NsimCombined'].keys())
        check_ndet = 0
        for k in active_runs: # loop over the runs O1, O2...
            ww = np.where(dict_detected['run'] == k)[0]
            nww = len(ww)
            check_ndet += nww
            if ndicts == 1:
                dict_detected['NsimCombined'][k] = dict_detected['Nsim'][k] # single dict case
            else:
                dict_detected['NsimCombined'][k] = np.sum([dict_detected['Nsim'][i][k] for i in range(len(dict_detected['Nsim']))]) # sum over the files to combine
                
        print("Check: combined total Nsim[{}]: {}".format(k,nww,dict_detected['NsimCombined'][k]))
        print("Got {} dicts for rescaling of probabilities.".format(ndicts))    
        rescale_initial_probabilites(dict_detected,prob_of_run)
        
        # create the injection object used by icarogw or gwcosmo
        
        if dict_detected['frame'] == 'detectors_frame':
            detected = injections_at_detector(m1d=dict_detected['mass_1'],
                                              m2d=dict_detected['mass_2'],
                                              dl=dict_detected['luminosity_distance'],
                                              prior_vals=dict_detected['pi_rescaled'],
                                              snr_det=dict_detected['SNR'],
                                              snr_cut=0,
                                              ifar=dict_detected['SNR']*np.inf,
                                              ifar_cut=0,
                                              ntotal=dict_detected['NsimTot_total'],
                                              Tobs=dict_detected['Tobs'])
        else:
            if ndicts > 1:
                dict_detected['H0_ref'] = dict_detected['H0_ref'][0]
                dict_detected['Om0_ref'] = dict_detected['Om0_ref'][0]

            cosmo_ref = standard_cosmology(Omega_m=dict_detected['Om0_ref'])
            detected = injections_at_source(cosmo_ref=cosmo_ref,
                                            H0_ref=dict_detected['H0_ref'],
                                            m1s=dict_detected['mass_1'],
                                            m2s=dict_detected['mass_2'],
                                            z=dict_detected['redshift'],
                                            prior_vals=dict_detected['pi_rescaled'],
                                            snr_det=dict_detected['SNR'],
                                            snr_cut=0,
                                            ifar=dict_detected['det_snr']*np.inf,
                                            ifar_cut=0,
                                            ntotal=dict_detected['NsimTot_total'],
                                            Tobs=dict_detected['Tobs'])
        
        self.write_injections_hdf5(detected,output,prob_of_run)
        
        fp = os.path.splitext(output)
        dfile = "stacked_"+os.path.basename(fp[0])+".p"
        pickle.dump(dict_detected,open(dfile,"wb"))
        print("Stacked data from {} dict files written in {}.".format(ndicts,dfile))

    def write_injections_hdf5(self,inj,output,prob_of_run):

        fp = os.path.splitext(output)
        if fp[1] != '.h5':
            output = fp[0] + ".h5"
            print("Can't save injections object in a pickle file, we use .h5 format. Will save in {}".format(output))

        h = h5py.File(output,"w")
        h.create_dataset('m1d',data=inj.m1d_original)
        h.create_dataset('m2d',data=inj.m2d_original)
        h.create_dataset('dl',data=inj.dl_original)
        h.create_dataset('pini',data=inj.ini_prior_original)
        h.create_dataset('snr',data=inj.snr_original)
        h.create_dataset('Tobs',data=inj.Tobs)
        h.create_dataset('ntotal',data=inj.ntotal_original)
        h.create_dataset('ifar',data=inj.ifar)
        # write prob_of_run
        for k in prob_of_run.keys():
            kname = 'prob_'+k
            h.create_dataset(kname,data=prob_of_run[k])
        h.close()
        file_size = os.path.getsize(output)
        print("gwcosmo injections object created in file {} ({} bytes).".format(output,file_size))
            
        
class detector_config(object):
    
    def __init__(self,prob_of_run,psd_dict):
        
        self.prob_of_run = prob_of_run
        self.keys = list(prob_of_run.keys()) # list of keys ['O1','O2','O3','O4']
        self.prob = list(prob_of_run.values()) # list of probas
        self.psd_dict = psd_dict # dict of the form {'O1': {'H1': {'frequency':.... {'L1':....
        
    def GetDetectorConfig(self):
        # draw among O1, O2, O3, O4
        LVCrun = np.random.choice(self.keys,1,replace=True,p=self.prob)[0]
        
        dets = []
        # draw online interferometers according to their duty cycle
        # we assume they are independent (they are online/offline independently)
        for det in self.psd_dict[LVCrun].keys():
            p = np.random.rand(1)[0]
            if p <= self.psd_dict[LVCrun][det]['duty_cycle'] and self.psd_dict[LVCrun][det]['duty_cycle'] >= 0:
                dets.append(det)

        return LVCrun, dets


def write_result_to_dict(result):
    
    '''
    This function converts the object in memory self.results (which is a big list of list), into a dict
    each element is a list composed of 2 elements: an integer (idx) and a dict containing injection parameters
    '''
    
    # structure of the object 'result': inj_params
    nevt = len(result)
    # deal with first element to copy the structure and first data
    idict = result[0]
    all_keys = list(idict.keys()) # get all injection keys
    for i in range(1,nevt): # 1 -> nevt-1
        ldict = result[i]
        for var in all_keys:
            idict[var] = np.hstack((idict[var],ldict[var]))            

    return idict
    
def convert_temp_file_to_dict(temp_file, header_msg, footer_msg, inj_msg):

    '''
    This function is called when we want to convert a temporary file into a dict for injections.
    This can be needed in case of crash of the code or if we want to get injections while they are still computed, i.e.
    without waiting for the normal code to end.
    The function returns 3 variables: the dictionnary computed from the list in the temp_file, the number of injections inside it,
    and a boolean telling if the returned dict should be ignored or not (it is the case when number of injections = 0)
    '''
    
    # expected format: temp_file is a pickle file containing a list
    # first item of the list is: ['header',header_dict]
    # all other items of the list are like:
    # ['NewInjection',
    # [12,
    # {'dec': -0.800192697726566,
    #  'ra': 0.44453410132332744,
    #  'theta_jn': 0.6127555351480456,
    #  'psi': 2.3751939066928043,
    #  'phase': 0.4103638539474935,
    #  'a_1': 0.6244476326815177,
    #  'a_2': 0.35941957055779633,
    #  'tilt_1': 0.7172080985916205,
    #  'tilt_2': 0.44697288768091553,
    #  'phi_12': 4.604601783970368,
    #  'phi_jl': 1.0209595930844262,
    #  'mass_1': 1.4451083092070696,
    #  'mass_2': 1.4193210843212622,
    #  'luminosity_distance': 44552.937806656075,
    #  'geocent_time': 47273.6088415261,
    #  'idx': 12,
    #  'SNR': 1.3618151878203781,
    #  'dt': 171.0,
    #  'pi': 1.4555244193873278e-05,
    #  'run': 'O3',
    #  'Ndet': 13,
    #  'NsimTot': 14,
    #  'seed': 36474848]]
    # last item is: ['footer',footer_dict]

    file_size = os.path.getsize(temp_file)
    # check if there are data in the file
    if file_size == 0:
        return 0, 0, 1
    try:
        fd = open(temp_file,"rb")
    except Exception as e:
        print(e)
        print("Could not open file {}. Exiting.".format(temp_file))
        sys.exit()
        
    try:
        evt_list = pickle.load(fd)
        full_dict, ninj, toskip = convert_list_to_dict(evt_list,header_msg,footer_msg,inj_msg)
    except Exception as e:
        print(e)
        print("Could not unpickle the file {}, file could be damaged, try to redownload it?\n Skipping file...".format(temp_file))
        full_dict = 0
        ninj = 0
        toskip = 1

    fd.close()
    return full_dict, ninj, toskip

def convert_list_to_dict(evt_list, header_msg, footer_msg, inj_msg):

    '''
    This function is used when reading data from a temporary file containing injections
    it converts the list of injections loaded in memory into a dict
    the list is actually a list of list, the first item being the header of the object and the other ones the injections
    we first store the injection data into a list of list in order to call the function
    write_result_to_dict to avoid having 2 functions decoding the data
    '''
    
    # evt_list is expected to be the list of data written in the temporary file during injections
    # first item is the header:
    # ['header',header_dict]
    # injection_msg = "NewInjection"
    # header_msg = "header"
    # footer_msg = "footer"

    hdict = {}
    fdict = {}
    idict = {}
    ninj = 0
    results = []
    for i in range(len(evt_list)):
        msg = evt_list[i]
        if msg[0] == header_msg: # first get the header from the list and copy it in the dict
            hdict = msg[1] # copy the header dict
        elif msg[0] == footer_msg:
            fdict = msg[1] # copy the footer dict
        elif msg[0] == inj_msg: # deal with injections data, store all injections
            results.append(msg[1]) # build a list of list (injections)
            ninj += 1
        else:
            print("Unknown message type in evt list: got {}.".format(msg[0]))

    # convert the 'results' injection list into a dict
    idict = write_result_to_dict(results)

    # copy header/footer dicts in idict
    idict.update(hdict)
    idict.update(fdict)

    # update maximum values
    idict['Ndet'] = np.max(idict['Ndet'])
    idict['NsimTot'] = np.max(idict['NsimTot'])
    
    toskip = False
    if ninj == 0:
        print("No injection in tmp file. Ignoring.")
        toskip = True

    return idict, ninj, toskip


def merge_tmpfile(file,header,footer,injection):

    '''
    This function considers a large number of temporary injections files, convert them into dicts.
    This is very useful when we use the temporary files from different clusters without waiting for all of injections to finish.
    Then all these dicts should be used by the 'combine' function to create the icarogw/gwcosmo injection object.
    the arguments 'header' and 'footer' are the strings used to recognize them in the tmpfile
    '''
    
    # file contains the list (full path) of the tmp files to be merged
    # first item is the header:
    # ['header',self.SNR_th,self.Tobs_year,self.frame,self.H0_ref,self.Om0_ref]
    # self.injection_msg = "NewInjection"
    # self.header_msg = "header"
    tmpfiles = []
    with open(file,'r') as f:
        for l in f:
            l = l.rstrip()
            tmpfiles.append(l)
        f.close()
        
    if len(tmpfiles) == 0:
        print("No tmp file to merge. Exiting.")
        sys.exit()

    tmpdir = tempfile.TemporaryDirectory().name
    print("Merged dict will be in dir {}".format(tmpdir))
    os.mkdir(tmpdir)
    nc = 0
    ninj_total = 0
    nskip_total = 0
    strlength = 12
    for i,f in enumerate(tmpfiles):
        print(i,f)
        ldict, ninj, toskip = convert_temp_file_to_dict(f,header,footer,injection)
        ninj_total += ninj
        nskip_total += toskip
        if not toskip:
            ostring = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(strlength))
            #tmpdir+"/dict_"+str(1e9*np.random.rand(1)[0])+"_"+os.path.basename(f)
            outfile = tmpdir+"/dict_"+ostring+"_"+os.path.basename(f) # add random value as file name can be duplicated
            write_object_to_file(ldict,outfile) # write the dict on disk
            nc += 1
    if nc >= 1:
        print("{} injection files have been converted to dict and stored in {}. Total of {} injections.".format(nc,tmpdir,ninj_total))

    print("{} injection files have been skipped.".format(nskip_total))
    
          
def write_object_to_file(myobject,filename):

    f = open(filename,"wb")
    pickle.dump(myobject,f)
    f.close()


def rescale_initial_probabilites(dict_detected,prob_of_run):

    '''
    This function uses the information contained in the combined dictionnary to recompute the initial probabilities
    of the injections
    the probabilities must be rescaled as follow: ini_prob *= (NOi/Ntotal)*(TObs/TobsOi) = (NOi/Ntotal)/prob_of_run[Oi]
    the ratio (TObs/TobsOi) is 1/prob_of_run
    '''
        
    dict_detected['pi_rescaled'] = copy.deepcopy(dict_detected['pi'])
    check_ndet = 0
    for k in prob_of_run.keys(): # loop over the runs O1, O2...
        ww = np.where(dict_detected['run'] == k)[0]
        nww = len(ww)
        check_ndet += nww        
        print("Check: combined total Nsim[{}]: {}".format(k,nww,dict_detected['NsimCombined'][k]))
        if nww > 0:
            print("Rescale probabilities for {}: {} events...".format(k,nww))
            # here we rescale the ini_prob
            # ini_prob *= (NOi/Ntotal)*(TObs/TobsOi) = (NOi/Ntotal)/prob_of_run[Oi]
            dict_detected['pi_rescaled'][ww] *= (dict_detected['NsimCombined'][k]/dict_detected['NsimTot_total'])/prob_of_run[k]
            #print((dict_detected['NsimCombined'][k]/dict_detected['NsimTot_total'])/prob_of_run[k])
            #print("prob_of_run {}: {}".format(k,prob_of_run[k]))
            #print("ratio N{}/Ntot: {}".format(k,dict_detected['NsimCombined'][k]/dict_detected['NsimTot_total']))

    if check_ndet == dict_detected['Ndet_tot']:
        print("All initial probs have been rescaled: {} vs {}.".format(check_ndet,dict_detected['Ndet_tot']))
    else:
        print("Anomaly in the counting of detected injections.")
