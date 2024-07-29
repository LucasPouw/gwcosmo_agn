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
from bilby.gw import utils as gwutils
import multiprocess as mp
import sys
import time
import copy
import tempfile
import random
import string
import scipy.integrate as integrate
import h5py
import json
from gwcosmo.utilities.injections_utilities import default_ifar_value

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

def get_dLmax_params_bbh(snr):
    '''
    This function returns the values of the parameters of the function dLmax(m1) for snrs 9, 10, 11, 12.
    There are 6 free parameters and the function is:
    dLmax(m1det) = (a[0]+a[1]*m1det)*exp(-(a[2]*m1det+a[3])**2/(a[4]+a[5]*m1det))
    '''
    # data from simulation
    data_sim = {'O1': {9: np.array([ 2.04066927e+03,  4.49451916e+03,  9.36428505e-05,
                                  -2.73318488e+02,  2.37776604e+04, -2.04580982e+01]),
                       10: np.array([ 1.60705883e+03,  3.95333232e+03, -1.20975096e-04,
                                   -2.31800903e+02,  1.68431401e+04, -1.44268218e+01]),
                       11: np.array([ 1.37634967e+03,  3.44095311e+03, -1.86026103e-05,
                                   -3.63472039e+02,  4.11833092e+04, -3.51712995e+01]),
                       12: np.array([ 1.27536347e+03,  2.90389014e+03, -2.77072971e-05,
                                   -3.54071305e+02,  3.94286118e+04, -3.37252981e+01])},
                'O2': {9: np.array([ 2.15802638e+02,  3.09376835e+02, -1.51781889e-01,
                                  -8.51524637e+01,  1.99551520e+04, -1.74670557e+01]),
                       10: np.array([ 1.87996715e+02,  3.02590390e+02, -1.89878216e-01,
                                   -1.47550193e+02,  4.28269025e+04, -3.82596806e+01]),
                       11: np.array([ 1.87175131e+02,  2.91182557e+02, -9.56461603e-02,
                                   -9.42680434e+01,  1.42508309e+04, -1.29381365e+01]),
                       12: np.array([ 1.69569604e+02,  2.53950387e+02, -1.71699447e-01,
                                   -1.73367436e+02,  4.79277018e+04, -4.38140882e+01])},
                'O3': {9: np.array([ 2.40835838e+03,  3.94037749e+03,  1.31850035e-05,
                                  -2.19009275e+02,  1.87719411e+04, -1.74208834e+01]),
                       10: np.array([ 1.91087170e+03,  3.28304392e+03,  4.87640179e-07,
                                   -1.99421860e+02,  1.55665911e+04, -1.44611754e+01]),
                       11: np.array([ 1.53215909e+03,  2.74192848e+03, -9.51077463e-05,
                                   -1.97159907e+02,  1.53870346e+04, -1.43452666e+01]),
                       12: np.array([ 1.29806220e+03,  2.44664338e+03,  7.53270027e-06,
                                   -2.31161468e+02,  2.10296468e+04, -1.95854594e+01])},
                'O4low': {9: np.array([ 2.44049953e+03,  2.96639576e+03,  5.81749937e-05,
                                     -4.07322075e+02,  8.35515214e+04, -8.01815715e+01]),
                          10: np.array([ 1.99008783e+03,  2.38128214e+03, -3.19004145e-05,
                                      -2.94719833e+02,  4.42557140e+04, -4.25652505e+01]),
                          11: np.array([ 1.67184068e+03,  1.96410539e+03,  3.47675031e-05,
                                      -1.99096031e+02,  2.05449796e+04, -1.98809544e+01]),
                          12: np.array([ 1.45673603e+03,  1.72357453e+03, -1.66473436e-06,
                                      -2.32150620e+02,  2.78913978e+04, -2.69648514e+01])},
                'O4high': {9: np.array([ 8.60111084e+02,  6.78385116e+02, -1.31055289e-01,
                                      -1.38539308e+02,  4.01875531e+04, -4.01875528e+01]),
                           10: np.array([ 5.75281226e+02,  4.77289796e+02, -2.32409361e-01,
                                       -1.58674238e+02,  8.11081454e+04, -8.11081432e+01]),
                           11: np.array([ 4.97577061e+02,  3.98378941e+02, -1.86840677e-01,
                                       -1.21963079e+02,  5.05000530e+04, -5.05000526e+01]),
                           12: np.array([ 4.43303905e+02,  3.33903815e+02, -1.32956128e-01,
                                       -8.03063400e+01,  2.40593765e+04, -2.40593765e+01])},
                'O4actual': {9: np.array([ 4.23233131e+03,  4.91528542e+03, -1.09484832e-06,
                                        -2.18188466e+02,  1.75019248e+04, -1.54155715e+01]),
                             10: np.array([ 3.44683840e+03,  3.91988171e+03,  1.61782225e-05,
                                         -2.05859789e+02,  1.57959196e+04, -1.39811527e+01]),
                             11: np.array([ 3.04136624e+03,  3.50044363e+03, -3.16950360e-05,
                                         -2.00775102e+02,  1.48264661e+04, -1.30584163e+01]),
                             12: np.array([ 2.74298860e+03,  3.18482483e+03, -5.43501242e-06,
                                         -1.98708796e+02,  1.43372541e+04, -1.25634487e+01])}}
    dLmax_m1 = {}
    for k in data_sim.keys():
        dLmax_m1[k] = {}
        dLmax_m1[k] = data_sim[k][snr]

    """
    if snr == 9:
        dLmax_m1['O1'] = [122.15726247, 207.22700382, 136.86499411, 284.60513109]
        dLmax_m1['O2'] = [206.11810764, 205.09670503,  30.7884025 , 271.33777237]
        dLmax_m1['O3'] = [221.29342581, 346.4577848 , 144.79389356, 300.18233506]
        dLmax_m1['O4high'] = [506.13296326, 472.93122242, 132.38615963, 356.78473458]
        dLmax_m1['O4low'] = [404.0427499 , 539.12447378, 267.86710129, 380.22198988]
        dLmax_m1['O4actual'] = [404.68120422, 734.50392847, 512.90612213, 421.30273413]
    elif snr == 10:
        dLmax_m1['O1'] = [100.23551329, 170.93163343, 132.2739466 , 284.22846187]
        dLmax_m1['O2'] = [169.88551387, 170.19039544,  32.31566023, 271.5215608 ]
        dLmax_m1['O3'] = [193.39220375, 274.73366595, 126.57766478, 294.86502435]
        dLmax_m1['O4high'] = [397.99068064, 402.51191324, 142.4032627 , 358.04799943]
        dLmax_m1['O4low'] = [330.62172027, 463.68052239, 290.81530559, 389.57811915]
        dLmax_m1['O4actual'] = [289.55272772, 493.46708178, 406.46717933, 390.53290099]
    elif snr == 11:
        dLmax_m1['O1'] = [ 88.87764289, 147.99582589, 139.11492612, 287.09436018]
        dLmax_m1['O2'] = [141.87553348, 146.14830225,  37.87513397, 275.02126004]
        dLmax_m1['O3'] = [170.89759958, 238.21000121, 139.13885567, 300.18067038]
        dLmax_m1['O4high'] = [332.5166634 , 353.01041243, 160.2572857 , 364.94558151]
        dLmax_m1['O4low'] = [268.72371536, 386.22874466, 270.99617381, 381.97915148]
        dLmax_m1['O4actual'] = [239.26238198, 402.92517163, 386.2662383,  385.22412433]
    elif snr == 12:
        dLmax_m1['O1'] = [ 79.66406436, 127.58171485, 132.67504191, 284.22116186]
        dLmax_m1['O2'] = [124.21537846, 125.74500718,  20.49070061, 265.74269184]
        dLmax_m1['O3'] = [150.09056948, 205.01138531, 131.76749229, 296.57518337]
        dLmax_m1['O4high'] = [299.9216596 , 312.34226974, 181.49196053, 377.80390419]
        dLmax_m1['O4low'] = [220.08277715, 324.77506694, 240.93653008, 368.43169707]
        dLmax_m1['O4actual'] = [219.9698398,  380.77703764, 427.91543449, 398.47682656]
    else:
        print("dLmax(m1) parameters not available for snr {}.".format(snr))
    """
    print(dLmax_m1)
    return dLmax_m1

def get_dLmax_params_nsbh(snr):
    '''
    This function returns the values of the parameters of the function dLmax(m1) for snrs 9, 10, 11, 12.
    Parameters are: a[0], a[1], a[2], a[3] with dLmax(m1) = 10**(a[0]*x^3+a[1]*x^2+a[2]*x+a[3]) and x=log10(m_1)
    '''
    dLmax_m1 = {}
    if snr == 9:
        dLmax_m1['O1'] = [ 0.00587152, -0.21975222,  1.02456944,  2.42359273]
        dLmax_m1['O2'] = [ 0.02435531, -0.2825121 ,  1.08022574,  2.4986606 ]
        dLmax_m1['O3'] = [ 0.02373704, -0.29138348,  1.10425765,  2.62831346]
        dLmax_m1['O4high'] = [ 0.02802821, -0.30738743,  1.12155647,  2.8240411 ]
        dLmax_m1['O4low'] = [ 0.02272994, -0.28057881,  1.08116767,  2.78453275]
    elif snr == 10:
        dLmax_m1['O1'] = [ 0.00455388, -0.21702421,  1.02942804,  2.33629673]
        dLmax_m1['O2'] = [ 0.01872344, -0.26118551,  1.0610112 ,  2.418382  ]
        dLmax_m1['O3'] = [ 0.01041187, -0.24113774,  1.05328889,  2.55853546]
        dLmax_m1['O4high'] = [ 0.03149434, -0.31977825,  1.13059301,  2.7441521 ]
        dLmax_m1['O4low'] = [ 0.02350469, -0.28172115,  1.07889854,  2.70514947]
    elif snr == 11:
        dLmax_m1['O1'] = [ 0.00980355, -0.23562446,  1.0434737 ,  2.26956586]
        dLmax_m1['O2'] = [ 0.02014312, -0.26395514,  1.05932468,  2.35230411]
        dLmax_m1['O3'] = [ 0.01464885, -0.25977284,  1.0779242 ,  2.47970703]
        dLmax_m1['O4high'] = [ 0.02371022, -0.28764923,  1.09666764,  2.68152237]
        dLmax_m1['O4low'] = [ 0.02343764, -0.2873342 ,  1.09212453,  2.63221537]
    elif snr == 12:
        dLmax_m1['O1'] = [ 0.01296042, -0.24867541,  1.05739071,  2.20965658]
        dLmax_m1['O2'] = [ 0.02531685, -0.28390746,  1.079732  ,  2.28938808]
        dLmax_m1['O3'] = [ 0.0145394 , -0.259253  ,  1.07658204,  2.42358978]
        dLmax_m1['O4high'] = [ 0.02282253, -0.28692155,  1.09920497,  2.62023754]
        dLmax_m1['O4low'] = [ 0.02009177, -0.27856124,  1.09009913,  2.57077879]
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

    nsbh: bool
    injections for the NSBH case, always in the detector frame
    if activated, the condition function p(m2|m1) is a power law with a maximal value set to 10 solar masses (detector frame)
    if not activated,  the condition function p(m2|m1) is a power law with a maximal value set to m1 for BBHs
    default: False

    combine: bool
    if True, then skip most of the __init__ for the executable to merge individual injection files
    if False, proceed for the full __init__
    default: False

    dump_inj_period: int
    it is the number of detected injections needed to trigger the dump on disk of the temporary injection file
    keep it large to avoid spending too much time writing those pickle file
    when a very large number of injections is required, the period is set to 1/50th of the requested number
    default: 200

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
                 nsbh=False,
                 combine=False,
                 dump_inj_period=200,
                 isrun = False):


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
            raise ValueError("The current version of the injection code does not treat priors from file. Exiting.")
                
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
                raise ValueError("Could not open file {}. Exiting.".format(list_to_dict))

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
                raise ValueError("Could not open file {}. Exiting.".format(tmpfile_to_dict))

            sys.exit()
            
        if tmp_to_stdout != None:
            # display on stdout the contents of a temporary injections file
            print("Displaying tmp file {} on stdout".format(tmp_to_stdout))            
            if os.path.isfile(tmp_to_stdout):
                try:
                    fd = open(tmp_to_stdout,"rb")
                except Exception as e:
                    print(e)
                    raise ValueError("Could not open file {}. Exiting.".format(tmp_to_stdout))
                try:
                    evt_list = pickle.load(fd)
                    for i in evt_list:
                        print(i)
                except Exception as e:
                    print(e)
                    print("Could not unpickle the file {}, file could be damaged, try to redownload it?\nSkipping file...".format(tmp_to_stdout))
            else:
                raise ValueError("Invalid file {}. Exiting.".format(tmp_to_stdout))

            sys.exit()

        if combine:
            return
            
        if days_of_runs == None:
            # use the same values as the default ones in gwcosmo/utilities/arguments.py
            days_of_runs = {'O4':330,'O3':330,'O2':268,'O1':129 } # assume O4 days = O3 days
            
        self.total_days = 0
        self.frame = frame
        self.SNR_th = SNR_thres # SNR threshold

        if self.frame != "detectors_frame":
            raise ValueError("Injections are computed only in the detector frame. Source frame not implemented yet. Exiting.")
        
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
            raise ValueError("Observation time Tobs = 0. Check parameter 'days_of_runs'. Exiting.")

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
            raise ValueError("Anomaly: probabilities don't add-up to 1 (ptot = {}), check the days of activity of runs! Exiting.".format(ptot))

        else: # rescale prob of runs, to reach exactly 1
            print(self.prob_of_run)
            for i in self.prob_of_run:
                self.prob_of_run[i] /= ptot
            print(self.prob_of_run)

        self.active_runs = list(self.prob_of_run.keys()) # will be used at some places
        
        self.Tobs_year = self.total_days/365.25

        # create bilby PSDs once for all
        self.bilby_psds = {k:{} for k in self.active_runs}
        
        self.psd_opts = psd_opts # copy optional run keyword (psd_opts is 'low' or 'high' for O4)
        if ('O4' in self.active_runs) and (self.psd_opts == None):
            print("O4 activated but no sensitivity is specified. Selecting the 'low' configuration (pessimistic).")
            self.psd_opts = 'low'
            
        nfiles_ok = 0
        print(duty_factors,self.active_runs)
        for LVCrun in self.active_runs:
            print(LVCrun)
            for ifo in detectors:
                if duty_factors[LVCrun][ifo] == 0:
                    print("Skip ifo {} as its prob of run == 0".format(ifo))
                    continue
                try:
                    # load strain data, these files have 2 columns: frequency - ASD
                    if LVCrun == 'O4': # there can be several psds for O4, 'actual', 'avg', 'high', 'low', 'O4b' sensitivities
                        if self.psd_opts == 'high' or self.psd_opts == 'low':
                            asd_file = asd_path+ifo+'_'+LVCrun+self.psd_opts+'_strain.txt'
                        elif self.psd_opts == 'actual':
                            if ifo == 'L1':
                                asd_file = asd_path+'LLO_O4a.txt'
                            elif ifo == 'H1':
                                asd_file = asd_path+'LHO_O4a.txt'
                            elif ifo == 'V1':
                                raise ValueError("There is no data from Virgo in O4a. You should remove Virgo for the detectors. Exiting.")
                        elif self.psd_opts == 'MDC':
                            if ifo == 'L1':
                                asd_file = asd_path+'aligo_O4high.txt'
                            elif ifo == 'H1':
                                asd_file = asd_path+'aligo_O4high.txt'
                            elif ifo == 'V1':
                                asd_file = asd_path+'avirgo_O4high_NEW.txt'
                        elif self.psd_opts == 'avg':
                            if ifo == 'L1':
                                asd_file = asd_path+'L1_O4a_avg.txt'
                            elif ifo == 'H1':
                                asd_file = asd_path+'H1_O4a_avg.txt'
                            elif ifo == 'V1':
                                raise ValueError("There is no data from Virgo in O4a. You should remove Virgo for the detectors. Exiting.")
                        elif self.psd_opts == 'late':
                            if ifo == 'L1':
                                asd_file = asd_path+'LLO_1384459938.txt'
                            elif ifo == 'H1':
                                asd_file = asd_path+'LHO_1388988918.txt'
                            elif ifo == 'V1':
                                raise ValueError("There is no data from Virgo in O4a. You should remove Virgo for the detectors. Exiting.")
                        elif self.psd_opts == 'O4b':
                            asd_file = asd_path+ifo+"_O4b_avg.txt"
                        else:
                            raise ValueError("No O4 sensitivity with name {}. You can choose among 'high', 'low', 'MDC', 'actual', 'avg',  'late', 'O4b'.")
                    else:
                        asd_file = asd_path+ifo+'_'+LVCrun+'_strain.txt'

                    print("Using sentivity file: {}".format(asd_file))
                    data = np.genfromtxt(asd_file)
                    self.psd_dict[LVCrun][ifo] = {}
                    self.psd_dict[LVCrun][ifo]['frequency'] = data[:,0]
                    self.psd_dict[LVCrun][ifo]['psd'] = data[:,1]**2 # data[:,1] is asd, so **2 for psd
                    self.psd_dict[LVCrun][ifo]['duty_cycle'] = duty_factors[LVCrun][ifo]
                    self.bilby_psds[LVCrun][ifo] = bl.gw.detector.PowerSpectralDensity(frequency_array=self.psd_dict[LVCrun][ifo]['frequency'],
                                                                                       psd_array=self.psd_dict[LVCrun][ifo]['psd'])
                    nfiles_ok += 1
                except Exception as e:
                    print('Problem in loading asd file for  {:}, {:}. Setting up without it.'.format(LVCrun,ifo))
                    print(e)

        if nfiles_ok == 0:
            raise ValueError("No asd file loaded. Check paths! Exiting.")
        
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
                                  'Mmax_det':1000.,
                                  'power_index_m1_det':-2.,
                                  'power_index_m2_det':1.,
                                  'power_index_dL':2.,
                                  'dLmin':0.1,
                                  'dLmax':80e3} # same value for all runs

        # check if we use dLmax(m1), available for SNRth = 9, 10, 11, 12
        if self.dLmax_depends_on_m1 and ((self.SNR_th != 9) and (self.SNR_th != 10) and (self.SNR_th != 11) and (self.SNR_th != 12)):
            raise ValueError("Using dLmax(m1) is OK for SNR_th = 9, 10, 11 or 12 only. Exiting.")

        # by default we compute injections for BBHs
        self.get_dLmax_params = get_dLmax_params_bbh
        self.condition = self.condition_func_m1m2_bbh
        self.dl_m1_functions =  self.get_condition_func_dl_m1_bbh()
        self.getdLmax = self.getdLmax_bbh

        self.nsbh = nsbh
        # deal with the NSBH case
        if self.nsbh:
            print("Using NSBH parameters.")
            self.Mmax_nsbh_detframe = 10
            self.get_dLmax_params = get_dLmax_params_nsbh
            self.condition = self.condition_func_m1m2_nsbh
            self.dl_m1_functions =  self.get_condition_func_dl_m1_nsbh()
            self.getdLmax = self.getdLmax_nsbh
        else:
            print("Using BBH parameters.")
            
        print("Using conditional probability p(m2|m1): {}".format(self.condition))
        self.set_priors() # here we set the priors in the global params space and in the reduced one if activated

        # dump priors to file in case of merging bbh-nsbh-bns
        pickle.dump(self.priors_dict_R,open("priors.p","wb"))
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

        # Deal with the multiprocessing stuff
        self.nprocs = nprocs
        print("Machine with {} cores. Will use {} processes.".format(mp.cpu_count(),self.nprocs))
        self.nclients = self.nprocs+1 # +1 for the main process
        self.manager = mp.Manager()
        self.shared_queue = mp.Queue() # prepare the communication pipe
        
        
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
                dLmax_m1 = self.get_dLmax_params(self.SNR_th) # get params for all runs (O1, O2, O3, O4low, O4high)
                if self.psd_opts == 'low': # rename the O4low or O4high or O4actual as O4, keep the correct one
                    print("Using 'O4low' sensitivity")
                    dLmax_m1['O4'] = dLmax_m1['O4low']
                elif self.psd_opts == 'high':
                    print("Using 'O4high' sensitivity")
                    dLmax_m1['O4'] = dLmax_m1['O4high']
                elif self.psd_opts == 'actual':
                    print("Using 'O4actual' sensitivity")
                    dLmax_m1['O4'] = dLmax_m1['O4actual']
                elif self.psd_opts == 'MDC':
                    print("WARNING MDC case: Using 'O4high' sensitivity for the random draws")
                    dLmax_m1['O4'] = dLmax_m1['O4high']
                elif self.psd_opts == 'avg':
                    print("WARNING AVG case: Using 'O4high' sensitivity for the random draws")
                    dLmax_m1['O4'] = dLmax_m1['O4high']
                elif self.psd_opts == 'late':
                    print("WARNING LATE case: Using 'O4high' sensitivity for the random draws")
                    dLmax_m1['O4'] = dLmax_m1['O4high']
                else:
                    print("ERROR in psd_opts.")
                    
                for k in self.active_runs: # keep only active runs
                    self.dLmax_m1_params[k] = dLmax_m1[k]
                        
            for k in self.active_runs:
                self.priors_dict_R[k] = self.generate_priors_R(k)

    ##### conditional functions for BBHs ############

    def get_condition_func_dl_m1_bbh(self):

        return dict(O1=self.condition_func_dlm1_O1_bbh,
                    O2=self.condition_func_dlm1_O2_bbh,
                    O3=self.condition_func_dlm1_O3_bbh,
                    O4=self.condition_func_dlm1_O4_bbh)
        
    def condition_func_dlm1_O1_bbh(self, reference_params, mass_1):
        
        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax_bbh('O1',mass_1))

    def condition_func_dlm1_O2_bbh(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax_bbh('O2',mass_1))

    def condition_func_dlm1_O3_bbh(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax_bbh('O3',mass_1))

    def condition_func_dlm1_O4_bbh(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax_bbh('O4',mass_1))

    def condition_func_m1m2_bbh(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['Mmin_det'], maximum=mass_1)

    def getdLmax_bbh(self,LVCrun,m1):

        '''
        old function: returns the value dLmax(m1) = (a[0]+a[1]*m1)*exp( -(m1+a[2])^2 / (2*a[3]^2) )
        new function: returns the value dLmax(m1) = (a0+a1*x)* np.exp(-(a2*x+a3)**2/(a4+a5*x))
        '''
        dlp = self.dLmax_m1_params[LVCrun]
        # old function dl = (dlp[0]+dlp[1]*m1)*np.exp( - (m1+dlp[2])**2 / (2 * dlp[3]**2) )
        # new function, valid for masses up to 1000
        dl = (dlp[0]+dlp[1]*m1)*np.exp( - (dlp[2]*m1+dlp[3])**2 / (dlp[4]+dlp[5]*m1) )
        #dlcst = {'O1':8e3,'O2':1e4,'O3':1.3e4}
        #return np.min([dl,dlcst[LVCrun]])
        #print(LVCrun,m1,dl,dlp)
        return dl

    
    ##### conditional functions for NSBHs ############
    
    def get_condition_func_dl_m1_nsbh(self):

        return dict(O1=self.condition_func_dlm1_O1_nsbh,
                    O2=self.condition_func_dlm1_O2_nsbh,
                    O3=self.condition_func_dlm1_O3_nsbh,
                    O4=self.condition_func_dlm1_O4_nsbh)
        
    def condition_func_dlm1_O1_nsbh(self, reference_params, mass_1):
        
        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax_nsbh('O1',mass_1))

    def condition_func_dlm1_O2_nsbh(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax_nsbh('O2',mass_1))

    def condition_func_dlm1_O3_nsbh(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax_nsbh('O3',mass_1))

    def condition_func_dlm1_O4_nsbh(self, reference_params, mass_1):

        return dict(minimum=self.priors_limits['dLmin'],maximum=self.getdLmax_nsbh('O4',mass_1))

    def condition_func_m1m2_nsbh(self, reference_params, mass_1):

        # m2_max <= self.Mmax_nsbh_detframe for NS in detector frame
        # careful to use np.minimum and not np.min!
        return dict(minimum=self.priors_limits['Mmin_det'], maximum=np.minimum(mass_1,self.Mmax_nsbh_detframe))

    def getdLmax_nsbh(self,LVCrun,m1):

        '''
        returns the value dLmax(m1) = 10**(a[0]*x^3+a[1]*x^2+a[2]*x+a[3]) and x=log10(m_1)
        '''
        dlp = self.dLmax_m1_params[LVCrun]
        x = np.log10(m1)
        dl = 10**(dlp[0]*(x**3)+dlp[1]*(x**2)+dlp[2]*x+dlp[3])
        return dl

    
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
                                                                 condition_func=self.condition,
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
                                                                 condition_func=self.condition,
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
        self.NdetTot = Counter(0,self.Nsamps) # min, max, to count the number of detected injections
        self.NsimTot = Counter(0) # min, to count the number of tries, whatever the considered run (O1, O2, O3, O4)
        self.Nsim = {k:Counter(0) for k in self.active_runs} # counter specific to the run id (O1, O2, O3, O4)
        self.Ndet = {k:Counter(0) for k in self.active_runs} # counter specific to the run id (O1, O2, O3, O4)
        self.ErrorInjections = Counter(0) # to count the number of failures in injections
        self.N_no_ifos = Counter(0) # to count the number of time we have no ifos online
        print("Ndet counter: {} -> {}".format(self.NdetTot.get_value(),self.NdetTot.max_value()))
        print("Nsim counter: {} -> {}".format(self.NsimTot.get_value(),self.NsimTot.max_value()))
        print("Nsim[Oi] counter:")
        for k in self.active_runs:
            print("\t{}:{}".format(k,self.Nsim[k].get_value()))
        print("Ndet[Oi] counter:")
        for k in self.active_runs:
            print("\t{}:{}".format(k,self.Ndet[k].get_value()))
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
        self.shared_queue.put([self.footer_msg,self.get_Nsim_dict(),self.get_Ndet_dict()])
        time.sleep(1)
        # then kill the writer
        self.shared_queue.put([self.kill_msg])
        wproc.join()

        if self.NdetTot.get_value() > 0:
            print("and dealing with results...")

            # convert the big self.results object into a dict
            self.dict_detected = write_result_to_dict(self.results)

            # add header and footer data
            self.dict_detected.update(self.header_dict.copy())
            self.dict_detected.update(self.get_Nsim_dict().copy()) # final values of Nsim etc

            # add dict for Ndet injections, by run Oi
            self.dict_detected.update(self.get_Ndet_dict().copy())

            # add simulation information, final values
            self.dict_detected['NdetTot'] = np.max(self.dict_detected['NdetTot'])
            self.dict_detected['NsimTot'] = np.max(self.dict_detected['NsimTot'])
            
            # and write it to file. This is NOT the icarogw/gwcosmo dict at this level
            # the icarogw/gwcosmo dict is created when running the executable create_injection with option --combine: doing this,
            # you merge several dicts, needed to have a large number of injections
            write_object_to_file(self.dict_detected,self.output_file_dict)
            # last check:
            nvals = 1
            if self.NdetTot.get_value() > 1: nvals = len(self.dict_detected['SNR'])
            print("NsimTot: {}, Ndet: {}, dict_size: {}, ErrorInjections: {}, #no ifos online: {}"
                  .format(self.NsimTot.get_value(),
                          self.NdetTot.get_value(),
                          nvals,
                          self.ErrorInjections.get_value(),
                          self.N_no_ifos.get_value()))
            for k in self.active_runs:
                print("Nsim[{}]: {}".format(k,self.Nsim[k].get_value()))
            # check Nsims: self.dict_detected['NsimTot'] is the max value of the shared counter of simulations
            # and self.dict_detected['Nsim']['tot'] is the sum of the  Nsim values per run
            isok = (self.dict_detected['NsimTot'] == self.dict_detected['Nsim']['tot'])
            print("Check: NsimTot should be: {}: {} ===> {}".format(self.dict_detected['NsimTot'],self.dict_detected['Nsim']['tot'],isok))
            det_values = 0
            for k in self.active_runs:
                print("Ndet[{}]: {}".format(k,self.Ndet[k].get_value()))
                det_values += self.Ndet[k].get_value()
            isok = (det_values == self.NdetTot.get_value())
            print("Check: NdetTot should be: {}: {} ===> {}".format(det_values,self.NdetTot.get_value(),isok))
        else:
            print("No injections computed, no output file produced.")
        

        print("Done.")
 

    def get_Nsim(self):
        # return dict of real-time values for all injections (detected or not)
        return {k:self.Nsim[k].get_value() for k in self.active_runs}

    def get_Ndet(self):
        # return dict of real-time values for the detected injections
        return {k:self.Ndet[k].get_value() for k in self.active_runs}
    
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

    def get_Nsim_dict(self):

        '''
        this function return the desired information in the footer
        it is set as the last line of temporary files and in the last line of the final dict
        we return the current values of the Nsims for the active runs in case we need to build an injection file before the end of the process
        '''
        tmpdict = dict(Nsim=self.get_Nsim())
        tmpdict['Nsim']['tot'] = np.sum(list(tmpdict['Nsim'].values()))
        return tmpdict # format 'Nsim': {'O1': 223, 'O2': 465, 'O3': 551, 'O4': 517, 'tot':1756}

    def get_Ndet_dict(self):

        '''
        this function returns the number of detected injection in each run Oi
        '''
        tmpdict = dict(Ndet=self.get_Ndet())
        tmpdict['Ndet']['tot'] = np.sum(list(tmpdict['Ndet'].values()))
        return tmpdict # format 'Ndet': {'O1': 223, 'O2': 465, 'O3': 551, 'O4': 517, 'tot':1756}
    
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
                            # between the time of the received injection and the time we get the counter in the line just below (get_Nsim_dict()):
                            footer = [self.footer_msg,self.get_Nsim_dict(),self.get_Ndet_dict()]
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
        ndet_child_tot = 0 # no reset
        ndet_child_tot_LVC = {k:0 for k in self.active_runs} # no reset
        n_nodets = 0
        while idx_n < len(n): # store results in self.results[idx_n], repeat until requested number is obtained

            start_0 = time.time()

            # increase the number of simulations, event if there are no available interferometers
            nsim_child_tot += 1            
            run_LVC, dets = self.detector_config.GetDetectorConfig(self.psd_opts) # draw a random detector configuration ([O1, O2, O3, O4], + [L, V, H])
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
                sum_SNR_sq += np.real(ifos.meta_data[ifo_string]['matched_filter_SNR'])**2 # + some noise with exp=0, var=1

            SNR = np.sqrt(sum_SNR_sq)

            if SNR >= self.SNR_th: # detected injection

                # we increment here by +nsims, as a new SNR is computed, avoid updating the shared counter at each injection
                # it means that in the temporary injection file we'll always have the correct Ndet and Nsim values
                ndet_global_tot = self.NdetTot.increment() # update Ndet counter
                ndet_global_LVC = self.Ndet[run_LVC].increment() # update Ndet counter
                ndet_child_tot += 1
                ndet_child_tot_LVC[run_LVC] += 1
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
                ini_prob_theta_jn = self.priors_dict_R[run_LVC].prob({'mass_1':injection_parameters['mass_1'],
                                                                      'mass_2':injection_parameters['mass_2'],
                                                                      'luminosity_distance':injection_parameters['luminosity_distance'],
                                                                      'theta_jn':injection_parameters['theta_jn']})
                ini_prob_spins = self.priors_dict_R[run_LVC].prob({'mass_1':injection_parameters['mass_1'],
                                                                   'mass_2':injection_parameters['mass_2'],
                                                                   'luminosity_distance':injection_parameters['luminosity_distance'],
                                                                   'a_1':injection_parameters['a_1'],
                                                                   'tilt_1':injection_parameters['tilt_1'],
                                                                   'a_2':injection_parameters['a_2'],
                                                                   'tilt_2':injection_parameters['tilt_2']})
                                                                   
            
                # add new keys to injections
                #injection_parameters['idx'] = idx_result # don't record the idx, never used afterwards
                injection_parameters['SNR'] = SNR
                #injection_parameters['dt'] = duration # don't record the dt, never used afterwards
                injection_parameters['pi'] = ini_prob
                injection_parameters['pi_theta_jn'] = ini_prob_theta_jn
                injection_parameters['pi_spins'] = ini_prob_spins
                injection_parameters['run'] = run_LVC
                injection_parameters['NdetTot'] = ndet_global_tot # write the value recorded previously (at the time of detected injection)
                injection_parameters['NsimTot'] = nsim_global_tot # write the value recorded previously (at the time of detected injection)
                #injection_parameters['seed'] = 0#seed_value

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
        for k in ndet_child_tot_LVC.keys():
            print("\tchild Ndet[{}]:{}".format(k,ndet_child_tot_LVC[k]))
        csimsum = np.sum(list(nsim_child_tot_LVC.values()))
        print("\tchild check: \sum nsim[Oi]: {} == nsimtot: {} => {}".format(csimsum,nsim_child_tot,csimsum==nsim_child_tot))
        cdetsum = np.sum(list(ndet_child_tot_LVC.values()))
        print("\tchild check: \sum ndet[Oi]: {} == ndettot: {} => {}".format(cdetsum,ndet_child_tot,cdetsum==ndet_child_tot))
            
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
                ndet_loop += data['NdetTot']
                nsim_loop += data['NsimTot']
                print("dict file {} data recorded (#{}), #injections: {}, #sims: {}, acc rate: {} %, Ndet_tot: {}, Nsim_tot: {}.".format(path,ndicts,data['NdetTot'],
                    data['NsimTot'],100*data['NdetTot']/data['NsimTot'],ndet_loop,nsim_loop))
                ar = 100.*data['NdetTot']/data['NsimTot'] # average acceptance in %
                avg_ratio += ar
                sq_ratio += ar**2

        if ndicts == 0:
            raise ValueError("No valid dict file found, nothing to merge. Exiting.")
            
        dict_detected['avg_ratio'] = avg_ratio/ndicts # average acceptance ratio
        if ndicts > 1:
            # stdev of the acceptance ratio
            dict_detected['std_ratio'] = np.sqrt((sq_ratio/ndicts-dict_detected['avg_ratio']**2)/(ndicts-1))
        else:
            dict_detected['std_ratio'] = 0
        print("average Ndet/Nsim in % over {} files: {} {}".format(ndicts,dict_detected['avg_ratio'],dict_detected['std_ratio']))

        # compute the combined number of detected injections and simulated
        dict_detected['NdetTot_total'] = np.sum(dict_detected['NdetTot'])
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
        active_runs = list(np.unique(dict_detected['run']))
        if ndicts > 1:
            prob_of_run = dict_detected['prob_of_run'][0]
        else:
            prob_of_run = dict_detected['prob_of_run']
            
        print("Got data for runs: {}".format(active_runs))
        dict_detected['NsimCombined'] = {k:0 for k in active_runs} # prepare dict for combined values of Nsim['Oi']
        dict_detected['NdetCombined'] = {k:0 for k in active_runs} # prepare dict for combined values of Ndet['Oi']

        # get Nsim for each run Oi for rescaling
        # and add a new integer field for O1-O2-O3-O4 -> {1,2,3,4} to avoid strings in the hdf5 file
        dict_detected['run_integer'] = np.zeros(len(dict_detected['run']),dtype=np.ushort)

        check_ndet = 0
        run_id = {}
        for k in active_runs:
            run_id[k] = int(k[k.find('O')+1]) # convert string 'O2' into an int, here 2
        print("run ids: {}".format(run_id))

        for k in active_runs: # loop over the runs O1, O2...
            ww = np.where(dict_detected['run'] == k)[0]
            dict_detected['run_integer'][ww] = run_id[k]
            nww = len(ww)
            check_ndet += nww
            if ndicts == 1:
                dict_detected['NsimCombined'][k] = dict_detected['Nsim'][k] # single dict case
                dict_detected['NdetCombined'][k] = dict_detected['Ndet'][k] # single dict case
            else:
                dict_detected['NsimCombined'][k] = np.sum([dict_detected['Nsim'][i][k] for i in range(len(dict_detected['Nsim']))]) # sum over the files to combine
                dict_detected['NdetCombined'][k] = np.sum([dict_detected['Ndet'][i][k] for i in range(len(dict_detected['Ndet']))]) # sum over the files to combine
            print("Check: combined total Ndet[{}]: {}".format(k,nww,dict_detected['NdetCombined'][k]))

        print(dict_detected)
        print("Got {} dicts for rescaling of probabilities.".format(ndicts))    
        rescale_initial_probabilites(dict_detected,prob_of_run)
        print("Probabilities rescaled by factors:\n{}".format(dict_detected['rescale']))

        # write the data in hdf5 file
        self.write_injections_hdf5(dict_detected,output,prob_of_run)
        
        fp = os.path.splitext(output)
        dfile = "stacked_"+os.path.basename(fp[0])+".p"
        pickle.dump(dict_detected,open(dfile,"wb"))
        print("Stacked data from {} dict files written in {}.".format(ndicts,dfile))

    def write_injections_hdf5(self,inj,output,prob_of_run):
        '''
        input:
           inj: object obtained after 'combine' of several dicts of injections
           output: file name with extension h5
           prob_of_run: write in h5 file the probability of run to remember later one the configuration of injections
        '''
        fp = os.path.splitext(output)
        if fp[1] != '.h5':
            output = fp[0] + ".h5"
            print("Can't save injections object in a pickle file, we use .h5 format. Will save in {}".format(output))

        h = h5py.File(output,"w")

        h.create_dataset('m1d',data=inj['mass_1'])
        h.create_dataset('m2d',data=inj['mass_2'])
        h.create_dataset('dl',data=inj['luminosity_distance'])
        h.create_dataset('ra',data=inj['ra'])
        h.create_dataset('dec',data=inj['dec'])
        h.create_dataset('theta_jn',data=inj['theta_jn'])
        h.create_dataset('a_1',data=inj['a_1'])
        h.create_dataset('a_2',data=inj['a_2'])
        h.create_dataset('tilt_1',data=inj['tilt_1'])
        h.create_dataset('tilt_2',data=inj['tilt_2'])
        h.create_dataset('pini',data=inj['pi_rescaled'])
        h.create_dataset('pini_theta_jn',data=inj['pi_theta_jn_rescaled'])
        h.create_dataset('pini_spins',data=inj['pi_spins_rescaled'])
        h.create_dataset('snr',data=inj['SNR'])
        h.create_dataset('Tobs',data=inj['Tobs'])
        h.create_dataset('ntotal',data=inj['NsimTot_total'])
        h.create_dataset('ifar',data=default_ifar_value+0*inj['SNR']) # no IFAR computed with these injections
        h.create_dataset('run',data=inj['run_integer']) # keep the information on the run id, needed when merging nsbh, bbh... inj files

        # write prob_of_run, rescale factors and other data
        for k in prob_of_run.keys():
            kname = 'prob_'+k
            h.create_dataset(kname,data=prob_of_run[k])
            kname = 'Nsim_'+k
            h.create_dataset(kname,data=inj['NsimCombined'][k])
            kname = 'Ndet_'+k
            h.create_dataset(kname,data=inj['NdetCombined'][k])
            if k in inj['rescale'].keys():
                kname = 'scale_'+k
                h.create_dataset(kname,data=inj['rescale'][k])

        h.create_dataset('NdetTot',data=inj['NdetTot_total'])
        h.create_dataset('NsimTot',data=inj['NsimTot_total'])
        h.close()
        file_size = os.path.getsize(output)
        print("injection data written in file {} ({} bytes).".format(output,file_size))


def read_injection_file(file):
    '''
        input:
           file: path to a filename, format hdf5, containing injection data
        output:
           1) injection object for selection effect computation
           2) the dict built from the hdf5 file, after removing of fields present in the injection object
    '''
    try:
        injdata = h5py.File(file,'r')
    except:
        raise ValueError("Cannot open hdf5 file {}. Exiting.".format(file))
        
    injdict = {k:injdata[k][()] for k in injdata.keys()} # copy all data to python dict
    injdata.close()
    inj_object = injections_at_detector(m1d=injdict['m1d'],
                                        m2d=injdict['m2d'],
                                        dl=injdict['dl'],
                                        prior_vals=injdict['pini'],
                                        snr_det=injdict['snr'],
                                        snr_cut=0,
                                        ifar=injdict['ifar'],
                                        ifar_cut=0,
                                        ntotal=injdict['ntotal'],
                                        Tobs=injdict['Tobs'])
    # remove useless data now that the injection object exists
    # but keep the fields needed for further use, in case of injection concatenation for instance
    # when we need to recompute the probabilities
    injdict.pop('m1d',None)
    injdict.pop('m2d',None)
    injdict.pop('dl',None)
    injdict.pop('pini',None)
    injdict.pop('snr',None)
    injdict.pop('ifar',None)
    return inj_object,injdict

class detector_config(object):
    
    def __init__(self,prob_of_run,psd_dict):
        
        self.prob_of_run = prob_of_run
        self.keys = list(prob_of_run.keys()) # list of keys ['O1','O2','O3','O4']
        self.prob = list(prob_of_run.values()) # list of probas
        self.psd_dict = psd_dict # dict of the form {'O1': {'H1': {'frequency':.... {'L1':....

    def GetDetectorConfig(self, isrun=None):

        # first, draw among O1, O2, O3, O4
        LVCrun = np.random.choice(self.keys,1,replace=True,p=self.prob)[0]
        
        # hack for MDC injections, cf https://git.ligo.org/simone.mastrogiovanni/micecatv1_mdc/-/blob/main/MDCutils.py?ref_type=heads
        if (LVCrun == 'O4'):
            if isrun == 'MDC':
                lucky = np.random.rand()    
                if lucky < 0.5:
                    ifos = ['H1','L1','V1']
                elif (lucky >= 0.5) & (lucky < 0.64):
                    ifos = ['H1','L1']
                elif (lucky >= 0.64) & (lucky < 0.78):
                    ifos = ['H1','V1']
                elif (lucky >= 0.78) & (lucky < 0.92):
                    ifos = ['L1','V1']
                elif (lucky >= 0.92) & (lucky < 0.94):
                    ifos = ['H1']
                elif (lucky >= 0.94) & (lucky < 0.96):
                    ifos = ['L1']
                elif (lucky >= 0.96) & (lucky < 0.98):
                    ifos = ['V1']
                else:
                    ifos = []

            elif isrun == 'O4b': # https://gwosc.org/detector_status/O4b/
                # I'm using the values as of 20240728200000 CEST = 20240728180000 UTC
                p1 = 0.154 # 1-fold
                p2 = 0.365 # 2-fold
                p3 = 0.379 # 3-fold
                pH = 0.518 # H1 uptime
                pL = 0.742 # L1 uptime
                pV = 0.760 # V1 uptime
                # we need the probas for 2-fold 'HL', 'HV', 'LV' and the probas for 1-fold 'H', 'L', 'V'
                # have to fix 2 values as the system is degenerated, for instance we fix the 1-fold pH1 and pL1
                pH1 = 0.03
                pL1 = 0.05
                res = np.array([pL-2*p3-p2-pL1-pH1+pH,p2-pL+p3+pL1,p2+p3-pH+pH1,p1-pH1-pL1]) # compute the probas [pHL,pHV,pLV,pV1]
                pV1 = res[3]
                p_pairs = res[0:3]/np.sum(res[0:3]) # normalized probas for 2-fold [pHL,pHV,pLV]
                p_single = np.array([pH1,pL1,pV1])/p1 # normalized probas for 1-fold [H,L,V]
                lucky = np.random.rand()
                aifos = ['H1','L1','V1']
                pairs = [0,1,2]
                lucky = np.random.rand()
                if lucky < p3: # 3 ifos online
                    ifos = aifos
                elif (lucky >= p3) & (lucky < p3+p2): # 2 ifos online
                    pair = np.random.choice(pairs,size=1,replace=False,p=p_pairs)
                    if pair == 0: ifos = ['H1','L1']
                    elif pair == 1: ifos = ['H1','V1']
                    else: ifos = ['L1','V1']
                elif (lucky >= p3+p2) & (lucky < p3+p2+p1):
                    ifos = list(np.random.choice(aifos,size=1,replace=False,p=np.array(p_single)))
                    ifos[0] = str(ifos[0]) # cast to built-in 'str' instead of numpy.str to avoid a crash with bilby.gw.detector.InterferometerList(dets)
                else:
                    ifos = []
                    
            else: # 'O4a'
                # we refine the random draw of H1, L1 using 1-fold and 2-fold uptimes, see https://gwosc.org/detector_status/O4a/
                # warning: on this web page, the probabilities don't add-up to 1: 53.4 + 29.7 + 16.6 = 99.7%
                # Derek Davis on mattermost DetCharTools 20230325: the missing 0.3% should be considered as 0 ifo
                # these are probas for 'observing' mode
                lucky = np.random.rand()    
                if lucky < 0.534:
                    ifos = ['H1','L1']
                elif (lucky >= 0.535) & (lucky < 0.69): #0.535 + 15.5 = 69% for L1 during O4a
                    ifos = ['L1']
                elif (lucky >= 0.69) & (lucky < 0.83): # 0.535 + 0.83 - 0.69 = 67.5% for H1 during O4a
                    ifos = ['H1']
                else: # proba = 17%
                    ifos = []

            return 'O4',ifos
        
        
        
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
        raise ValueError("Could not open file {}. Exiting.".format(temp_file))
        
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
    # footer_msg = "footer" contains 2 dicts

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
            fdict.update(msg[1])
            fdict.update(msg[2])
            #fdict[d1.keys()] = d1.values() # copy the first dict of the footer
            #fdict[d2.keys()] = d2.values() # copy the second dict of the footer
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
    idict['NdetTot'] = np.max(idict['NdetTot'])
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
        raise ValueError("No tmp file to merge. Exiting.")

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
    dict_detected['pi_theta_jn_rescaled'] = copy.deepcopy(dict_detected['pi_theta_jn'])
    dict_detected['pi_spins_rescaled'] = copy.deepcopy(dict_detected['pi_spins'])
    check_ndet = 0
    dict_detected['rescale'] = {}
    for k in prob_of_run.keys(): # loop over the runs O1, O2...
        ww = np.where(dict_detected['run'] == k)[0]
        nww = len(ww)
        check_ndet += nww        
        print("Check: combined total Nsim[{}]: {}".format(k,nww,dict_detected['NsimCombined'][k]))
        if nww > 0:
            print("Rescale probabilities for {}: {} events...".format(k,nww))
            # here we rescale the ini_prob
            # ini_prob *= (NOi/Ntotal)*(TObs/TobsOi) = (NOi/Ntotal)/prob_of_run[Oi]
            dict_detected['rescale'][k] = (dict_detected['NsimCombined'][k]/dict_detected['NsimTot_total'])/prob_of_run[k]
            dict_detected['pi_rescaled'][ww] *= dict_detected['rescale'][k]
            dict_detected['pi_theta_jn_rescaled'][ww] *= dict_detected['rescale'][k]
            dict_detected['pi_spins_rescaled'][ww] *= dict_detected['rescale'][k]
            #print((dict_detected['NsimCombined'][k]/dict_detected['NsimTot_total'])/prob_of_run[k])
            #print("prob_of_run {}: {}".format(k,prob_of_run[k]))
            #print("ratio N{}/Ntot: {}".format(k,dict_detected['NsimCombined'][k]/dict_detected['NsimTot_total']))

    if check_ndet == dict_detected['NdetTot_total']:
        print("All initial probs have been rescaled: {} vs {}.".format(check_ndet,dict_detected['NdetTot_total']))
    else:
        print("Anomaly in the counting of detected injections.")
