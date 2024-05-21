"""
Module for managing Injections and calculate selection effects.

Benoit Revenu, Christos Karathanasis, Simone Mastrogiovanni
"""

import numpy as _np
import copy as _copy
import sys as _sys
import h5py as _h5py
import pickle as _pickle
import bilby as _bilby
from astropy import units as _u
from scipy.special import logsumexp as _logsumexp


__all__=['injections_at_detector','injections_at_source']


class Injections():
    
    def __init__(self,m1d,m2d,dl,prior_vals,snr_det,snr_cut,ifar,ifar_cut,ntotal,Tobs,condition_check=False,Nobs=-1):

        """
        This class is used to manage a list of detected injections to calculated
        GW selection effects. This class uses injections which are given in source frame.

        Parameters
        ----------
        m1d: _np.arrray
            Mass 1 detector frame of detected events (provide if file_injection is not provided)
        m2d: _np.arrray
            Mass 2 detector frame of detected events (provide if file_injection is not provided)
        dl: _np.arrray
            luminosity distance of detected events (provide if file_injection is not provided)
        prior_vals: _np.arrray
            Used prior draws for injections (provide if file_injection is not provided). This array contains the injections probabilities either
            in the source frame or detector frame, according to the child class that initializes it (either injections_at_detector or injections_at_source)
        snr_det: _np.arrray
            SNR of detected events (provide if file_injection is not provided)
        snr_cut: float
            Set different to 0 if you want to apply a different SNR cut.
        ifar: _np.array
            IFAR of the detected events (IFAR = Inverse False Alarm Rate)
        ifar_cut: float
            Set different to 0 if you wanto to apply a different IFAR cut.
        ntotal: float
            Total number of simulated injections (detected and not). This is necessary to compute the expected number of detections
        Tobs: float
            Length of time for the run in years (used to calculate rates)
        condition_check: boolean, True or False
            is True, we check wether the injections cover the entire prior range
        Nobs: integer
            it's the number of observed GW events in the current analysis
        """
        # Saves what you provided in the class
        self.condition_check = condition_check
        self.ntotal = 1.*ntotal # floating value to prevent an overflow when computing ntotal**2
        self.ntotal_original = 1.*ntotal # floating value
        self.dl_original = dl
        self.m1d_original = m1d
        self.m2d_original = m2d
        self.snr_original = snr_det
        self.ini_prior_original = prior_vals
        self.ifar = ifar
        self.Tobs = Tobs
        self.Nobs = Nobs # needed for the check on Neff >= 4*Nobs, the check will give True in the default case (Nobs==-1)
        self.update_cut(snr_cut,ifar_cut)

    def get_selected_idx(self,snr_cut=0,ifar_cut=0):

        print('Selecting injections with SNR {:f} OR IFAR {:f} yr'.format(snr_cut,ifar_cut))
        self.snr_cut = snr_cut
        self.ifar_cut = ifar_cut
        return _np.where((self.snr_original>self.snr_cut) | (self.ifar>self.ifar_cut))[0]

    def set_selected_idx(self,idet):

        self.idet = idet
        self.m1det = self.m1d_original[self.idet]
        self.m2det = self.m2d_original[self.idet]
        self.dldet = self.dl_original[self.idet]
        self.ini_prior = self.ini_prior_original[self.idet]
        self.log_jacobian_det = self.log_origin_jacobian[self.idet]
    
    def update_cut(self,snr_cut=0,ifar_cut=0,fraction=None):

        idet = self.get_selected_idx(snr_cut,ifar_cut)
        
        #Sub-sample the selected injections in order to reduce the computational load
        if fraction is not None and (fraction > 0) and (fraction <= 1):
            idet = _np.random.choice(idet,size=int(len(idet)*fraction),replace=False)
            self.ntotal = int(self.ntotal_original*fraction)
            print('Working with a total of {:d} injections'.format(len(idet)))

        self.set_selected_idx(idet)
    
    def detector_frame_to_source_frame(self,cosmo,m1det,m2det,dldet):
        
        z_samples = cosmo.z_dgw(dldet)
        den = 1+z_samples
        ms1 = m1det/den
        ms2 = m2det/den

        return ms1, ms2, z_samples


    def detector_to_source_jacobian(self,z,cosmo):
        
        zp1 = z+1
        jacobian = zp1*zp1*cosmo.ddgw_dz(z)

        return jacobian

    
    def update_VT(self,cosmo,m_prior,z_prior,z_prior_norm):
        """
        This method updates the sensitivity estimations.

        Parameters
        ----------
        m_prior : mass prior class
            mass prior class from the prior.mass module
        z_prior : redshift prior class
        """

        self.ms1, self.ms2, self.z_samples = self.detector_frame_to_source_frame(cosmo,self.m1det,self.m2det,self.dldet)

        # Checks if the injections cover the entire prior range. If not, throws an errror if the flag is True
        if self.condition_check:
            mass_a = _np.hstack([ms1,ms2])
            if (_np.max(mass_a)<m_prior.mmax) | (_np.min(mass_a)>m_prior.mmin):
                print('The  injections source frame masses are not enough to cover all the prior range')
                print('Masses prior range {:.2f}-{:.2f} Msol'.format(m_prior.mmin,m_prior.mmax))
                print('Injections range {:.2f}-{:.2f} Msol'.format(_np.min(mass_a),_np.max(mass_a)))
                _sys.exit()


        log_numer = _np.log(z_prior(self.z_samples))+m_prior.log_joint_prob(self.ms1,self.ms2)
        log_jacobian_term = _np.log(_np.abs(self.detector_to_source_jacobian(self.z_samples, cosmo)))-self.log_jacobian_det
        self.log_weights_astro = log_numer-_np.log(self.ini_prior)-log_jacobian_term
        self.log_weights = self.log_weights_astro - _np.log(z_prior_norm)
        # This is the Volume-Time in which we expect to detect. You can multiply it by R_0 Tobs to get the expected number of detections in Gpc^3 yr
        self.VT_sens=_np.exp(_logsumexp(self.log_weights_astro))/self.ntotal
        # This is the fraction of events we expect to detect, a.k.a. the selection effect
        self.VT_fraction=self.VT_sens/z_prior_norm

    def calculate_Neff(self):
        """
        Calculates the effective number of injections contributing to the calculation of the selection effect.
        returns Neff and the boolean result of (Neff>= 4*self.Nobs) and the variance
        See Eq 9 of (https://arxiv.org/pdf/1904.10879.pdf) for more details
        """
        mean = self.VT_fraction
        var = _np.exp(_logsumexp(self.log_weights*2))/(self.ntotal**2)-(mean**2)/self.ntotal
        Neff = (mean**2)/var
        return Neff, (Neff >= 4*self.Nobs), var

    def expected_number_detection(self,R0):
        """
        This method will return the expected number of GW detection given the injection set. Tobs is automatically saved in the class creation

        Parameters
        ----------
        R0 : float
            Merger rate in comoving volume in Gpc-3yr-1
        """
        return self.VT_sens*R0*self.Tobs

    def gw_only_selection_effect(self):
        """
        Will evaluate the GW-only selection effect using the set of injections

        Returns
        -------
        Selection effect (float)
        """
        return self.VT_fraction

    
class injections_at_detector(Injections):
    """
    A class to handle a list of detected GW signals from simulations in detectors frame. This can be used to
    evaluate selection effects or detection expectations under some priors
    """
    
    def __init__(self,m1d,m2d,dl,prior_vals,snr_det,snr_cut,ifar,ifar_cut,ntotal,Tobs,condition_check=False):
        
        self.log_origin_jacobian = _np.zeros(len(m1d))
        Injections.__init__(self,m1d,m2d,dl,prior_vals,snr_det,snr_cut,ifar,ifar_cut,ntotal,Tobs,condition_check=False)

    
class injections_at_source(Injections):
    """
    A class to handle a list of detected GW signals from simulations in source frame. This can be used to
    evaluate selection effects or detection expectations under some priors
    """

    def __init__(self,cosmo_ref,m1s,m2s,z,prior_vals,snr_det,snr_cut,ifar,ifar_cut,ntotal,Tobs,condition_check=False):

        self.cosmo_ref = cosmo_ref
        self.z_original = z
        self.m1s_original = m1s
        self.m2s_original = m2s
        
        # Convert from source frame to detector frame and select injections according to SNR and IFAR
        m1d, m2d, dl = self.source_frame_to_detector_frame(self.cosmo_ref,self.m1s_original,self.m2s_original,self.z_original)
        self.log_origin_jacobian = _np.log(_np.abs(self.detector_to_source_jacobian(self.z_original,self.cosmo_ref)))
        Injections.__init__(self,m1d,m2d,dl,prior_vals,snr_det,snr_cut,ifar,ifar_cut,ntotal,Tobs,condition_check=False)
        
        
    def source_frame_to_detector_frame(self,cosmo,ms_1,ms_2,z_samples):
        
        dl = cosmo.dgw_z(self, z_samples)

        md1 = ms_1*(1+z_samples)
        md2 = ms_2*(1+z_samples)

        return md1, md2, dl
