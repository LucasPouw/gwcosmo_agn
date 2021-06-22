"""
gwcosmoLikelihood Module
Rachel Gray, Archisman Ghosh, Ignacio Magana, John Veitch, Ankan Sur

"""
from __future__ import absolute_import
import numpy as np
from numpy import inf

import warnings
warnings.filterwarnings("ignore")

from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, truncnorm

from .utilities.standard_cosmology import dl_zH0,M_mdl,L_M
from .utilities.schechter_function import M_Mobs
from .utilities.schechter_params import SchechterParams
from .prior.catalog import color_names, color_limits
from .likelihood.skymap import ra_dec_from_ipix,ipix_from_ra_dec
import healpy as hp

import time
import progressbar


################################################################################
################################# THE MAIN CLASSES #############################
################################################################################

class gwcosmoLikelihood(object):
    """
    Class for holding basic GW info for calculating the likelihood on H0
    """

    def __init__(self, px_zH0, pD_zH0, zprior, zrates, zmax=10.):
        """
        Parameters
        ----------
        px_zH0 : object
            GW data, p(x|z,H0)
        pD_zH0 : object
            probability of detection, p(D|z,H0)
        zprior : object
            redshift prior, p(z)
        zrates : object
            rate evolution function, p(s|z)
        zmax : float
            The upper redshift limit for the universe (default=10.)
        """

        self.px_zH0 = px_zH0
        self.pD_zH0 = pD_zH0
        self.zprior = zprior
        self.zrates = zrates
        self.zmax = zmax

    def px_zH0_times_pz_times_ps_z(self, z, H0):
        """
        p(x|z,H0)*p(z)*p(s|z)

        Parameters
        ----------
        z : float
            redshift
        H0 : float
            Hubble constant value in kms-1Mpc-1
        """

        return self.px_zH0(z,H0)*self.zprior(z)*self.zrates(z)

    def pD_zH0_times_pz_times_ps_z(self, z, H0):
        """
        p(D|z,H0)*p(z)*p(s|z)

        Parameters
        ----------
        z : float
            redshift
        H0 : float
            Hubble constant value in kms-1Mpc-1
        """

        return self.pD_zH0(z,H0)*self.zprior(z)*self.zrates(z)

    def px_OH0(self, H0, skyprob=1.):
        """
        Evaluate p(x|O,H0).

        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        skyprob : float, optional
            GW sky probability covered by area O (default=1.)

        Returns
        -------
        float
            p(x|O,H0)
        """

        integral = quad(self.px_zH0_times_pz_times_ps_z,0.,self.zmax,args=[H0],epsabs=0,epsrel=1.49e-4)[0]
        return integral * skyprob

    def pD_OH0(self, H0, skyprob=1.):
        """
        Evaluate p(D|O,H0).

        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        skyprob : float, optional
            pdet probability covered by area O (default=1.)

        Returns
        -------
        float
            p(D|O,H0)
        """

        integral = quad(self.pD_zH0_times_pz_times_ps_z,0.,self.zmax,args=[H0],epsabs=0,epsrel=1.49e-4)[0]
        return integral * skyprob


class GalaxyCatalogLikelihood(gwcosmoLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the galaxy
    catalogue method.

    Parameters
    ----------
    skymap : gwcosmo.likelihood.skymap.skymap object
        provides p(x|Omega) and skymap properties
    fast_cosmology : gwcosmo.utilities.standard_cosmology.fast_cosmology object
        Cosmological model
    Kcorr : bool, optional
        Should K corrections be applied to the analysis? (default=False)
        Will raise an error if used in conjunction with a galaxy catalogue
        without sufficient color information.
    mth : float, optional
        Specify an apparent magnitude threshold for the galaxy catalogue
        (default=None). If none, mth is estimated from the galaxy catalogue.
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=None)
    zmax : float, optional
        The upper redshift limit for integrals (default=10.). Should be well
        beyond the highest redshift reachable by GW data or selection effects.
    zuncert : bool, optional
        Should redshift uncertainties be marginalised over? (Default=True).

    """

    def __init__(self, skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=False, zmax=10.):
        """
        Parameters
        ----------
        skymap : object
            The GW skymap
        observation_band : str
            Observation band (eg. 'B', 'K', 'u', 'g')
        fast_cosmology : object
            Fast cosmology
        px_zH0 : object
            GW data, p(x|z,H0)
        pD_zH0 : object
            probability of detection, p(D|z,H0)
        zprior : object
            redshift prior, p(z)
        zrates : object
            rate evolution function, p(s|z)
        luminosity_prior : object
            absolute magnitude prior, p(M|H0)
        luminosity_weights : object
            luminosity weighting function, p(s|M)
        Kcorr : bool, optional
            Should K corrections be applied to the analysis? (default=False)
        zmax : float, optional
            The upper redshift limit for the universe (default=10.)
        """

        super().__init__(px_zH0, pD_zH0, zprior, zrates, zmax=zmax)

        self.skymap = skymap
        self.cosmo = fast_cosmology
        self.luminosity_prior = luminosity_prior
        self.luminosity_weights = luminosity_weights

        self.Kcorr = Kcorr
        self.band = observation_band
        sp = SchechterParams(self.band)
        self.Mmin_obs = sp.Mmin
        self.Mmax_obs = sp.Mmax


    def px_zH0_times_pz_times_ps_z_times_pM_times_ps_M(self, M, z, H0):
        """
        p(x|z,H0)*p(z)*p(s|z)*p(M|H0)*p(s|M)

        Parameters
        ----------
        M : float
            absolute magnitude
        z : float
            redshift
        H0 : float
            Hubble constant value in kms-1Mpc-1
        """

        return self.px_zH0(z,H0)*self.zprior(z)*self.zrates(z)*self.luminosity_prior(M,H0)*self.luminosity_weights(M)

    def pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M(self, M, z, H0):
        """
        p(D|z,H0)*p(z)*p(s|z)*p(M|H0)*p(s|M)

        Parameters
        ----------
        M : float
            absolute magnitude
        z : float
            redshift
        H0 : float
            Hubble constant value in kms-1Mpc-1
        """

        return self.pD_zH0(z,H0)*self.zprior(z)*self.zrates(z)*self.luminosity_prior(M,H0)*self.luminosity_weights(M)

    def pxD_GH0(self, H0, sampz, sampm, sampra, sampdec, sampcolor, count):
        """
        Evaluate p(x|G,H0) and p(D|G,H0) using galaxy samples.

        Parameters
        ----------
        H0 : array of floats
            Hubble constant value(s) in kms-1Mpc-1
        sampz, sampm, sampra, sampdec, sampcolor : arrays of floats
            redshift, apparent magnitude, right ascension, declination and
            colour samples
        count : the number of samples which belong to 1 galaxy

        Returns
        -------
        arrays
            numerator and denominator
        """

        # TODO: Move into the catalog class
        Kcorr = self.full_catalog.get_k_correction(self.band, sampz, color_names[self.band], sampcolor)

        tempsky = self.skymap.skyprob(sampra, sampdec)*self.skymap.npix

        zweights = self.zrates(sampz)

        tempnum = np.zeros([len(H0)])
        tempden = np.zeros([len(H0)])
        for k,h in enumerate(H0):
            numinner = self.px_zH0(sampz,h)
            deninner = self.pD_zH0(sampz,h)
            sampAbsM = M_mdl(sampm, self.cosmo.dl_zH0(sampz, h), Kcorr=Kcorr)

            # for samples which are fainter than the faint end of the Schechter function
            # set contribution to zero.
            Mmax = M_Mobs(h,self.Mmax_obs)
            sel = np.where(sampAbsM > Mmax)[0] # identify samples fainter than model allows
            Lweights = self.luminosity_weights(sampAbsM)
            if self.luminosity_weights.luminosity_weights == False:
                Lweights = np.ones(len(sampAbsM))*Lweights
            Lweights[sel] = 0 # set faint sample contribution to zero

            normsamp = 1./count

            tempnum[k] = np.sum(numinner*tempsky*Lweights*zweights*normsamp)
            tempden[k] = np.sum(deninner*Lweights*zweights*normsamp)

        return tempnum,tempden

    def pxD_GH0_multi(self,H0, z, sigmaz, m, ra, dec, color, nfine=10000, ncoarse=10, zcut=10.):
        """
        Evaluate p(x|G,H0) and p(D|G,H0) using a list of galaxies.
        Wrapper for pxD_GH0().

        Parameters
        ----------
        H0 : array of floats
            Hubble constant value(s) in kms-1Mpc-1
        z, sigmaz, m, ra, dec, color : arrays of floats
            redshift, redshift 1 sigma uncertainty, apparent magnitude, right
            ascension (radians), declination (radians) and colour for a set of
            galaxies
        nfine : int, optional
            The number of samples to take from galaxies which are sampled finely
        ncoarse : int, optional
            The number of samples to take from galaxies which are sampled coarsely
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=10.)

        Returns
        -------
        arrays
            p(x|G,H0), p(D|G,H0)
        """

        nGal = len(z)
        galindex_sep = {}
        if self.luminosity_weights.luminosity_weights == True:
            # TODO: find better selection criteria for sampling
            print(m)
            mlim = np.percentile(np.sort(m),0.01) # more draws for galaxies in brightest 0.01 percent
            mlim = np.ceil(mlim * 10) / 10.0 # round up to nearst dp to avoid rounding error where no galaxies are selected
            samp_res = {'fine': nfine, 'coarse': ncoarse}
            galindex = {'fine': np.where(m <= mlim)[0], 'coarse': np.where(mlim < m)[0]}

            # for arrays with more than 1million entries, break into sub arrays
            no_chunks_coarse = int(np.ceil(len(galindex['coarse'])/1000000))
            chunks_coarse = np.array_split(galindex['coarse'],no_chunks_coarse)
            galindex_sep['coarse'] = {i+1 : chunks_coarse[i] for i in range(no_chunks_coarse)}
            galindex_sep['fine'] = {i : galindex['fine'] for i in range(1)}
        else:
            samp_res = {'coarse': ncoarse}
            galindex = {'coarse': np.arange(len(z))}

            # for arrays with more than 1million entries, break into sub arrays
            no_chunks_coarse = int(np.ceil(len(galindex['coarse'])/1000000))
            chunks_coarse = np.array_split(galindex['coarse'],no_chunks_coarse)
            galindex_sep['coarse'] = {i : chunks_coarse[i] for i in range(no_chunks_coarse)}

        K = sum(len(v) for v in galindex.values()) # total number of sub arrays
        tempnum = np.zeros([K,len(H0)])
        tempden = np.zeros([K,len(H0)])

        # loop over sub arrays of galaxies
        for i,key in enumerate(samp_res):
            print('{} galaxies are getting sampled {}ly'.format(len(galindex[key]),key))
            for n, key2 in enumerate(galindex_sep[key]):
                zs = z[galindex_sep[key][key2]]
                sigmazs = sigmaz[galindex_sep[key][key2]]
                ms = m[galindex_sep[key][key2]]
                ras = ra[galindex_sep[key][key2]]
                decs = dec[galindex_sep[key][key2]]
                colors = color[galindex_sep[key][key2]]

                sampz, sampm, sampra, sampdec, sampcolor, count = gal_nsmear(zs, sigmazs, ms, ras, decs, colors, samp_res[key], zcut=zcut)

                tempnum[key2,:],tempden[key2,:] = self.pxD_GH0(H0, sampz, sampm, sampra, sampdec, sampcolor, count)

        num = np.sum(tempnum,axis=0)/nGal
        den = np.sum(tempden,axis=0)/nGal

        return num,den

    def pGB_DH0(self, H0, mth, skyprob, zcut=10.):
        """
        Evaluate p(G|D,H0) and p(B|D,H0).

        The probability that the host galaxy of a detected GW event is inside
        or beyond the galaxy catalogue.

        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        mth : float
            Apparent magnitude threshold
        skyprob : float, optional
            pdet probability covered by area G(B) (default=1.)
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=10.)

        Returns
        -------
        floats
            p(G|D,H0), p(B|D,H0), num, den
            where (num/den)*skyprob = p(G|D,H0)
        """

        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)

        num = dblquad(self.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: Mmin,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax),args=
                      [H0],epsabs=0,epsrel=1.49e-4)[0]

        den = dblquad(self.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,self.zmax,lambda x: Mmin,lambda x: Mmax,args=[H0],epsabs=0,epsrel=1.49e-4)[0]
        integral = num/den

        pG = integral*skyprob
        pB = (1.-integral)*skyprob
        return pG, pB, num, den

    def px_BH0(self, H0, mth, skyprob ,zcut=10.):
        """
        Evaluate p(x|B,H0).

        If zcut >= zmax then a single integral is performed.
        If zcut < zmax then an additional integral is performed.

        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        mth : float
            Apparent magnitude threshold
        skyprob : float, optional
            GW sky probability covered by area G(B) (default=1.)
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=10.)

        Returns
        -------
        float
            p(x|B,H0)
        """

        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)

        below_zcut_integral = dblquad(self.px_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax),
                                      lambda x: Mmax,args=[H0],epsabs=0,epsrel=1.49e-4)[0]

        above_zcut_integral = 0.
        if zcut < self.zmax:
            above_zcut_integral = dblquad(self.px_zH0_times_pz_times_ps_z_times_pM_times_ps_M,zcut,self.zmax,lambda x: Mmin, lambda x: Mmax,args=[H0],
                                                                                                                                epsabs=0,epsrel=1.49e-4)[0]

        integral = below_zcut_integral + above_zcut_integral

        return integral * skyprob

    def pD_BH0(self, H0, mth, skyprob, zcut=10.):
        """
        Evaluate p(D|B,H0).

        If zcut >= zmax then a single integral is performed.
        If zcut < zmax then an additional integral is performed.

        Parameters
        ----------
        H0 : float
            Hubble constant value in kms-1Mpc-1
        mth : float
            Apparent magnitude threshold
        skyprob : float, optional
            pdet probability covered by area G(B) (default=1.)
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=10.)

        Returns
        -------
        float
            p(D|B,H0)
        """

        Mmin = M_Mobs(H0,self.Mmin_obs)
        Mmax = M_Mobs(H0,self.Mmax_obs)

        below_zcut_integral = dblquad(self.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,0.,zcut,lambda x: min(max(M_mdl(mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax),
                                      lambda x: Mmax,args=[H0],epsabs=0,epsrel=1.49e-4)[0]

        above_zcut_integral = 0.
        if zcut < self.zmax:
            above_zcut_integral = dblquad(self.pD_zH0_times_pz_times_ps_z_times_pM_times_ps_M,zcut,self.zmax,lambda x: Mmin, lambda x: Mmax,args=[H0],
                                          epsabs=0,epsrel=1.49e-4)[0]

        integral = below_zcut_integral + above_zcut_integral

        return integral * skyprob


class SinglePixelGalaxyCatalogLikelihood(GalaxyCatalogLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the galaxy
    catalogue method.

    Parameters
    ----------
    pixel_index : Index of the healpy pixel to analyse
    galaxy_catalog : gwcosmo.prior.catalog.galaxyCatalog object
        The galaxy catalogue
    skymap : gwcosmo.likelihood.skymap.skymap object
        provides p(x|Omega) and skymap properties
    fast_cosmology : gwcosmo.utilities.standard_cosmology.fast_cosmology object
        Cosmological model
    Kcorr : bool, optional
        Should K corrections be applied to the analysis? (default=False)
        Will raise an error if used in conjunction with a galaxy catalogue
        without sufficient color information.
    mth : float, optional
        Specify an apparent magnitude threshold for the galaxy catalogue
        (default=None). If none, mth is estimated from the galaxy catalogue.
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=None)
    zmax : float, optional
        The upper redshift limit for integrals (default=10.). Should be well
        beyond the highest redshift reachable by GW data or selection effects.
    zuncert : bool, optional
        Should redshift uncertainties be marginalised over? (Default=True).

    """

    def __init__(self, pixel_index, galaxy_catalog, skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=False, mth=None, zcut=None, zmax=10.,zuncert=True, complete_catalog=False, nside=32, nside_low_res = None):
        """
        Parameters
        ----------
        pixel_index : int
            The healpy index of the pixel being analysed (assuming low-res nside)
        galaxy_catalog : object
            The galaxy catalogue
        skymap : object
            The GW skymap
        observation_band : str
            Observation band (eg. 'B', 'K', 'u', 'g')
        fast_cosmology : object
            Fast cosmology
        px_zH0 : object
            GW data, p(x|z,H0)
        pD_zH0 : object
            probability of detection, p(D|z,H0)
        zprior : object
            redshift prior, p(z)
        zrates : object
            rate evolution function, p(s|z)
        luminosity_prior : object
            absolute magnitude prior, p(M|H0)
        luminosity_weights : object
            luminosity weighting function, p(s|M)
        Kcorr : bool, optional
            Should K corrections be applied to the analysis? (default=False)
        mth : float, optional
            Specify an apparent magnitude threshold for the galaxy catalogue
            (default=None). If none, mth is estimated from the galaxy catalogue.
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=None)
        zmax : float, optional
            The upper redshift limit for the universe (default=10.)
        zuncert : bool, optional
            Should redshift uncertainties be marginalised over? (Default=True)
        complete_catalog : bool, optional
            is the galaxy catalogue already complete? (Default=False)
        nside : int, optional
            The high-resolution value of nside to subdivide the current pixel into
        """

        super().__init__(skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=Kcorr, zmax=zmax)
        self.nside_low_res = nside_low_res
        self.zcut = zcut
        self.complete_catalog = complete_catalog
        self.full_catalog = galaxy_catalog

        # Set redshift and colour limits based on whether Kcorrections are applied
        if Kcorr == True:
            if zcut is None:
                self.zcut = 0.5
            self.full_catalog = self.full_catalog.apply_color_limit(observation_band,
                                                                    *color_limits[color_names[observation_band]])
        else:
            if zcut is None:
                self.zcut = self.zmax
            self.color_limit = [-np.inf,np.inf]

        #TODO make this changeable from command line?
        self.nfine = 10000
        self.ncoarse = 10

        self.hi_res_nside = nside
        # Get the coordinates of the hi-res pixel centres
        pixra, pixdec = ra_dec_from_ipix(self.hi_res_nside, np.arange(hp.pixelfunc.nside2npix(self.hi_res_nside)), nest=skymap.nested)
        # compute the low-res index of each of them
        ipix = ipix_from_ra_dec(nside_low_res, pixra, pixdec, nest=skymap.nested)
        # Keep the ones that are within the current coarse pixel
        self.sub_pixel_indices = np.arange(hp.pixelfunc.nside2npix(self.hi_res_nside))[np.where(ipix == pixel_index)]
        print(f'Pixel {pixel_index} at nside={nside_low_res} contains pixels {self.sub_pixel_indices} at nside={self.hi_res_nside}')
        if zuncert == False:
            self.nfine = 1
            self.ncoarse = 1
            self.galsigmaz = np.zeros(len(self.galz))

        self.pxG = 0.
        self.pDG = 1.
        self.pG = 1.
        self.pxB = 0.
        self.pDB = 1.
        self.pB = 0.
        self.pxO = 0.
        self.pDO = 1.
        self.pO = 0.

        self.pixel_area_hi_res = 1./hp.nside2npix(self.hi_res_nside)

        hi_res_skyprob = hp.pixelfunc.ud_grade(skymap.prob, self.hi_res_nside, order_in='NESTED', order_out='NESTED')
        self.hi_res_skyprob = hi_res_skyprob/np.sum(hi_res_skyprob) #renormalise

        self.mth_map={}
        for i, idx in enumerate(self.sub_pixel_indices):
            ra, dec = ra_dec_from_ipix(self.hi_res_nside, idx)
            pix_catalog = galaxy_catalog.select_pixel(self.hi_res_nside, idx)
            self.mth_map[i] = pix_catalog.magnitude_thresh(observation_band, ra, dec)

    def full_pixel(self, H0, z, sigmaz, m, ra, dec, color, mth, px_Omega=1., pOmega=1.):
        """
        Compute the full likelihood on H0 for a single (sub-)pixel.

        Parameters
        ----------
        H0 : array of floats
            Hubble constant values in kms-1Mpc-1
        complete_catalog : bool, optional
            Assume that the galaxy catalogue is complete? (default=False)

        Returns
        -------
        float
            Returns likelihood, pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO
            where likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO
        """

        pG = np.ones(len(H0))
        pxB = np.zeros(len(H0))
        pDB = np.ones(len(H0))
        pB = np.zeros(len(H0))

        num = np.zeros(len(H0))
        den = np.zeros(len(H0))

        print(f'Computing the in-catalogue part with {len(m)} galaxies')
        pxG, pDG = self.pxD_GH0_multi(H0, z, sigmaz, m, ra, dec, color, nfine=self.nfine, ncoarse=self.ncoarse, zcut=self.zcut)

        if not self.complete_catalog:
            print('Computing the beyond catalogue part')
            for i,h in enumerate(H0):
                pG[i], pB[i], num[i], den[i] = self.pGB_DH0(h, mth, pOmega, zcut=self.zcut)
                pxB[i] = self.px_BH0(h, mth, px_Omega, zcut=self.zcut)
            if self.zcut == self.zmax:
                pDB = (den - num) * pOmega
            else:
                print('Computing all integrals explicitly as zcut < zmax: this will take a little longer')
                for i,h in enumerate(H0):
                    pDB[i] = self.pD_BH0(h, mth, pOmega, zcut=self.zcut)

        likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB
        return pxG, pDG, pG, pxB, pDB, pB

    def likelihood(self,H0):
        """
        Compute the likelihood on H0 for a single pixel

        Parameters
        ----------
        H0 : array of floats
            Hubble constant values in kms-1Mpc-1

        Returns
        -------
        float
            likelihood
        """

        self.pxG = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pDG = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pG = np.zeros([len(H0),len(self.sub_pixel_indices)])

        self.pxB = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pDB = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pB = np.zeros([len(H0),len(self.sub_pixel_indices)])

        self.pxO = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pDO = np.zeros([len(H0),len(self.sub_pixel_indices)])
        self.pO = np.zeros(len(self.sub_pixel_indices))

        if inf in self.mth_map.values():
            temp_pxO = np.zeros(len(H0))
            temp_pDO = np.zeros(len(H0))
            print('Not all of this pixel has catalogue support. Computing the out of catalogue contribution')
            for i,h in enumerate(H0):
                temp_pxO[i] = self.px_OH0(h, skyprob=1.)
                temp_pDO[i] = self.pD_OH0(h, skyprob=1.)

        # loop over sub-pixels
        for i, idx in enumerate(self.sub_pixel_indices):
            px_Omega = self.hi_res_skyprob[idx]
            ra, dec = ra_dec_from_ipix(self.hi_res_nside, idx)
            subcatalog = self.full_catalog.select_pixel(self.hi_res_nside, idx)
            # TODO: Does this need a coarse pixel for better estimation?
            mth = self.mth_map[i]
            #mth = subcatalog.magnitude_thresh(self.band, ra, dec)

            if mth == np.inf:
                self.pxO[:,i] = temp_pxO*px_Omega
                self.pDO[:,i] = temp_pDO*self.pixel_area_hi_res
                self.pO[i] = self.pixel_area_hi_res

                self.pxG[:,i] = np.zeros(len(H0))
                self.pDG[:,i] = np.ones(len(H0))
                self.pG[:,i] = np.zeros(len(H0))
                self.pxB[:,i] = np.zeros(len(H0))
                self.pDB[:,i] = np.ones(len(H0))
                self.pB[:,i] = np.zeros(len(H0))
            else:
                # Apply cuts to catalog
                subcatalog = subcatalog.apply_redshift_cut(self.zcut)
                if color_names[self.band] is None:
                    clim = [-inf, inf]
                else:
                    clim = color_limits[color_names[self.band]]
                subcatalog = subcatalog.apply_color_limit(self.band, clim[0], clim[1]).apply_magnitude_limit(self.band, mth)

                print('mth in this sub-pixel: {}'.format(mth))
                print('Ngal in this sub-pixel: {}'.format(len(subcatalog)))
                z = subcatalog['z']
                ra = subcatalog['ra']
                dec = subcatalog['dec']
                m = subcatalog.get_magnitudes(self.band)
                sigmaz = subcatalog['sigmaz']
                color = subcatalog.get_color(self.band) # TODO: Fix based on Kcorr

                if len(subcatalog)==0:
                    self.pxG[:,i] = np.zeros(len(H0))
                else:
                    self.pxG[:,i], self.pDG[:,i], self.pG[:,i], self.pxB[:,i], self.pDB[:,i], self.pB[:,i] = self.full_pixel(H0, z, sigmaz, m, ra, dec, color, self.mth_map[i], px_Omega=px_Omega, pOmega=self.pixel_area_hi_res)
                self.pxO[:,i] = np.zeros(len(H0))
                self.pDO[:,i] = np.ones(len(H0))
                self.pO[i] = 0.

        sub_likelihood = np.zeros([len(H0),len(self.sub_pixel_indices)])
        for i in range(len(self.sub_pixel_indices)):
            sub_likelihood[:,i] = (self.pxG[:,i] / self.pDG[:,i]) * self.pG[:,i] + (self.pxB[:,i] / self.pDB[:,i]) * self.pB[:,i] + (self.pxO[:,i] / self.pDO[:,i]) * self.pO[i]
        likelihood = np.sum(sub_likelihood,axis=1)
        return likelihood

    def return_components(self):
        """
        Returns pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO
        where each contains values for i sub-pixels
        and likelihood = sum_i[(pxG_i / pDG_i) * pG_i + (pxB_i / pDB_i) * pB_i + (pxO_i / pDO_i) * pO_i]
        """

        return self.pxG, self.pDG, self.pG, self.pxB, self.pDB, self.pB, self.pxO, self.pDO, self.pO

    def __call__(self, H0):
        return self.likelihood(H0)


class WholeSkyGalaxyCatalogLikelihood(GalaxyCatalogLikelihood):
    """
    Calculate the likelihood on H0 from one GW event, using the galaxy
    catalogue method.
    """

    def __init__(self, galaxy_catalog, skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=False, mth=None, zcut=None, zmax=10.,zuncert=True, complete_catalog=False, sky_thresh = 0.999):
        """
        Parameters
        ----------
        galaxy_catalog : object
            The galaxy catalogue
        skymap : object
            The GW skymap
        observation_band : str
            Observation band (eg. 'B', 'K', 'u', 'g')
        fast_cosmology : object
            Fast cosmology
        px_zH0 : object
            GW data, p(x|z,H0)
        pD_zH0 : object
            probability of detection, p(D|z,H0)
        zprior : object
            redshift prior, p(z)
        zrates : object
            rate evolution function, p(s|z)
        luminosity_prior : object
            absolute magnitude prior, p(M|H0)
        luminosity_weights : object
            luminosity weighting function, p(s|M)
        Kcorr : bool, optional
            Should K corrections be applied to the analysis? (default=False)
        mth : float, optional
            Specify an apparent magnitude threshold for the galaxy catalogue
            (default=None). If none, mth is estimated from the galaxy catalogue.
        zcut : float, optional
            An artificial redshift cut to the galaxy catalogue (default=None)
        zmax : float, optional
            The upper redshift limit for the universe (default=10.)
        zuncert : bool, optional
            Should redshift uncertainties be marginalised over? (Default=True)
        complete_catalog : bool, optional
            is the galaxy catalogue already complete? (Default=False)

        TODO: UPDATE this for new catalog classes
        """

        super().__init__(skymap, observation_band, fast_cosmology, px_zH0, pD_zH0, zprior, zrates, luminosity_prior, luminosity_weights, Kcorr=Kcorr, zmax=zmax)

        self.mth = mth
        self.zcut = zcut
        self.complete_catalog = complete_catalog
        self.catalog = galaxy_catalog

        # Set redshift and colour limits based on whether Kcorrections are applied
        if Kcorr == True:
            if zcut is None:
                self.zcut = 0.5
            self.catalog = self.catalog.apply_color_limit(observation_band,
                                                          *color_limits[color_names[observation_band]])
        else:
            if zcut is None:
                self.zcut = self.zmax
            self.color_limit = [-np.inf,np.inf]

        if mth is None:
            mth = self.catalog.magnitude_thresh(observation_band)

        print('Catalogue apparent magnitude threshold: {}'.format(self.mth))

        #TODO make this changeable from command line?
        self.nfine = 10000
        self.ncoarse = 10

        self.nGal = len(self.catalog)

        if zuncert == False:
            self.nfine = 1
            self.ncoarse = 1
            self.galsigmaz = np.zeros(len(self.galz))

        self.OmegaG, self.px_OmegaG = skymap.region_with_sample_support(self.catalog['ra'],
                                                                       self.catalog['dec'],
                                                                       sky_thresh)
        self.OmegaO = 1. - self.OmegaG
        self.px_OmegaO = 1. - self.px_OmegaG

        self.pxG = None
        self.pDG = None
        self.pG = 1.
        self.pxB = 0.
        self.pDB = 1.
        self.pB = 0.
        self.pxO = 0.
        self.pDO = 1.
        self.pO = 0.

    def likelihood(self,H0):
        """
        Compute the likelihood on H0

        Parameters
        ----------
        H0 : array of floats
            Hubble constant values in kms-1Mpc-1

        Returns
        -------
        float
            likelihood
        """

        self.pG = np.ones(len(H0))
        self.pxB = np.zeros(len(H0))
        self.pDB = np.ones(len(H0))
        self.pB = np.zeros(len(H0))
        self.pxO = np.zeros(len(H0))
        self.pDO = np.ones(len(H0))
        self.pO = 0.

        num = np.zeros(len(H0))
        den = np.zeros(len(H0))

        # Apply cuts to catalog
        subcatalog = self.catalog.apply_redshift_cut(self.zcut)
        if color_names[self.band] is None:
            clim = [-inf, inf]
        else:
            clim = color_limits[color_names[self.band]]
            subcatalog = subcatalog.apply_color_limit(self.band, clim[0], clim[1]).apply_magnitude_limit(self.band, mth)

            print('mth in this sub-pixel: {}'.format(mth))
            print('Ngal in this sub-pixel: {}'.format(len(subcatalog)))
        galz = subcatalog['z']
        galra = subcatalog['ra']
        galdec = subcatalog['dec']
        galm = subcatalog.get_magnitudes(self.band)
        galsigmaz = subcatalog['sigmaz']
        galcolor = subcatalog.get_color(self.band) # TODO: Fix based on Kcorr
        print('Computing the in-catalogue part')
        self.pxG, self.pDG = self.pxD_GH0_multi(H0, galz, galsigmaz, galm, galra,
                                                galdec, galcolor, nfine=self.nfine,
                                                ncoarse=self.ncoarse, zcut=self.zcut)

        if not self.complete_catalog:
            print('Computing the beyond catalogue part')
            for i,h in enumerate(H0):
                self.pG[i], self.pB[i], num[i], den[i] = self.pGB_DH0(h, self.mth, self.OmegaG,zcut=self.zcut)
                self.pxB[i] = self.px_BH0(h, self.mth, self.px_OmegaG, zcut=self.zcut)
            if self.zcut == self.zmax:
                self.pDB = (den - num) * self.OmegaG
            else:
                print('Computing all integrals explicitly as zcut < zmax: this will take a little longer')
                for i,h in enumerate(H0):
                    self.pDB[i] = self.pD_BH0(h, self.mth, self.OmegaG, zcut=self.zcut)
            print("{}% of this event's sky area appears to have galaxy catalogue support".format(self.px_OmegaG*100))
            if self.px_OmegaG < 0.999:
                self.pO = self.OmegaO
                #self.pDO = den * self.OmegaO ### alternative to calculating pDO directly below, but requires both px_OH0 and pD_OH0 to use dblquad (not quad) ###
                print('Computing the contribution outside the catalogue footprint')
                for i,h in enumerate(H0):
                    self.pxO[i] = self.px_OH0(h, skyprob=self.px_OmegaO)
                    self.pDO[i] = self.pD_OH0(h, skyprob=self.OmegaO)

        likelihood = (self.pxG / self.pDG) * self.pG + (self.pxB / self.pDB) * self.pB + (self.pxO / self.pDO) * self.pO
        return likelihood

    def return_components(self):
        """
        Returns pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO
        where likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO
        """

        return self.pxG, self.pDG, self.pG, self.pxB, self.pDB, self.pB, self.pxO, self.pDO, self.pO

    def __call__(self, H0):
        return self.likelihood(H0)



class DirectCounterpartLikelihood(gwcosmoLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the counterpart
    method.

    This method is fast relative to the catalog methods, as it does not
    require an integral over either sky or absolute magnitude, only redshift.
    """

    def __init__(self, counterpart_z,counterpart_sigmaz, px_zH0, pD_zH0, zprior, zrates, zmax=10.):
        """
        Parameters
        ----------
        counterpart_z : float
            redshift of EM counterpart
        counterpart_sigmaz : float
            1 sigma uncertainty on redshift of EM counterpart
        px_zH0 : object
            GW data, p(x|z,H0) along LOS of counterpart
        pD_zH0 : object
            probability of detection, p(D|z,H0)
        zprior : object
            redshift prior, p(z)
        zrates : object
            rate evolution function, p(s|z)
        zmax : float
            The upper redshift limit for the universe (default=10.)
        """

        self.counterpart_z = counterpart_z
        self.counterpart_sigmaz = counterpart_sigmaz
        super().__init__(px_zH0, pD_zH0, zprior, zrates, zmax=zmax)

        self.px = None
        self.pD = None

    def px_H0(self,H0):
        """
        Returns p(x|H0,counterpart)
        The likelihood of the GW data given H0 and direct counterpart.

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(x|H0,counterpart)
        """
        zsmear =  z_nsmear(self.counterpart_z, self.counterpart_sigmaz, 10000)
        num = np.zeros(len(H0))
        for k,h in enumerate(H0):
            num[k] = np.sum(self.px_zH0(zsmear,h)*self.zrates(zsmear))
        return num

    def likelihood(self,H0):
        """
        Compute the likelihood on H0

        Parameters
        ----------
        H0 : array of floats
            Hubble constant values in kms-1Mpc-1

        Returns
        -------
        float
            likelihood
        """

        px = self.px_H0(H0)
        pD = np.zeros(len(H0))
        for i,h in enumerate(H0):
            pD[i] = self.pD_OH0(h, skyprob=1.)
        likelihood = px/pD
        self.px = px
        self.pD = pD
        return likelihood

    def return_components(self):
        """
        Returns pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO
        where likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO
        """
        return self.px, self.pD, 1., 0., 1., 0., 0., 1., 0.

    def __call__(self, H0):
        return self.likelihood(H0)



class EmptyCatalogLikelihood(gwcosmoLikelihood):
    """
    Calculate the likelihood of H0 from one GW event, using the empty catalogue
    method.

    Calculations assuming no EM data (either counterpart or catalog).
    All information comes from the distance distribution of GW events
    or population assumptions which have not yet been marginalized over.

    This method is fast relative to the catalog methods, as it does not
    require an integral over either sky or absolute magnitude, only redshift.
    """

    def __init__(self, px_zH0, pD_zH0, zprior, zrates, zmax=10.):
        """
        Parameters
        ----------
        px_zH0 : object
            GW data, p(x|z,H0)
        pD_zH0 : object
            probability of detection, p(D|z,H0)
        zprior : object
            redshift prior, p(z)
        zrates : object
            rate evolution function, p(s|z)
        zmax : float
            The upper redshift limit for the universe (default=10.)
        """

        super().__init__(px_zH0, pD_zH0, zprior, zrates, zmax=zmax)

        self.px = None
        self.pD = None

    def likelihood(self,H0):
        """
        Compute the likelihood on H0

        Parameters
        ----------
        H0 : array of floats
            Hubble constant values in kms-1Mpc-1

        Returns
        -------
        float
            likelihood
        """

        px = np.zeros(len(H0))
        pD = np.zeros(len(H0))
        for i,h in enumerate(H0):
            px[i] = self.px_OH0(h, skyprob=1.)
            pD[i] = self.pD_OH0(h, skyprob=1.)
        likelihood = px/pD
        self.px = px
        self.pD = pD
        return likelihood

    def return_components(self):
        """
        Returns pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO
        where likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO
        """
        return 0., 1., 0., 0., 1., 0., self.px, self.pD, 1.

    def __call__(self, H0):
        return self.likelihood(H0)



################################################################################
################################ ADDITIONAL CLASSES ############################
################################################################################

class LuminosityWeighting(object):
    """
    Host galaxy probability relation to luminosity: proportional to luminosity
    """

    def __init__(self):
        self.luminosity_weights = True

    def weighted_call(self, M):
        """
        Luminosity weighting

        Parameters
        ----------
        M : float
            absolute magnitude

        Returns
        -------
        float
            Luminosity
        """

        return L_M(M)

    def __call__(self, M):
        return self.weighted_call(M)

class UniformWeighting(object):
    """
    Host galaxy probability relation to luminosity: uniform
    """

    def __init__(self):
        self.luminosity_weights = False

    def unweighted_call(self, M):
        """
        Uniform weighting

        Parameters
        ----------
        M : float
            absolute magnitude

        Returns
        -------
        float
            1.
        """
        return 1.

    def __call__(self, M):
        return self.unweighted_call(M)


class RedshiftEvolutionMadau():
    """
    Merger rate relation to redshift: Madau model
    """

    def __init__(self,hyper_params_evolution):
        """
        Parameters
        ----------
        hyper_params_evolution : dict
            dictionary of redshift evolution parameters
        """
        self.redshift_evolution = True
        self.Lambda, self.beta, self.zp = hyper_params_evolution['Lambda'], hyper_params_evolution['madau_beta'], hyper_params_evolution['madau_zp']

    def evolving(self, z):
        """
        Madau rate evolution

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """

        C = 1+(1+self.zp)**(-self.Lambda-self.beta)
        return C*((1+z)**self.Lambda)/(1+((1+z)/(1+self.zp))**(self.Lambda+self.beta)) #Equation 2 in https://arxiv.org/pdf/2003.12152.pdf

    def __call__(self, z):
        """
        Madau rate evolution, shifted to detector frame with
        additional factor of 1/(1+z)

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """
        return self.evolving(z)/(1.+z)



class RedshiftEvolutionPowerLaw():
    """
    Merger rate relation to redshift: power-law model
    """

    def __init__(self,hyper_params_evolution):
        """
        Parameters
        ----------
        hyper_params_evolution : dict
            dictionary of redshift evolution parameters
        """

        self.redshift_evolution = True
        self.Lambda = hyper_params_evolution['Lambda']

    def evolving(self, z):
        """
        Power-law rate evolution

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """
        return (1+z)**self.Lambda

    def __call__(self, z):
        """
        Power-law rate evolution, shifted to detector frame with
        additional factor of 1/(1+z)

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """
        return self.evolving(z)/(1.+z)

class RedshiftNonEvolution():
    """
    Merger rate relation to redshift: no evolution
    """

    def __init__(self,hyper_params_evolution):
        """
        Parameters
        ----------
        hyper_params_evolution : dict
            dictionary of redshift evolution parameters
        """

        self.redshift_evolution = False

    def constant(self, z):
        """
        No rate evolution

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """
        return 1.

    def __call__(self, z):
        """
        No rate evolution, but GW signals are shifted to detector frame with
        additional factor of 1/(1+z)

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
        """
        return self.constant(z)/(1.+z)


################################################################################
################################ INTERNAL FUNCTIONS ############################
################################################################################


def z_nsmear(z, sigmaz, nsmear, zcut=10.):
    """
    Draw redshift samples from a galaxy.

    Ensure no samples fall below z=0.
    Remove samples above the redshift cut. zcut cannot be used as an upper limit
    for the draw, as this will cause an overdensity of support.

    Parameters
    ----------
    z : float
        redshift
    sigmaz : float
        1 sigma redshift uncertainty
    nsmear : float
        number of samples to take from galaxy
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=10.)

    Returns
    -------
    array of floats
        nsmear redshift samples (excluding those above zcut)
    """
    a = (0.0 - z) / sigmaz
    zsmear = truncnorm.rvs(a, 5, loc=z, scale=sigmaz, size=nsmear)
    zsmear = zsmear[np.where(zsmear<zcut)[0]].flatten()
    return zsmear #TODO order these before returning them?


def gal_nsmear(z, sigmaz, m, ra, dec, color, nsmear, zcut=10.):
    """
    Draw redshift samples from a set of galaxies.
    Ensure no samples fall below z=0
    Remove samples above the redshift cut. zcut cannot be used as an upper limit
    for the draw, as this will cause an overdensity of support.

    Parameters
    ----------
    z : array of floats
        galaxy redshifts
    sigmaz : array of floats
        1 sigma redshift uncertainty
    m : array of floats
        galaxy apparent magnitudes
    ra, dec : array of floats
        galaxy right ascension and declinations
    color : array of floats
        galaxy colors
    nsmear : float
        number of samples to take from all galaxy
    zcut : float, optional
        An artificial redshift cut to the galaxy catalogue (default=10.)

    Returns
    -------
    array
        sampz, sampm, sampra, sampdec, sampcolor, count
    """

    # get redshift samples, carefully not going below zero
    a = (0.0 - z) / sigmaz
    sampz = truncnorm.rvs(a, 5, loc=z, scale=sigmaz, size=[nsmear,len(z)]).flatten('F')

    # repeat arrays for other gal parameters to give each sample full info
    sampcolor = np.repeat(color,nsmear)
    sampm = np.repeat(m,nsmear)
    sampra = np.repeat(ra,nsmear)
    sampdec = np.repeat(dec,nsmear)
    count = np.ones(len(sampz))*nsmear

    # remove samples above the redshift cut
    ind = np.where(sampz < zcut)[0]
    sampz = sampz[ind]
    sampcolor = sampcolor[ind]
    sampm = sampm[ind]
    sampra = sampra[ind]
    sampdec = sampdec[ind]
    count = count[ind]

    # sort array in ascending order so that px_zH0 and pD_zH0 don't freak out
    ind = np.argsort(sampz)
    sampz = sampz[ind]
    sampcolor = sampcolor[ind]
    sampm = sampm[ind]
    sampra = sampra[ind]
    sampdec = sampdec[ind]
    count = count[ind]

    return sampz, sampm, sampra, sampdec, sampcolor, count


