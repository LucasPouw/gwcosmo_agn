"""Module containing functionality for creation and management of galaxy catalogs."""

import numpy as np
from astropy.table import Table
from astropy import units as u
import astropy.constants as constants
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
import pkg_resources

# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo', 'prior/catalog_data')
groupmembers4993 = np.genfromtxt(catalog_data_path + "/NGC4993group.txt", usecols=0) #NGC 4993's group
pgc_KTgroups,groupgal = np.genfromtxt(catalog_data_path + "/KTgroups.txt", usecols=(0,4), unpack=True)

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.27)
apex_helio_to_3k = Galactic(l=264.14*u.degree, b=48.26*u.degree, radial_velocity=371.0*u.km/u.s)

z_err_fraction = 0.06
a_err_fraction = 0.08
h = 0.7
L0 = 3.0128e28
Lsun = 3.828e26
Lstar = 1.2e10*h**(-2)*Lsun

# Set Bmag cut
lum_cut_frac = 0.0

# Set to 1.0 to use NGC4993 only, else set to 0.0
useNGC4993only = 0.0

# Pass through a pre-selected catalog
usePreselectedCat = 0.0

# Global Methods
def vcorr(gc,apex):
    b = gc.b
    l = gc.l
    v = gc.radial_velocity
    bapex = apex.b
    lapex = apex.l
    vapex = apex.radial_velocity
    return (v + vapex*( np.sin(b)*np.sin(bapex) + np.cos(b)*np.cos(bapex)*np.cos(l-lapex) ))

def blue_luminosity_from_mag(m,d):
    """
    Returns the blue luminosity in units of L_10 given the apparent
    magnitude and the luminosity distance
    """
    M_blue_solar = 5.48 # Binney & Tremaine
    MB = m - 5.0 * np.log10( d / 10.0e-6 )
    lum = np.power( 10, (M_blue_solar - MB)/2.5 - 10.0 )
    if lum_cut_frac > 0.0:
        lumCut, idx = lum_cut(lum,lum_cut_frac)
    else: 
        lumCut = lum
        idx = np.arange(len(lumCut))   
    return lumCut, idx

def replace_bad_magnitudes(aa):
    maa = np.ma.masked_values(aa,-999.)
    return maa.filled(1e20).data

def lum_cut(aa,lum_cut_value):
    lumidx = np.where(aa>lum_cut_value*Lstar)
    return aa[lumidx], lumidx

class galaxy(object):
    ''' Class for galaxy objects
    '''
    def __init__(self,
                index = 0,
                astropy_row = 0,
                pgc_number = 0,
                galaxy_name = 0,
                cluster = 0,
                ra = 0,
                dec = 0,
                z = 0,
                z_err = 0,
                distance = 0,
                distance_error = 0,
                angle_error = 0,
                abs_mag_r = 0,
                abs_mag_k = 0,
                lumB = 0,
                lumK=0):
        """Galaxy catalog class... 
        Parameters
        """
        self.index = index
        self.astropy_row = astropy_row    
        self.pgc_number = pgc_number
        self.galaxy_name = galaxy_name
        self.cluster = cluster
        self.ra = ra
        self.dec = dec
        self.z = z
        self.z_err = z_err
        self.distance = distance
        self.distance_error = distance_error
        self.angle_error = angle_error
        self.abs_mag_r = abs_mag_r
        self.abs_mag_k = abs_mag_k
        self.lumB = lumB
        self.lumK = lumK
        
    #@staticmethod
    def load_astropy_row(self, index, row):
        self.index = index
        self.astropy_row = row
        self.pgc_number = row['PGC']
        self.galaxy_name = row['Galaxy Name']
        self.cluster = row['Cluster']
        self.ra = row['RA']
        self.dec = row['Dec']
        self.z = row['z']
        self.distance = row['Distance']
        self.distance_error = row['distance_error']
        self.abs_mag_r = row['abs_mag_r']
        self.abs_mag_k = row['abs_mag_k']


class catalog(object):
    ''' Class for galaxy catalog objects
    '''
    def __init__(self,
                catalog_file = catalog_data_path + "/gladecatalogv2.3.dat",
                catalog_format = 'ascii',
                catalog = 0,
                pgc_number = 0,
                galaxy_name = 0,
                ra = 0,
                dec = 0,
                z = 0,
                z_err = 0,
                distance = 0,
                distance_error = 0,
                angle_error = 0,
                lumB = 0):
        """Galaxy catalog class... 
        Parameters
        """
        self.catalog_file = catalog_file
        self.catalog_format = catalog_format
        self.catalog = catalog
        self.pgc_number = pgc_number
        self.galaxy_name = galaxy_name
        self.ra = ra
        self.dec = dec
        self.z = z
        self.z_err = z_err
        self.distance = distance
        self.distance_error = distance_error
        self.angle_error = angle_error
        self.lumB = lumB

    def load_catalog_galaxy_objects(self):
        t = Table.read(self.catalog_file,format=self.catalog_format)
        nGalaxies = len(t)
        galaxies = {}
        for k in range(0, nGalaxies):
            galaxies[str(k)] == t[k]

    def load_catalog(self):
        self.catalog = Table.read(self.catalog_file,format=self.catalog_format)

    def load_catalog_counterpart(self):
        t = Table.read(self.catalog_file,format=self.catalog_format)
        t = t[np.where(t['Galaxy Name']=='NGC4993')]
        self.catalog = t

    def remove_clusters(self):
        t = self.catalog
        gc_flag = t['Cluster']
        no_gc_sel = gc_flag == 'C'
        t.remove_rows(no_gc_sel)
        self.catalog = t

    def remove_nans(self):
        t = self.catalog
        nan_flag = t['abs_mag_r']
        no_nan_sel = np.isnan(nan_flag)
        t.remove_rows(no_nan_sel)
        self.catalog = t

    def group_velocity_corr(self):
        # group correction (from Maya)
        allgroups = np.unique(groupgal)
        groupmembersKT = [[] for i in range(len(allgroups))]
        pgcCat = self.pgc_number
        zsel_group = self.z
        for i in range(len(allgroups)):
            for j,g in enumerate(groupgal):
                if g==allgroups[i]:
                    groupmembersKT[i].append(pgc_KTgroups[j])

        zs_group = [[] for i in range(len(allgroups))]
        for i in range(len(allgroups)):
            for j, pgc in enumerate(pgcCat):
                if pgc in groupmembersKT[i]:
                    zs_group[i].append(zsel_group[j])
        zs_groupaverage = [np.mean(zs_group[i]) for i in range(len(allgroups))]

        for i in range(len(allgroups)):
            for j, pgc in enumerate(pgcCat):
                if pgc in groupmembersKT[i]:
                    zsel_group[j] = zs_groupaverage[i]
        ###NGC 4993 group correction (just to match the same p.v. used for counterpart case, this is not necessary)
        for i, pgc in enumerate(pgcCat):
            if pgc in groupmembers4993:
                zsel_group[i] = 3017./constants.c.to('km/s').value
        return zsel_group 

    def extract_galaxies(self):
        tt = self.catalog
        pgcCat = tt['PGC']
        ra = tt['RA']
        ra = np.pi * ra / 180.0
        
        dec = tt['Dec']
        dec = np.pi * dec / 180.0
        
        z = tt['z']
        dist = tt['Distance']

        angle_error = a_err_fraction / ( dist )
        
        dist_err = tt['Distance Error']
        fractional_err = np.ma.masked_values(dist_err,-999.) / dist
        dist_err = dist * fractional_err.filled(z_err_fraction).data

        z_err = z * fractional_err.filled(z_err_fraction).data
        
        B = replace_bad_magnitudes( tt['abs_mag_r'] )
        lumB, idx = blue_luminosity_from_mag(B,dist)
        pgcCat=pgcCat[idx]
        ra = ra[idx]
        dec = dec[idx]
        dist = dist[idx]
        z = z[idx]
        lumB = lumB[idx]
        angle_error = angle_error[idx]
        dist_err = dist_err[idx]
        z_err = z_err[idx]
        # hack to weight the extra galaxies at 11.429 with zero blue luminosity
        lumB = np.ma.masked_where( ((dist>11.4291) & (dist<11.4293)) , lumB).filled(0)   
        
        # hack to remove deal with Galactic Center and SN1987a
        if useNGC4993only > 0:
            glist = []
        else:
            glist = ['GALCENTER','ESO056-115']
        for g in glist:
            delete_i = np.argmax(tt['Galaxy Name']==g)
            if delete_i:
                lumB[delete_i]=0
        
        return (pgcCat,ra,dec,dist,z,lumB,angle_error,dist_err,z_err)

    def extract_galaxies_counterpart(self):
        tt = self.catalog
        pgcCat = tt['PGC']
        ra = tt['RA']
        ra = np.pi * ra / 180.0
        
        dec = tt['Dec']
        dec = np.pi * dec / 180.0
        
        z = tt['z']
        dist = tt['Distance']

        angle_error = a_err_fraction / ( dist )
        
        dist_err = tt['Distance Error']
        fractional_err = np.ma.masked_values(dist_err,-999.) / dist
        dist_err = dist * fractional_err.filled(z_err_fraction).data

        z_err = z * fractional_err.filled(z_err_fraction).data
        
        B = replace_bad_magnitudes( tt['abs_mag_r'] )
        lumB, idx = blue_luminosity_from_mag(B,dist)
        pgcCat=pgcCat[idx]
        ra = ra[idx]
        dec = dec[idx]
        dist = dist[idx]
        z = z[idx]
        lumB = lumB[idx]
        angle_error = angle_error[idx]
        dist_err = dist_err[idx]
        z_err = z_err[idx]
        # hack to weight the extra galaxies at 11.429 with zero blue luminosity
        lumB = np.ma.masked_where( ((dist>11.4291) & (dist<11.4293)) , lumB).filled(0)
        
        return (pgcCat,ra,dec,dist,z,lumB,angle_error,dist_err,z_err)

    def extract_galaxies_table(self):
        tt = self.catalog
        pgcCat = tt['PGC']
        ra = tt['RA']
        dec = tt['Dec']
        z_err = tt['z_err']
        dist = tt['Distance']
        dist_err = tt['Distance Error']
        angle_error = tt['angle_error']
        lumB = tt['lumB']
        z = tt['z']

        return (pgcCat,ra,dec,dist,z,lumB,angle_error,dist_err,z_err)

    def generate_catalog_counterpart(self):
        catalog.load_catalog_counterpart(self)
        catalog.remove_clusters(self)
        catalog.remove_nans(self)
        t = self.catalog
        g_galactic = ICRS(ra=t['RA']*u.degree, dec=t['Dec']*u.degree, radial_velocity=t['z']*constants.c.to('km/s')).transform_to(Galactic)
        z=vcorr(g_galactic, apex_helio_to_3k)/constants.c.to('km/s')  
        t.replace_column('z', np.ma.masked_where(z<0.0,z).filled(0))
        t.replace_column('Distance', cosmo.luminosity_distance(t['z']).value)
        t.sort('Distance')
        distmax=1000.0
        nt = t[np.argmax(t['Distance']>0.1):np.argmin(t['Distance']<distmax)]
        self.catalog = nt
        (pgcCat,ra,dec,dist,z,lumB,angle_error,dist_err,z_err) = catalog.extract_galaxies_counterpart(self)

        self.pgc_number = pgcCat
        self.ra = ra
        self.dec = dec
        self.z_err = z_err
        self.distance = dist
        self.distance_error = dist_err
        self.angle_error = angle_error
        self.lumB = lumB
        self.z = z
        self.z = catalog.group_velocity_corr(self)

        self.catalog= Table([self.pgc_number,self.ra,self.dec,self.distance,self.z,self.lumB,self.angle_error,self.distance_error,self.z_err],
                            names=('PGC', 'RA', 'Dec', 'Distance', 'z', 'lumB', 'angle_error','Distance Error','z_err'))

        return self.pgc_number,self.ra,self.dec,self.distance,self.z,self.lumB,self.angle_error,self.distance_error,self.z_err

    def generate_catalog(self):
        catalog.load_catalog(self)
        catalog.remove_clusters(self)
        catalog.remove_nans(self)
        t = self.catalog
        g_galactic = ICRS(ra=t['RA']*u.degree, dec=t['Dec']*u.degree, radial_velocity=t['z']*constants.c.to('km/s')).transform_to(Galactic)
        z=vcorr(g_galactic, apex_helio_to_3k)/constants.c.to('km/s')  
        t.replace_column('z', np.ma.masked_where(z<0.0,z).filled(0))
        t.replace_column('Distance', cosmo.luminosity_distance(t['z']).value)
        t.sort('Distance')
        distmax=1000.0
        nt = t[np.argmax(t['Distance']>0.1):np.argmin(t['Distance']<distmax)]
        self.catalog = nt
        (pgcCat,ra,dec,dist,z,lumB,angle_error,dist_err,z_err) = catalog.extract_galaxies(self)

        self.pgc_number = pgcCat
        self.ra = ra
        self.dec = dec
        self.z_err = z_err
        self.distance = dist
        self.distance_error = dist_err
        self.angle_error = angle_error
        self.lumB = lumB
        self.z = z
        self.z = catalog.group_velocity_corr(self)

        self.catalog= Table([self.pgc_number,self.ra,self.dec,self.distance,self.z,self.lumB,self.angle_error,self.distance_error,self.z_err],
                            names=('PGC', 'RA', 'Dec', 'Distance', 'z', 'lumB', 'angle_error','Distance Error','z_err'))

        return self.pgc_number,self.ra,self.dec,self.distance,self.z,self.lumB,self.angle_error,self.distance_error,self.z_err