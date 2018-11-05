#!/usr/bin/python
"""
This script preprocesses Glade v2.3 catalog into production use.
Ignacio Magana
"""
import numpy as np
import astropy
import astropy.constants as constants
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
import pkg_resources

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.27)

z_err_fraction = 0.06
a_err_fraction = 0.08
useNGC4993only = 0.0

catalog_data_path = pkg_resources.resource_filename('gwcosmo', 'data/catalog_data/')
groupmembers4993 = np.genfromtxt(catalog_data_path + "NGC4993group.txt", usecols=0) #NGC 4993's group
pgc_KTgroups,groupgal = np.genfromtxt(catalog_data_path + "KTgroups.txt", usecols=(0,4), unpack=True)
catalog_file = catalog_data_path + "gladecatalogv2.3.dat"

apex_helio_to_3k = Galactic(l=264.14*u.degree, b=48.26*u.degree, radial_velocity=371.0*u.km/u.s)

def remove_clusters(nt):
    gc_flag = nt['Cluster']
    no_gc_sel = gc_flag == 'C'
    nt.remove_rows(no_gc_sel)
    return nt

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
    return lum

def replace_bad_magnitudes(aa):
    maa = np.ma.masked_values(aa,-999.)
    return maa.filled(1e20).data

def remove_nans(aa):
    nanidx = np.where(np.isnan(aa)==False)
    return aa[nanidx]

def lum_cut(aa,lum_cut_value):
    lumidx = np.where(aa>lum_cut_value*Lstar)
    return aa[lumidx], lumidx

def extract_galaxies(tt):
    galname = tt['Galaxy Name']
    cluster_name = tt['Cluster']
    pgcCat = tt['PGC']
    ra = tt['RA']
    dec = tt['Dec']    
    z = tt['z']
    dist = tt['Distance']

    angle_error = a_err_fraction / ( dist )
    
    dist_err = tt['Distance Error']
    fractional_err = np.ma.masked_values(dist_err,-999.) / dist
    dist_err = dist * fractional_err.filled(z_err_fraction).data

    z_err = z * fractional_err.filled(z_err_fraction).data
    
    B = replace_bad_magnitudes( tt['abs_mag_r'] )
    lumB = blue_luminosity_from_mag(B,dist)

    # hack to weight the extra galaxies at 11.429 with zero blue luminosity
    lumB = np.ma.masked_where( ((dist>11.4291) & (dist<11.4293)) , lumB).filled(0)   
    
    # hack to remove deal with Galactic Center and SN1987a
    glist = ['GALCENTER','ESO056-115']
    for g in glist:
        delete_i = np.argmax(tt['Galaxy Name']==g)
        if delete_i:
            lumB[delete_i]=0
            
    # hack to remove nans
    nanidx = np.where(np.isnan(lumB)==False)

    galname = galname[nanidx]
    cluster_name = cluster_name[nanidx]
    pgcCat=pgcCat[nanidx]
    ra = ra[nanidx]
    dec = dec[nanidx]
    dist = dist[nanidx]
    z = z[nanidx]
    lumB = lumB[nanidx]
    angle_error = angle_error[nanidx]
    dist_err = dist_err[nanidx]
    z_err = z_err[nanidx]
    B = B[nanidx]
    
    return (pgcCat,galname,cluster_name,ra,dec,z,dist,dist_err,B,lumB)

def group_velocity_corr(z,pgcCat):
    # group correction (from Maya)
    allgroups = np.unique(groupgal)
    groupmembersKT = [[] for i in range(len(allgroups))]
    zsel_group = z
    for i in range(len(allgroups)):
        for j,g in enumerate(groupgal):
            if g==allgroups[i]:
                groupmembersKT[i].append(pgc_KTgroups[j])

    zs_group = [[] for i in range(len(allgroups))]
    for i in range(len(allgroups)):
        for j, pgc in enumerate(pgcCat):
            if pgc in groupmembersKT[i]:
                zs_group[i].append(z[j])
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

def main():
    t = Table.read(catalog_file,format='ascii')
    t = remove_clusters(t)
    g_galactic = ICRS(ra=t['RA']*u.degree, dec=t['Dec']*u.degree, \
        radial_velocity=t['z']*constants.c.to('km/s')).transform_to(Galactic)
    z=vcorr(g_galactic, apex_helio_to_3k)/constants.c.to('km/s')  
    t.replace_column('z', np.ma.masked_where(z<0.0,z).filled(0))
    t.replace_column('Distance', cosmo.luminosity_distance(t['z']).value)
    t.sort('Distance')

    (pgcCat,galname,cluster_name,ra,dec,z,dist,dist_err,bmag,lumB) = extract_galaxies(t)
    z = group_velocity_corr(z,pgcCat)

    data = Table([pgcCat,galname,cluster_name,ra,dec,z,dist,dist_err,bmag,lumB], \
        names=['PGC','Galaxy Name','Cluster','RA', 'Dec', 'z', 'Distance','Distance Error','Bmag','lumB'])
    astropy.io.ascii.write(data, 'gladecatalogv2.3_corrected.dat')

if __name__ == "__main__":
    main()