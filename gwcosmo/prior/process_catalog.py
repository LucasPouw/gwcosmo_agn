import numpy as np
import gwcosmo
import pylab as plt


groupmembers4993 = np.genfromtxt(catalog_data_path + "NGC4993group.txt", usecols=0) #NGC 4993's group
pgc_KTgroups,groupgal = np.genfromtxt(catalog_data_path + "KTgroups.txt", usecols=(0,4), unpack=True)