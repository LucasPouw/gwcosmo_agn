import scipy.constants as sc
import Common_func as cf
#Basic constants
snrThrComb = 12.0
mSolar = 1.989e30
Mpc = 3.0857e22
c = sc.c
G = sc.G

# Evenly spaced distance vector
dmax = 7500.0
dsize = 1000

# Pixels properties
nside = 1
nsideDet = 16
dpi = 250

# Number of samples for marginalising over inc, psi, mass
nSamp = 50000
alpha = 2.35
pools = 22
mmin = 5
mmax = 100

#Sampler set up
nWalk = 1000
nBurn = 0
nSteps = 20000
threads = 44
amax = 50
rmin = 0.0001
rmax = 250

lenSteps = nBurn + nSteps

# Parameters to set up the num calculation
nNum = 20000					# number of mass samples to calculate num
fmin = 10						# min frequency


runsList = [cf.dictRun('O1', 'aligo_early.txt', 'early', 51.5, 3),
			cf.dictRun('O2', 'aligo_mid.txt', 'mid', 117, 7)
			]

eventsList = [cf.dictEvent('GW150914', 'O1', 'allSsp_postGW150914.dat'),
			# cf.dictEvent('GW151226', 'O1', 'allSsp_postGW151226.dat'),
			# cf.dictEvent('LVT151012', 'O1', 'allSsp_postLVT151012.dat'),
			# cf.dictEvent('GW170104', 'O2', 'allIMRPPsp_post_GW170104.dat'),
			# cf.dictEvent('GW170608', 'O2', 'allIMRPpv2_posterior_samples_GW170608.dat'),
			# cf.dictEvent('GW170818', 'O2', 'allIsp_post_170818.dat'),
			# cf.dictEvent('GW170819', 'O2', 'posterior_samples_all_GW170809_C02_Cleaned_HLV_IMRPhenomDpseudoFourPN_alignedspinzprior.dat'),

			# cf.dictEvent('GW170729', 'O2', 'posterior_samples_all_GW170729_C02_Cleaned_HLV_IMRPhenomDpseudoFourPN_alignedspinzprior.dat'),
			# cf.dictEvent('GW170823', 'O2', 'Jacob.Lange_GW170823-G298936-IMRPv2-lalnest-samples-C02-cleaned-H1L1-uniform-spin-mag-prior-fmin10.dat'),
			cf.dictEvent('GW170814', 'O2', 'Jacob.Lange_GW170814-G297595-IMRPv2-lalnest-samples-C02-cleaned-H1L1V1-uniform-spin-mag-prior-fmin20.dat')
			]


#File paths
samplesPath = './Results/Sampling/Data/Samples_%s.p'
probPath = './Results/Sampling/Data/Prob_%s.p'
acceptancePath = './Results/Sampling/Data/AcceptFrac_%s.p'
covarPath = './Results/Sampling/Data/Covariance_%s.p'
meanMapPath = './Results/Sampling/Plots/Mean_map_%s_%s'
stdMapPath = './Results/Sampling/Plots/STD_map_%s_%s'
maxProbPath = './Results/Sampling/Plots/Max_prop_%s_%s'
probPlotPath = './Results/Sampling/Plots/Probability_graph.png'

samplesLastPath = './Results/Sampling/Data/Samples_last.p'
probMeanPath = './Results/Sampling/Data/Probability_mean.p'
lastStepPath = './Results/Sampling/Data/Last_step.p'

mapPath = './Results/Pofd/Pofd_mean_%s'
survivalPath = './Results/Pofd/Survival_%s'
objSamplesPath = './Results/Pofd/Samples_distribution'

distPath = './Results/Pofd/Pofd_mollview_%s_%sMpc'

















