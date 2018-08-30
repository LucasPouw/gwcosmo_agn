import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg') #No display environment
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool
import datetime
import lal
import time
import scipy.interpolate
import scipy.integrate
import scipy.special
import scipy.stats

import Setup as p
import Common_func as cf


class Pofd:
	"""Generates the p(detection|omega, dist), assumes the power law distribution of masses"""
	def __init__(self):
		self.__d = np.linspace(1.0, p.dmax, p.dsize)
		self.__nPix = hp.nside2npix(p.nsideDet)
		self.__pixArr = range(self.__nPix)
		theta, phi = hp.pix2ang(p.nsideDet,self.__pixArr)
		self.__ra, self.__dec = phi, np.pi/2 - theta
		q = np.random.rand(p.nSamp)
		self.__inc = np.arcsin(2.0*q - 1.0) 				# Uniform in cos(inclination)
		self.__psi = np.random.rand(p.nSamp)*2.0*np.pi 		# Uniform in polarisation
		self.__m1, self.__m2 = self.__massDist() 			# Power law distribution for m1, m2
		self.mtotMin = 2*p.mmin*p.mSolar

	def __massDist(self):
		""" Minimum mass assumed to be 5 M_solar, total mass always less than 100 M_solar"""
		normConst = 1.0/(1-p.alpha)*(np.power(p.mmax-p.mmin, 1-p.alpha)-np.power(p.mmin, 1-p.alpha))
		rand = np.random.random_sample(p.nSamp)
		primary = self.__primaryM(rand, normConst)
		secondary = np.array([self.__secondaryM(primary[i]) for i in range(primary.size)])
		return primary*p.mSolar, secondary*p.mSolar


	def __primaryM(self, y, normConst):
		""" Returns the power law distributed m1 """
		h = y*normConst*(1-p.alpha)+np.power(p.mmin, 1 - p.alpha)
		return np.power(h, (1.0/(1 - p.alpha)))


	def __secondaryM(self, m1):
		""" Returns the uniformly distributed m2, always smaller than m1"""
		lim = min(m1, (p.mmax-m1))
		return scipy.stats.uniform.rvs(loc=p.mmin, scale=(lim - p.mmin))


	def setPSD(self, path, plot=False):
		"""
		Set up the PSD for LIGO Hanford and Livingston.
		Note: will want to include the Virgo PSD at a later date
		Optional: Plot the PSD if 'plot=True'
		"""
		PSDdata = np.genfromtxt(path)
		PSD = scipy.interpolate.interp1d(PSDdata[:,0],PSDdata[:,1])
		if plot == True:
			plt.figure()
			plt.loglog(PSDdata[:,0],PSD(PSDdata[:,0]))
			plt.show()
		return PSD


	def __antennaResponse(self, gmst, RA, Dec, psi, det):
		"""
		Compute detector response to signal with specific ra, dec, psi, and time
		"""
		detMap = {'H1': lal.LALDetectorIndexLHODIFF, \
				  'H2': lal.LALDetectorIndexLHODIFF, \
				  'L1': lal.LALDetectorIndexLLODIFF, \
				  'G1': lal.LALDetectorIndexGEO600DIFF, \
				  'V1': lal.LALDetectorIndexVIRGODIFF, \
				  'T1': lal.LALDetectorIndexTAMA300DIFF, \
				  'AL1': lal.LALDetectorIndexLLODIFF, \
				  'AH1': lal.LALDetectorIndexLHODIFF, \
				  'AV1': lal.LALDetectorIndexVIRGODIFF, \
				  'E1': lal.LALDetectorIndexE1DIFF, \
				  'E2': lal.LALDetectorIndexE2DIFF, \
				  'E3': lal.LALDetectorIndexE3DIFF}
		try:
			detector = detMap[det]
		except KeyError:
			raise ValueError, 'ERROR. Key %s is not a valid detector name.' % (det)
		# Get detector
		detval = lal.CachedDetectors[detector]
		response = detval.response
		fp, fc = lal.ComputeDetAMResponse(response, RA, Dec, psi, gmst)
		return fp, fc


	def __fmax(self, m):
		""" Returns maximum frequency given the total mass"""
		return 1/(np.power(6.0,3.0/2.0)*np.pi*m)*p.c**3/p.G


	def __numFmax(self, pathPSD):
		"""
		Generate the interpolated function num_fmax as a function of f_max
		"""
		PSD = self.setPSD(pathPSD)
		def I(f):
			return np.power(f,-7.0/3.0)/(PSD(f)**2) 
		fmax = self.__fmax(self.mtotMin)
		fmaxArr = np.linspace(p.fmin, fmax, p.nNum)
		numFmax = np.array([scipy.integrate.quad(I, p.fmin, fmaxArr[i], epsabs=0, epsrel=1.49e-4)[0]\
					for i in range(p.nNum)])
		return scipy.interpolate.interp1d(fmaxArr, numFmax)


	def __snrSquared(self, RA, Dec, inc, psi, m1, m2, detector, gmst, interpolNum):
		"""
		The optimal SNR**2 for one detector, marginalising over sky location, inclination, polarisation, mass
		"""
		fplus,fcross = self.__antennaResponse(gmst, RA, Dec, psi, detector)
		mtot = m1+m2
		mchirp = np.power(m1*m2,3.0/5.0)/np.power(mtot,1.0/5.0)
		A = float(np.sqrt(fplus**2*(1.0+np.cos(inc)**2)**2 + fcross**2*4.0*np.cos(inc)**2) \
			* np.sqrt(5.0*np.pi/96.0)*np.power(np.pi,-7.0/6.0) * np.power(mchirp,5.0/6.0))	# (dofz(z,H0)*Mpc)
		fmax = self.__fmax(mtot)
		num = interpolNum(fmax)
		return 4.0*A**2*num*np.power(p.G,5.0/3.0)/p.c**3.0


	def generatePofD_DLRADec(self, gmst=0):
		"""Calculate the survival function at every point on the skymap."""
		pool = ProcessingPool(p.pools)
		for item in p.runsList:
			interpolNum = self.__numFmax(item['psdPath'])
			print 'Computing p(det|dist, RA, Dec) for run %s.' %(item['run'])
			start = datetime.datetime.time(datetime.datetime.now())
			start2 = time.time()
			print 'Started at:', start
			
			def loopKiller(k):
				rho = np.zeros((p.nSamp, 1))
				for n in range(p.nSamp):
					rho[n] = self.__snrSquared(self.__ra[k], self.__dec[k], self.__inc[n], \
								self.__psi[n], self.__m1[n], self.__m2[n], 'H1', gmst, interpolNum) \
							+ self.__snrSquared(self.__ra[k], self.__dec[k], self.__inc[n], \
								self.__psi[n], self.__m1[n], self.__m2[n], 'L1', gmst, interpolNum)

				d = (self.__d.reshape((p.dsize, 1))).transpose()
				d = 1.0/(d*p.Mpc)**2
				survival = np.matmul(rho, d)
				survival = scipy.stats.ncx2.sf(p.snrThrComb**2, 4, survival)
				return np.sum(survival,0)/p.nSamp

			estTime = time.time()
			loopKiller(0)
			sec = ((time.time()-estTime))*self.__nPix/float(p.pools)
			print 'Estimated end:', cf.addSecs(start, sec)


			start = time.time()
			loopOut = pool.map(loopKiller, range(self.__nPix))
			pofd_dLRADec = np.vstack(loopOut)
			print 'Actual end:', datetime.datetime.time(datetime.datetime.now())
			print 'Elapsed sec:', time.time() - start2
			cf.savePickle(pofd_dLRADec, item['pofdPath'])
		pool.close()
		pool.join()
		pool.clear()


	def plotSamples(self):
		"""
		Plot to make sure all parameters are following the expected distribution,
		including the RA and Dec generated from healpy
		"""
		fig,ax = plt.subplots(2, 3)
		plt.subplots_adjust(wspace= 0.2)
		ax[0][0].hist(self.__ra, bins='auto')
		ax[0][0].set_xlabel('$\\alpha$')
		ax[0][0].set_ylabel('p($\\alpha$)')

		ax[0][1].hist(self.__dec, bins='auto')
		ax[0][1].set_xlabel('$\delta$')
		ax[0][1].set_ylabel('p($\delta$)')

		ax[0][2].hist(self.__inc, bins='auto')
		ax[0][2].set_xlabel('$\iota$')
		ax[0][2].set_ylabel('p($\iota$)')

		ax[1][0].hist(self.__psi, bins='auto')
		ax[1][0].set_xlabel('$\psi$')
		ax[1][0].set_ylabel('p($\psi$)')

		ax[1][1].hist(self.__m1/p.mSolar, bins='auto')
		ax[1][1].set_xlabel('$M_1$ (M_solar)')
		ax[1][1].set_ylabel('p($M_1$)')

		ax[1][2].hist(self.__m2/p.mSolar, bins='auto')
		ax[1][2].set_xlabel('$M_2$ (M_solar)')
		ax[1][2].set_ylabel('p($M_2$)')

		plt.savefig(p.objSamplesPath, dpi=p.dpi)
		plt.close('all')


	def plotSurvival(self):
		"""Plots the survival graph as a function of distance for each pixel, for each run"""
		N = 2  # nside to generate pixel centers and later convert to pixels on a detailed map
		if p.nsideDet > N:
			arr = np.arange(hp.nside2npix(N))
			theta, phi = hp.pix2ang(N, arr)
			pixPlot = hp.ang2pix(p.nsideDet, theta, phi)
		else:
			pixPlot = self.__pixArr

		for item in p.runsList:
			plt.figure()
			plt.grid(True)
			pofd_dLRADec = cf.openPickle(item['pofdPath'])
			survivalFunc = scipy.interpolate.interp1d(self.__d, pofd_dLRADec, bounds_error=False, fill_value=1e-10)
			for i in pixPlot:
				plt.plot(self.__d, survivalFunc(self.__d)[i,:])
			plt.xlabel('$d_L$ (Mpc)')
			plt.ylabel('$p(D|d_L,I)$')
			plt.title('Survival function for %s' %(item['run']))
			path = p.survivalPath %(item['run'])
			plt.savefig(path, dpi=p.dpi)
			plt.close('all')

		
	def __survivalMap(self, DL, RA, Dec, gmstrad, pofdPath):
		"""
		Returns the probability of detection when given luminosity distance (Mpc),right ascension,
		declination, and time (radians)
		"""
		pofd_dLRADec = cf.openPickle(pofdPath)
		survivalFunc = scipy.interpolate.interp1d(self.__d, pofd_dLRADec, bounds_error=False, fill_value=1e-10)
		hpxmap = survivalFunc(DL)[self.__pixArr]
		return hp.get_interp_val(hpxmap, np.pi/2.0 - Dec, RA - gmstrad)


	def plotDistribution(self, gmstrad=0.0):
		a1 = np.arange(100, 1000, 100)
		a2 = np.arange(1000, 3000, 500)
		dist = np.append(a1, a2)
		for item in p.runsList:
			for d in dist:
				desc = 'Mollview probability of detection, run %s' %(item['run'])
				unit = '$p(D|\Omega, d_L = %d  Mpc, I)$' % (d)
				path = p.distPath %(item['run'], str(d))
				plt.figure()
				hpxmap = np.zeros(self.__nPix, dtype=np.float)
				hpxmap[self.__pixArr] = self.__survivalMap(d, self.__ra, self.__dec, gmstrad, item['pofdPath'])[self.__pixArr]
				hp.mollview(hpxmap, title=desc, unit=unit)
				plt.savefig(path, dpi=p.dpi)
				plt.close('all')


	def run(self, data=True, plots=True):
		if data == True:
			self.generatePofD_DLRADec()
		if plots == True:
			self.plotDistribution()
			self.plotSurvival()
			self.plotSamples()


def main():
	probDet = Pofd()
	probDet.run()




if __name__=='__main__':
	main()










