import datetime
import os
import errno
import cPickle
import joblib
import os
import errno

def addSecs(tm, secs):
	fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
	fulldate = fulldate + datetime.timedelta(seconds=secs)
	return fulldate.time()


def openPickle(path):
	try:
		obj = joblib.load(path)	
		return obj
	except IOError as e:
		print os.strerror(e.errno)
		return -1


def savePickle(obj, path):
	if not os.path.exists(os.path.dirname(path)):
		try:
			os.makedirs(os.path.dirname(path))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	
	joblib.dump(obj, path, compress=True, protocol=cPickle.HIGHEST_PROTOCOL)


def dictEvent(name, run, postSample):
	d = {
		'name' :					name,
		'run' : 					run,
		'postSamplePath' : 			'./Data/Run_%s/Posterior_samples/%s' %(run, postSample),
		'pofdInstantPath' : 		'./Data/Run_%s/Events/Pofd_%s.p' %(run, name)
		}
	return d


def dictRun(run, psdFile, desc, T, nobs):
	d = {
		'run' : 				run,
		'psdPath': 				'./Data/PSD_data/%s' %(psdFile),
		'pofdPath': 			'./Data/Run_%s/Events/Pofd_%s.p' %(run, desc),
		'pofdMargPath' : 		'./Data/Run_%s/Events/Pofd_%s_marg.p' %(run, desc),
		'pofdAvPath' : 			'./Data/Run_%s/Events/Pofd_%s_average.p' %(run, desc),
		'T' : 					T,
		'nobs' : 				nobs
		}
	return d


