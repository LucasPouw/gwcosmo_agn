#!/bin/sh
exec python gwcosmo_single_posterior
--posterior_samples /home/ignacio.magana/src/gwcosmo/gwcosmo/data/posterior_samples/first2years-data/2016/lalinference_mcmc/${Process}/posterior_samples.hdf5
--plot True
--save True
--outputfile event_${Process}
--PRBlikelihood False