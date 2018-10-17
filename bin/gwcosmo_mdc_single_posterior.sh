#!/bin/sh
exec python3 /home/ignacio.magana/src/gwcosmo/bin/gwcosmo_single_posterior.py \
--method statistical \
--posterior_samples /home/ignacio.magana/src/gwcosmo/gwcosmo/data/posterior_samples/first2years-data/2016/lalinference_mcmc/${Process}/posterior_samples.hdf5 \
--galaxy_catalog_default mdc \
--mdc_version '2.1' \
--plot True \
--save True \
--outputfile event_${Process} \