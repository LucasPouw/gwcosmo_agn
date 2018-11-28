#!/bin/sh
conda activate gwcosmo
gwcosmo_single_posterior \
--implementation master \
--method statistical \
--posterior_samples /home/ignacio.magana/first2years-data/2016/lalinference_mcmc/${Process}/posterior_samples.hdf5 \
--skymap /home/ignacio.magana/first2years-data/2016/lalinference_mcmc/${Process}/skypos/skymap.fits.gz \
--mass_distribution BNS \
--galaxy_catalog mdc \
--mdc_version '1.0' \
--min_H0 10 \
--max_H0 210 \
--bins_H0 2000 \
--completion False \
--galaxy_weighting False \
--outputfile ./mdc/event_${Process}