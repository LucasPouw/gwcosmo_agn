#!/bin/sh
exec python3 /home/ignacio.magana/src/gwcosmo/bin/gwcosmo_single_posterior.py \
--method statistical \
--posterior_samples /home/ignacio.magana/src/gwcosmo/gwcosmo/data/posterior_samples/first2years-data/2016/lalinference_mcmc/${Process}/posterior_samples.hdf5 \
--mass_distribution BNS \
--galaxy_catalog mdc \
--mdc_version '2.1' \
--min_H0 10 \
--max_H0 210 \
--bins_H0 2000 \
--min_dist 0.1 \
--max_dist 400 \
--bins_dist 2000 \
--use_3d_kde False \
--completion False \
--galaxy_weighting False \
--plot True \
--save True \
--outputfile event_${Process} \