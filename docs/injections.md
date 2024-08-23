# User guide for injections (as of 202310):

 

## What are those 'injections' computed by `gwcosmo_create_injections`?

The cosmological and population analysis of GW events requires computing a large number of integrals in a parameter space of dimension 12 (for the BBH case for instance), to properly take into account the selection effects. It is possible to estimate these integrals using the Monte Carlo integration method, that relies on using a large set of random values (called injections), drawn from a known probability density function. Then we can replace the integration by a simple summation over those random values; this is called "preference sampling". The MC integration step is done in the likelihood part of gwcosmo and `gwcosmo_create_injections` produces the set of random values. In pratice, one injection is a set of values corresponding to the parameters ($m_1,m_2,d_L,\text{ra},\text{dec},t_0$ and other intrinsic parameters) of a CBC (compact binary coalescence)  that would be detected by your instrumental setup, defined by a list of interferometers, their sensitivities, their duty cycles and the SNR threshold.

### Computation of injections

The injections are computed in the detector frame, for the masses $(m_1, m_2)$ and the luminosity distance $d_L$ for a fixed value of the SNR (the one your are using in your analysis). The injections are computed efficiently for SNRs $\in\{9,10,11,12\}$. For other values of the SNR the code will be slow. When the SNR $\in\{9,10,11,12\}$, the luminosity distance $d_L$ is drawn in a reduced parameter space where $d_L$ depends on $m_1$. This makes the code very efficient as it prevents computing injections we know by advance won't pass the SNR threshold. If one wants to draw $d_L$ independently of $m_1$, or for an SNR $\notin\{9,10,11,12\}$, we must use the argument `--dLmax_depends_on_m1 0` (which is `1` by default). Using the reduced parameter space for other values of the SNR is not possible in the current version of the code.

The usage of the code can be displayed with the command `gwcosmo_create_injections --help`.

### Example on a single computer (not relevant to get a full injection set):

The command below compute 100 injections i.e. 100 random CBC parameters having a SNR>2, using 8 cpus (i.e. the 100 detected injections are split between 8 processes) and drawing random values in a global parameter space where $d_L$ is independent of $m_1$:
```
gwcosmo_create_injections --Nsamps 100 --snr 2 --cpus 8 --output_dir my_injections --dLmax_depends_on_m1 0
```
when the code finishes, it creates a directory called  `injection_files` by default (you can set the name you wish with the flag `--output_dir`) and after execution, you will find 2 files in this directory:
- `tmp_detected_events_1_2.00.p`: a single file used by all subprocesses, for the temporary detected injections (written as a list in a pickle format)
- `detected_events_1_2.00.p` when the code terminates normally, converts the temporary filename into a dictionary format.

In these filenames, the `1` corresponds to the run number which is modified when a large campaign of injections is started and the `2.00` is the SNR threshold.

The actual `gwcosmo-injections` object used in the analysis is computed in a second step as the normal way is to add-up a large number of injections of the type `detected_events_1_2.00.p`.


### If the code crashes (computer down, process killed...)

In case of a crash, the final dict file `detected_events_1_2.00.p` is not created automatically but the data is not lost as it is still in the temporary file  `tmp_detected_events_1_2.00.p` which can in turn be converted manually into the correct dict with the command:
```
gwcosmo_create_injections --tmp_to_dict ./tmp_detected_events_1_2.00.p
```
this creates a file `dict_tmp_detected_events_1_2.00.p`
Note that this dictionary file is not exactly the same than the one created after a normal termination of the code: the order of injections differs so it will not affect the computation of the selection effects.

It is also possible to dump on stdout the contents of a temporary file (for instance in real time to check that everything runs normally), with the command:
```
gwcosmo_create_injections --tmp_to_stdout tmp_detected_events_1_2.00.p
```

### TYPICAL SITUATION: usage on a cluster with `dag/htcondor`, for O1-O2-O3-O4:
You may need to install the `htcondor` python package with pip: `pip install htcondor`.<br>
By default, injections are computed for O1-O2-O3-O4 with the following duty cycles and run durations:
```
Using observation days: {'O1': 129, 'O2': 268, 'O3': 330, 'O4': 330}
Using duty factors: {'O4': {'H1': 0.75, 'L1': 0.75, 'V1': 0.75}, 'O3': {'H1': 0.7459, 'L1': 0.7698, 'V1': 0.7597}, 'O2': {'H1': 0.653, 'L1': 0.618, 'V1': 0.0777}, 'O1': {'H1': 0.646, 'L1': 0.574, 'V1': -1}}
Using probability of run: {'O1': 0.12204351939451277, 'O2': 0.2535477767265847, 'O3': 0.3122043519394513, 'O4': 0.3122043519394513}
Using detectors: ['H1', 'L1', 'V1']
Using Tobs (years): 2.8939082819986313
Using O4 sensitivity: low
```
by default, the pessimistic O4 sensitivity is assumed. You can choose the optimistic one using `----O4sensitivity high`. A duty factor of -1 means the interferometer is not taking data as it is the case for Virgo during O1.
```
gwcosmo_create_injections_dag --cpus 16 --nruns 200 --Nsamps 1000 --snr 9 --output_dir injections_snr9 --days_of_O4 0 --dLmax_depends_on_m1 1
```
will ask for 200 jobs, each of them using 16 cpus during execution, 200 runs from 1 to 200 each of them will stop after 1000 detected injections (above SNR=9). It takes ~ 15 minutes for a single job to finish, using 16 cpus so that if the cluster runs all jobs at the same time, we'll have 2e5 detected injections in 15 minutes with SNR = 9. A file `run.sub` is created in the directory `injections_snr9` containing commands for `condor`.  You really should use a specific name for the directory (here `--output_dir injections_snr9`), the default name being `injections_files`.

To actually start the computation, go to your output directory `injections_snr9`  and run the command:
```
condor_dag_submit dagfile.dag
```
it will run the script `run.sub` with varying value of the `--run` flag of `gwcosmo_create_injections` from 1 to 200. Then each process (200 here) will create its own injection file (temporary format) in the common subdirectory called `injection_files` (its name if fixed when using the `dag/htcondor` method). After normal termination, each run (fed by 16 processes in our example) will create its own dictionary of injections. All final dictionaries should then be manually merged (combined) to create a single big dictionary + the actual injections object (h5 file) needed for the analysis:
```
gwcosmo_create_injections --combine 1 --path_combine injections_snr9/injection_files/ [--output_combine all_injections.h5]
```
the name of the final injections object h5df file can be set in the command-line with the flag `--output_combine all_injections.h5`
by default, the name is set internally to `combined_pdet_SNR_9.00.h5`, in the hdf5 file format. You can extract the data of the h5 file with the python commands:
```
injdata = h5py.File(opts.injections_path,'r')
injections = injections_at_detector(m1d=np.array(injdata['m1d']),
                                    m2d=np.array(injdata['m2d']),
                                    dl=np.array(injdata['dl']),
                                    prior_vals=np.array(injdata['pini']),
                                    snr_det=np.array(injdata['snr']),
                                    snr_cut=0,
                                    ifar=np.inf+0*np.array(injdata['m1d']),
                                    ifar_cut=0,
                                    ntotal=np.array(injdata['ntotal']),
                                    Tobs=np.array(injdata['Tobs']))
injdata.close()
```
This step is done in the `gwcosmo` analyses (dark or bright sirens).

We also create the pickle file containing the stacked dicts, its name is the same than the injections one with `stacked_` added in prefix. The stacked file is mainly created for additional checks if needed, it is not needed for the analysis.

**You should check the output of the `combine` step, to be sure that the combined files are correctly processed.** For instance, you'll see at the end of the output:
```
average Ndet/Nsim in % over 200 files: 0.2585322716335059 0.000567826545117124
Check: combined total Nsim[O3]: 106469
Got 200 dicts for rescaling of probabilities.
Check: combined total Nsim[O1]: 30754
Rescale probabilities for O1: 30754 events...
Check: combined total Nsim[O2]: 62777
Rescale probabilities for O2: 62777 events...
Check: combined total Nsim[O3]: 106469
Rescale probabilities for O3: 106469 events...
All initial probs have been rescaled: 200000 vs 200000.
```

the average `Ndet/Nsim=0.258532.. %` should be the same among the 200 files so that the `stddev=0.000567...` should be very small. If this is not the case of if the number check `200000 vs 200000` is not true, this indicates a problem.

### PARTICULAR CASE: ask for a huge number of injections or when not using the reduced parameter space: usage on several clusters:

if we want a (very) large injection set, and we know by advance that the computation will take time (several days), it's more efficient to use several clusters at the same time (CIT, livingston, whatever). Be sure to run injections in the very same configuration between them (same duty factors, interferometer sensitivities etc).

You will have several injections temporary files written on several computers at different places so the idea is to copy, at the same place, all temporary files: you need to use separate directories as temporary files can have the same name; so be sure to store them in different directories (i.e. ./cluster1, ./cluster2, ./cluster3...)

once all temp files are gathered, you have to create a file (let's say injections_file_list.txt) containing the list of these files with full paths
this file has lines such as:
```
/.../cluster1/my_injections/tmp_detected_events_18_2.00.p
/.../cluster4/my_injections/tmp_detected_events_13_2.00.p
/.../cluster14/my_injections/tmp_detected_events_18_2.00.p
```

then you run `gwcosmo_create_injections` with those arguments:
```
gwcosmo_create_injections --merge_tmpfile_list injections_file_list.txt
```
As an example, to retrieve efficiently these numerous temporary files you can use a command such as:
```
rsync -e ssh -rauvz --include "*/" --include "tmp*.p" --exclude "*" albert.einstein@cit:/home/albert.einstein/injections_snr9_cit /home/ae/injections_cit
rsync -e ssh -rauvz --include "*/" --include "tmp*.p" --exclude "*" albert.einstein@livingston:/home/albert.einstein/injections_snr9_liv /home/ae/injections_liv
```
then build the file containing the paths of all temporary files:
```
find /home/ae/injections\* -name \*tmp_detected\*.p > my_full_list.txt
```
and finally:
```
gwcosmo_create_injections --merge_tmpfile_list my_full_list.txt
```

during this step, each tmp file is converted into a dict and a random string is added to the filename as we can have the same name for different tmp files

the code will create a temporary directory (the path is indicated in the stdout) and all temp files will be converted in dictionaries in this temporary directory, for example its unique path is like
`/var/folders/py/mq1rbc3d41d97zs67brz4lv00000gn/T/tmpbpyskftk`

once all dicts are created, we merge them into a single one in a h5 format, that can be used by gwcosmo for the cosmological analysis:
```
gwcosmo_create_injections --combine 1 --path_combine /var/folders/py/mq1rbc3d41d97zs67brz4lv00000gn/T/tmpbpyskftk --output_combine all_injections.h5
```
2 files are written after execution:

- `all_injections.h5` which contains the final injection data for cosmology studies
   
- a file named `stacked_all_injections.p` that contains the stacked dictionnaries (you probably won't have to use it).





