
Generating injections
=====================

**User guide for injections (as of 20230418):**

The injections are computed in the detector frame. By defaut, when the SNR is set to 9, the luminosity distance dL is drawn in a reduced parameter space where dL depends on m1. This prevents computing injections we know by advance won't pass the SNR threshold. If one want to draw dL independently of m1, or for an SNR != 9, we must use the argument `--dLmax_depends_on_m1 0` (which is `1` by default). Using the reduced parameter space with any SNR is possible and easy but it's not available in the current version of the code.

*usage on a single computer (not realistic for a full injection set):*


```
./create_injections --Nsamps 100 --snr 2 --cpus 8 --output_dir my_injections --dLmax_depends_on_m1 0
```


=> creates dir 'my_injections' and after execution, there are 2 files :
tmp_detected_events_1_2.00.p for the temporary detected injections (written as list in a pickle format)
and detected_events_1_2.00.p when the code terminates normally, contains the injections parameters in dict format. The actual 'gwcosmo-injections' object used in the analysis is computed in a second step as the normal way is to add-up a large number of injections.


***FOR TEST: usage on a single computer with crash of the code (for instance after the user killed the code):***


in case of a crash, the dict file is not created automatically
we can use the temporary file tmp_detected_events_1_2.00.p and convert it into the correct dict:
first, it's possible to dump on stdout its contents, to check it:


```
create_injections --tmp_to_stdout tmp_detected_events_1_2.00.p
```


the option --tmp_to_stdout can be used anytime, for instance to check if new injections regularly arrive as the Ndet and Nsim are written in real time

if everything is OK (i.e. there is indeed a list of injections), we can convert the file into the needed dictionnary:

```
create_injections --tmp_to_dict ./tmp_detected_events_1_2.00.p
```


this creates a file dict_tmp_detected_events_1_2.00.p
careful: this dict file is not exactly the same than the one created after normal termination of the code, only the order of injections differs

***TYPICAL SITUATION: usage on a cluster, for O1-O2-O3:***
```
./create_injections_dag --cpus 16 --nruns 200 --Nsamps 1000 --snr 9 --output_dir injections_snr9 --days_of_O4 0 --dLmax_depends_on_m1 1
```
will ask for 200 jobs, each of them using 16 cpus during execution, 200 runs from 1 to 200 each of them will stop after 1000 detected injections (above SNR=9). It takes ~ 15 minutes for a single job to finish, using 16 cpus so that if the cluster runs all jobs at the same time, we'll have 2e5 detected injections in 15 minutes with SNR = 9.

To start the computation, go to directory 'my_injections'  and run the bash command 
```
condor_dag_submit dagfile.dag
```
it will run the script run.sub with varying value of the '--run' flag of create_injections from 1 to 200

then each process (200 here) will create its own injection file (temporary format)

once finished, each process creates its own dict from its dedicated injections. All final dicts should then be merged (combined) to create a single big dictionnary + the actual injections object needed for the analysis:
```
./create_injections --combine 1 --path_combine my_injections/injection_files/ [--output_combine all_injections.h5]
```
the name of the final injections object h5df file can be set in the command-line with the flag `--output_combine all_injections.h5`
by default, the name is set internally

injections are written in the hdf5 file format

we also create the pickle file containing the stacked dicts, its name is the same than the injections one with `stacked_` added in prefix. The stacked file is mainly created for additional checks if needed, it is not needed for the analysis.

You should check the output of the `combine` step, to be sure that the combined files are correctly processed. For instance, you'll see at the end of the output:
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

the average Ndet/Nsim should be the same among all files so that the stddev should be very small.

***PARTICULAR CASE: ask for a huge number of injections or when not using the reduced parameter space: usage on several clusters:***

if we want a (very) large injection set, and we know by advance that the computation will take time (several days), it's more efficient to use several clusters at the same time (CIT, livingston, whatever). Be sure to run injections in the very same configuration between them (same SNRth, same duty factors etc).

You will have several injections temporary files written on several computers at different places so the idea is to copy on a given computer all temporary files: you need to use separate directories as temporary files can have the same name; so be sure to separate them in different directories (i.e. ./cluster1, ./cluster2, ./cluster3...)

once all temp files are gathered, you have to create a file (let's say injections_file_list.txt) containing the list of these files with full path
this file has lines such as:
```
/.../cluster1/my_injections/tmp_detected_events_18_2.00.p
/.../cluster4/my_injections/tmp_detected_events_13_2.00.p
/.../cluster14/my_injections/tmp_detected_events_18_2.00.p
```

```
./create_injections --merge_tmpfile_list injections_file_list.txt
```
to retrieve efficiently the temporary files you can use a command such as:
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
./create_injections --merge_tmpfile_list my_full_list.txt
```


each tmp file is converted into a dict and a random string is added to the filename as we can have the same name for different tmp files

the code will create a temporary directory (the path is indicated in the stdout) and all temp files will be converted in dictionnaries in the temporary directory, for example its unique path is
`/var/folders/py/mq1rbc3d41d97zs67brz4lv00000gn/T/tmpbpyskftk`

once all dicts are created, we merge them into a single one at the icaro-gwcosmo format:
```
./create_injections --combine 1 --path_combine /var/folders/py/mq1rbc3d41d97zs67brz4lv00000gn/T/tmpbpyskftk --output_combine all_injections.h5
```
2 files are written in this call: all_injections.h5 which contains the inj dict in icarogw/gwcosmo format
and a file named stacked_all_injections.p that contains the stacked dictionnaries


