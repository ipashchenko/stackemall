# stackemall
Code for creating stacks of polarization VLBI maps and their uncertainty analysis. 

## Requirements
These are mostly from dependencies of ``ve`` submodule: ``astropy``, ``scikit-image``, ``scikit-learn``, ``numpy``. ``scipy``, ``pycircstat``, ``matplotlib`` (this list can be incomplete!).
``GNU parallel`` is used to parallelize the workflow for hundreds of sources. ``Difmap`` is used to CLEAN ``UVFITS`` visibility data sets.

## Installation
```
git clone --recurse-submodules git@github.com:ipashchenko/stackemall.git
```

## Files
* ``stack.py`` - handles stacking process: creating, saving and plotting stack images.
* ``create_artificial_data.py`` - replicates multiepoch UVFITS files using CLEAN-models of the original data and specified error models.
* ``stack_utils.py`` - keeps utility functions used by all other modules.
* ``run_mojave_source.py`` - implements the workflow. The only necessary thing to fill in this script is
``results_dir`` - directory to store the results (e.g. ``/mnt/storage/ilya/MOJAVE_pol_stacking``).
* ``core_effsets.txt`` - file with sources, epochs and core offsets used to align multiepoch images. It is also used to infer the number of epochs in stack for given source.
* ``VLBA_EB_residuals_D.json`` - file with residual D-terms estimates for each antenna. Used to model residual D-term uncertainty.  
* ``final_clean`` - version of D. Homan ``difmap`` final CLEAN script used in the analysis of MOJAVE polarization stacks.
* ``source_dec.txt`` - file with declinations of sources. Used to calculate common circular beam. Based on M. Lister fits of cubic polynomial to all sources.
* ``sources_to_process.txt`` - file with sources to process in parallel jobs using ``GNU parallel``.

## Usage
Script ``run_mojave_source.py`` used to run simulations has only one positional argument - source B1950 name:
```
$ python run_mojave_source.py 0539-057
```
However, in ``__main__`` part of the script one can change:
* ``n_mc`` - number of realizations.
* ``sigma_scale_amplitude``, ``noise_scale``, ``sigma_evpa_deg`` - parameters of the residual noise added. See code for explanation.
* ``n_epochs_not_masked_min`` & ``n_epochs_not_masked_min_std`` - minimal number of non-masked epochs for calculating means and stds.

## Using GNU parallel
Suppose we have a column of source names to process in a file ``sources_to_process.txt``. Then to run processing in 20
parallel jobs one can use:
```
$ parallel --files --results result_{1} [--retry-failed] --joblog /home/ilya/github/stackemall/log --jobs 20 -a sources_to_process_short.txt "python run_mojave_source.py"

``` 
Here for source ``mysource`` data streams``stdout`` and ``stderr`` will be redirected to ``result_mysource`` and ``result_mysource.err`` files. The result exit status (with timing)
will be logged in file ``log``. If some jobs are failed ``--retry--failed`` can be used to re-run them (after fixing the problem that caused failure). However, directories with intermediate results for the failed
sources should be removed using ``stack_utils.remove_dirs_with_failed_jobs`` function.