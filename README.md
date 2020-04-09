# stackemall
Code for creating stacks of polarization VLBI maps and their uncertainty analysis. 

## Requirements
These are mostly from dependencies of ``ve`` submodule: ``astropy``, ``scikit-image``, ``scikit-learn``, ``numpy``. ``scipy``, ``pycircstat``, ``matplotlib`` (this list can be incomplete!)

## Installation
```
git clone --recurse-submodules git@github.com:ipashchenko/stackemall.git
```

## Files
Supposed to be run on ``CALCULON`` server. 

* ``stack.py`` handles stacking process: creating, saving and plotting stack images.

* ``create_artificial_data.py`` replicates multiepoch UVFITS files using CLEAN-models of the original data and specified errors.

* ``stack_utils.py`` keeps utility functions used by all other modules.

* ``run_mojave_source.py`` implements the workflow. The only necessary thing to fill in this script is
``jet_dir`` - directory on ``CALCULON`` mirrot of ``jet`` machine. It will be used to store the results (e.g. ``/mnt/jet1/ilya/MOJAVE_pol_stacking``).
* ``core_effsets.txt`` - file with sources, epochs and core offsets used to align multiepoch images. It is also used to infer the number of epochs in stack for given source.
* ``VLBA_EB_residuals_D.json`` - file with residual D-terms estimates for each antenna. Used to model residual D-term uncertainty.  


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
```bash
$ parallel --files --results result_{1} --retry-failed --bar --joblog /home/ilya/github/stackemall/log --jobs 20 -a sources_to_process_short.txt "python run_mojave_source.py"

``` 
Here for each source ``stdout`` and ``stderr`` will be redirected to ``result_source.seq`` and ``result_source.err`` files. The result exit status (with timing)
will be logged in file ``log``. Is some jobs are failed ``--retry--failed`` can be used to re-run them (with fixed python code).