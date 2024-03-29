import matplotlib
matplotlib.use("Agg")
import os
import sys
import glob
import shutil
import numpy as np
sys.path.insert(0, 've/vlbi_errors')
from spydiff import clean_difmap
from uv_data import UVData
from from_fits import (create_model_from_fits_file, create_clean_image_from_fits_file)
from bootstrap import create_random_D_dict, create_const_amp_D_dict, boot_ci


def rename_mc_stack_files(dir_all_files, mojave_format=True):
    all_stack_files = glob.glob(os.path.join(dir_all_files, "*.uvf"))
    all_stack_files = [os.path.split(fn)[-1] for fn in all_stack_files]
    realizations = set([fn.split('_')[1] for fn in all_stack_files])
    cwd = os.getcwd()
    os.chdir(dir_all_files)
    for realization in realizations:
        print("Making directory for realization {}".format(realization))
        if os.path.exists("artificial_{}".format(realization)):
            continue
        os.mkdir("artificial_{}".format(realization))
    os.chdir(cwd)

    for fn in all_stack_files:
        realization = fn.split('_')[1]
        if mojave_format:
            original_fn = fn.split('_')[2]+'_'+fn.split('_')[3]+'_'+fn.split('_')[4]
        else:
            original_fn = fn.split('_')[2]+'_'+fn.split('_')[3]

        print("Moving {}".format(fn))
        shutil.move(os.path.join(dir_all_files, fn),
                    os.path.join(dir_all_files, "artificial_{}".format(realization), original_fn))


def downscale_uvdata_by_freq(uvdata):
    if abs(uvdata.hdu.data[0][0]) > 1:
        downscale_by_freq = True
    else:
        downscale_by_freq = False
    return downscale_by_freq


class ArtificialDataCreator(object):
    def __init__(self, uvfits_file, path_to_clean_script, mapsize_clean, beam,
                 shift=None, models=None, working_dir=None, noise_from_V=True,
                 omit_residuals=False, do_smooth=True):
        """
        :param uvfits_file:
            UVFITS file with self-calibrated and D-terms corrected data.
        :param models: (optional)
            Dictionary with keys - Stokes parameters and values - instances of
            ``Model`` class with ``stokes`` attribute and ``ft(uv)`` method. If
            ``None`` than create models from CLEANed ``uvfits_file``.
            (default: ``None``)
        :param mapsize_clean: (optional)
            Tuple of number of pixels nad pixel size (mas).
        :param beam: (optional)
            Tuple of bmaj (mas), bmin (mas) and bpa (deg).
        :param shift: (optional)
            Shifts to apply to map. If ``None`` than do not apply shifts.
            (default: ``None``)
        :param working_dir: (optional)
            Directory for storing files. If ``None`` than use CWD. (default: ``None``)
        :param noise_from_V: (optional)
            Use noise estimated from Stokes V? If not than use successive
            differences approach. (default: ``True``)
        :param omit_residuals: (optional)
            Used if ``models`` is None for creating data generating CC-models.
        :param do_smooth: (optional)
            Used if ``models`` is None for creating data generating CC-models.
        """
        self.uvfits_file = uvfits_file
        self.path_to_clean_script = path_to_clean_script
        self.models = models
        self.shift = shift
        self.mapsize_clean = mapsize_clean
        self.beam = beam
        self.noise_from_V = noise_from_V
        self.omit_residuals = omit_residuals
        self.do_smooth = do_smooth

        self.stokes = ("I", "Q", "U")

        if working_dir is None:
            working_dir = os.getcwd()
        elif not os.path.exists(working_dir):
            os.makedirs(working_dir)
        self.working_dir = working_dir

        # Paths to FITS files with clean maps (the same parameters) for each Stokes.
        self.ccfits_files = dict()
        for stokes in self.stokes:
            self.ccfits_files[stokes] = os.path.join(self.working_dir, "cc_{}.fits".format(stokes))

        self._image_ctor_params = dict()
        if self.models is None:
            self._clean_original_data_with_the_same_params()

            # CLEAN models for original data
            self.ccmodels = dict()
            for stokes in self.stokes:
                ccmodel = create_model_from_fits_file(self.ccfits_files[stokes])
                self.ccmodels[stokes] = ccmodel
        else:
            self.ccmodels = None

    def _clean_original_data_with_the_same_params(self):
        print("Cleaning uv-data with the same parameters: mapsize = {}, beam = {}".format(self.mapsize_clean, self.beam))
        if self.shift is not None:
            shift = self.shift
        else:
            shift = None
        print("UVFITS {} shift = {}".format(self.uvfits_file, shift))
        for stokes in self.stokes:
            print("Stokes {}".format(stokes))
            clean_difmap(fname=self.uvfits_file, outfname="cc_{}.fits".format(stokes),
                         stokes=stokes, outpath=self.working_dir, beam_restore=self.beam,
                         mapsize_clean=self.mapsize_clean, shift=shift,
                         path_to_script=self.path_to_clean_script,
                         show_difmap_output=False, omit_residuals=self.omit_residuals,
                         do_smoot=self.do_smooth)

    def create_images(self, d_term=None, noise_scale=1.0, sigma_scale_amplitude=None,
                      sigma_evpa=None, constant_dterm_amplitude=False,
                      ignore_cross_hands=True):
        """
        :param d_term: (optional)
            Mappable with keys [antenna], where values are constant
            residual amplitudes (if ``constant_dterm_amplitude=True``) or std
            of the Re/Im of the residual D-terms. If ``None`` than do not apply.
        :param noise_scale: (optional)
            Scale factor for the original noise std to add to the model visibilities. (default: 1.0)
        :param sigma_scale_amplitude: (optional)
            STD of R and L gains residual amplitude uncertainty. If ``None``
            than do not scale. (default: ``None``)
        :param sigma_evpa:
            Standard deviation of the EVPA uncertainty [deg]. If ``None`` than do not apply.
            (default: ``None``)
        :param constant_dterm_amplitude: (optional)
            Model of the D-term residuals: constant amplitude with random phase of random normal Re/Im
            with specified std. (default: constant amplitude)
        :param ignore_cross_hands:
            Boolean. Use Q & U in generating model visibilities? Ignoring them is useful for
            analysing D-terms, while using them is essential for bootstrap-like analysis.
            (default: True)
        """
        # Get uv-data and CLEAN model
        # ``ignore`` option for 1226 1996_03_22
        uvdata = UVData(self.uvfits_file, verify_option="ignore")
        if self.noise_from_V:
            # Dictionary with keys - baselines and values - array of noise std
            # for IFs.
            noise = uvdata.noise()
        else:
            noise = uvdata.noise(use_V=False)

        for baseline in noise:
            noise[baseline] *= noise_scale
        uvdata.zero_data()

        # External model of CLEAN models from supplied UVFITS files?
        if self.models is not None:
            models = self.models
        else:
            models = self.ccmodels

        if ignore_cross_hands:
            models = [models["I"]]
        else:
            models = [models[stokes] for stokes in ("I", "Q", "U",)]
        uvdata.substitute(models)

        print("   * adding noise...")
        uvdata.noise_add(noise)

        if sigma_scale_amplitude is not None:
            print("   * adding scaling of amplitudes of all hands...")
            # The same scale
            scale = 1.0+np.random.normal(0, sigma_scale_amplitude, size=1)[0]
            uvdata.scale_hands(scale_r=scale, scale_l=scale)
            # Independent scales
            # scale_r = 1.0 + np.random.normal(0, sigma_scale_amplitude, size=1)[0]
            # scale_l = 1.0 + np.random.normal(0, sigma_scale_amplitude, size=1)[0]
            # uvdata.scale_hands(scale_r=scale_r, scale_l=scale_l)

        if d_term is not None:
            # Getting dictionary with keys - [antenna name][integer of IF]["R"/"L"]
            # and values - complex D-terms.
            if constant_dterm_amplitude:
                print("Constant amplitude D-term residuals! ==================")
                d_dict = create_const_amp_D_dict(uvdata, amp_D=d_term)
            else:
                d_dict = create_random_D_dict(uvdata, sigma_D=d_term)

            print("   * adding D-terms...")
            uvdata.add_D(d_dict)

        uvf_savename = os.path.split(self.uvfits_file)[-1]
        uvf_savename = "artificial_" + uvf_savename
        uvf_dterms = os.path.join(self.working_dir, uvf_savename)

        downscale_by_freq = downscale_uvdata_by_freq(uvdata)

        if sigma_evpa is not None:
            fi = np.random.normal(0, np.deg2rad(sigma_evpa))
            print("   * rotating EVPA...")
            uvdata.rotate_evpa(fi)

        uvdata.save(uvf_dterms, rewrite=True, downscale_by_freq=downscale_by_freq)

    def mc_create_uvfits(self, n_mc, d_term, sigma_scale_amplitude=None,
                         noise_scale=1.0, sigma_evpa=None,
                         constant_dterm_amplitude=False, ignore_cross_hands=False):
        import shutil
        for i in range(n_mc):
            print("Creating artificial data {} of {} :".format(i+1, n_mc))
            # Only create artificial uvfits files
            self.create_images(d_term, noise_scale, sigma_scale_amplitude, sigma_evpa,
                               constant_dterm_amplitude, ignore_cross_hands)
            # Rename them
            uvfits_fn = os.path.split(self.uvfits_file)[-1]
            shutil.move(os.path.join(self.working_dir, "artificial_{}".format(uvfits_fn)),
                        os.path.join(self.working_dir, "artificial_{}_{}".format(str(i+1).zfill(3), uvfits_fn)))

    def remove_cc_fits(self):
        """
        Remove FITS files with CC for all Stokes.
        """
        for stokes in self.stokes:
            try:
                os.unlink(self.ccfits_files[stokes])
            # E.g. using ready CC-models for simulations
            except FileNotFoundError:
                pass

