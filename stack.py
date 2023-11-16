import matplotlib
matplotlib.use("Agg")
import datetime
import os
import shutil
import sys
import numpy as np
from collections import Mapping
from stack_utils import (pol_mask, stat_of_masked, find_image_std, find_bbox,
                         correct_ppol_bias, image_of_nepochs_not_masked,
                         get_mask_for_ccstack, filter_CC)
sys.path.insert(0, 've/vlbi_errors')
from spydiff import clean_difmap
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file,
                       create_model_from_fits_file)
from image import Image
from image import plot as iplot
import matplotlib.pyplot as plt
from astropy.io import fits as pf
from astropy.stats import mad_std
import astropy.units as u


# # FIXME: Put command file to local dir
# def convert_difmap_model_file_to_CCFITS(difmap_model_file, stokes, mapsize, restore_beam, uvfits_template, out_ccfits,
#                                         shift=None, show_difmap_output=True):
#     """
#     Using difmap-formated model file (e.g. flux, r, theta) obtain convolution of your model with the specified beam.
#
#     :param difmap_model_file:
#         Difmap-formated model file. Use ``JetImage.save_image_to_difmap_format`` to obtain it.
#     :param stokes:
#         Stokes parameter.
#     :param mapsize:
#         Iterable of image size and pixel size (mas).
#     :param restore_beam:
#         Beam to restore: bmaj(mas), bmin(mas), bpa(deg).
#     :param uvfits_template:
#         Template uvfits observation to use. Difmap can't read model without having observation at hand.
#     :param out_ccfits:
#         File name to save resulting convolved map.
#     :param shift: (optional)
#         Shift to apply. Need this because wmodel doesn;t apply shift. If
#         ``None`` then do not apply shift. (default: ``None``)
#     :param show_difmap_output: (optional)
#         Boolean. Show Difmap output? (default: ``True``)
#     """
#     stamp = datetime.datetime.now()
#     command_file = "difmap_commands_{}".format(stamp.isoformat())
#     difmapout = open(command_file, "w")
#     difmapout.write("observe " + uvfits_template + "\n")
#     difmapout.write("select " + stokes + "\n")
#     difmapout.write("rmodel " + difmap_model_file + "\n")
#     difmapout.write("mapsize " + str(mapsize[0] * 2) + "," + str(mapsize[1]) + "\n")
#     if shift is not None:
#         # Here we need shift, because in CLEANing shifts are not applied to
#         # saving model files!
#         difmapout.write("shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n")
#     print("Restoring difmap model with BEAM : bmin = " + str(restore_beam[1]) + ", bmaj = " + str(restore_beam[0]) + ", " + str(restore_beam[2]) + " deg")
#     # default dimfap: false,true (parameters: omit_residuals, do_smooth)
#     difmapout.write("restore " + str(restore_beam[1]) + "," + str(restore_beam[0]) + "," + str(restore_beam[2]) +
#                     "," + "true,false" + "\n")
#     difmapout.write("wmap " + out_ccfits + "\n")
#     difmapout.write("exit\n")
#     difmapout.close()
#
#     shell_command = "difmap < " + command_file + " 2>&1"
#     if not show_difmap_output:
#         shell_command += " >/dev/null"
#     os.system(shell_command)
#     os.unlink(command_file)



def convert_difmap_model_file_to_CCFITS(difmap_model_file, stokes, mapsize, restore_beam, uvfits_template, out_ccfits,
                                        shift=None, show_difmap_output=True):
    """
    Using difmap-formated model file (e.g. flux, r, theta) obtain convolution of your model with the specified beam.

    :param difmap_model_file:
        Difmap-formated model file. Use ``JetImage.save_image_to_difmap_format`` to obtain it.
    :param stokes:
        Stokes parameter.
    :param mapsize:
        Iterable of image size and pixel size (mas).
    :param restore_beam:
        Beam to restore: bmaj(mas), bmin(mas), bpa(deg).
    :param uvfits_template:
        Template uvfits observation to use. Difmap can't read model without having observation at hand.
    :param out_ccfits:
        File name to save resulting convolved map.
    :param shift: (optional)
        Shift to apply. Need this because wmodel doesn't apply shift. If
        ``None`` then do not apply shift. (default: ``None``)
    :param show_difmap_output: (optional)
        Boolean. Show Difmap output? (default: ``True``)
    """
    from subprocess import Popen, PIPE

    cmd = "observe " + uvfits_template + "\n"
    cmd += "select " + stokes + "\n"
    cmd += "rmodel " + difmap_model_file + "\n"
    cmd += "mapsize " + str(mapsize[0] * 2) + "," + str(mapsize[1]) + "\n"
    if shift is not None:
        # Here we need shift, because in CLEANing shifts are not applied to
        # saving model files!
        cmd += "shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n"
    print("Restoring difmap model with BEAM : bmin = " + str(restore_beam[1]) + ", bmaj = " + str(restore_beam[0]) + ", " + str(restore_beam[2]) + " deg")
    # default dimfap: false,true (parameters: omit_residuals, do_smooth)
    cmd += "restore " + str(restore_beam[1]) + "," + str(restore_beam[0]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
    cmd += "wmap " + out_ccfits + "\n"
    cmd += "exit\n"

    with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
        outs, errs = difmap.communicate(input=cmd)
    if show_difmap_output:
        print(outs)
        print(errs)


def CCFITS_to_difmap(ccfits, difmap_mdl_file, shift=None):
    hdus = pf.open(ccfits)
    hdus.verify("silentfix")
    data = hdus[1].data
    deg2mas = u.deg.to(u.mas)
    with open(difmap_mdl_file, "w") as fo:
        for flux, ra, dec in zip(data['FLUX'], data['DELTAX'], data['DELTAY']):
            ra *= deg2mas
            dec *= deg2mas
            if shift is not None:
                ra -= shift[0]
                dec -= shift[1]
            theta = np.rad2deg(np.arctan2(ra, dec))
            r = np.hypot(ra, dec)
            fo.write("{} {} {}\n".format(flux, r, theta))


class Stack(object):
    def __init__(self, uvfits_files, mapsize_clean, beam, path_to_clean_script,
                 shifts=None, box_files=None, shifts_errors=None, working_dir=None,
                 create_stacks=True, n_epochs_not_masked_min=1,
                 n_epochs_not_masked_min_std=5, use_V=False,
                 omit_residuals=False, do_smooth=True,
                 mask_cc_models_using_their_stack=False):
        """
        :param uvfits_files:
            Iterable of UVFITS files with self-calibrated and D-terms corrected data.
        :param mapsize_clean:
            Tuple of number of pixels and pixel size (mas).
        :param beam:
            Tuple of bmaj (mas), bmin (mas) and bpa (deg).
        :param path_to_clean_script:
            Path to Dan Homan CLEANing script.
        :param shifts: (optional)
            Iterable of shifts to apply to maps. If ``None`` than do not apply shifts.
            (default: ``None``)
        :param shifts_errors: (optional)
            Iterable of the parameters of 2D Gaussian distribution (maj[mas], min[mas], bpa[deg])
            to use to model error in the derived core shifts. If ``None`` then do not model this
            error. (default: ``None``)
        :param working_dir: (optional)
            Directory for storing files. If ``None`` than use CWD. (default: ``None``)
        :param create_stacks: (optional)
            Create stack images during initialization?
            If ``False`` than they should be created explicitly using
            corresponding methods (default: ``True``).
        :param n_epochs_not_masked_min: (optional)
            Minimal number of non-masked epochs in pixel of stacked image to
            consider when obtaining mean. (default: ``1``)
        :param n_epochs_not_masked_min_std: (optional)
            Minimal number of non-masked epochs in pixel of stacked image to
            consider when obtaining std. (default: ``5``)
        :param use_V: (optional)
            Boolean. Consider Stokes V stacking? (default: ``None`)
        :param omit_residuals: (optional)
            Difmap CLEAN parameter. (default: ``False``)
        :param do_smooth: (optional)
            Difmap CLEAN parameter. (default: ``True``)
        :param mask_cc_models_using_their_stack: (optional)
            Boolean. If ``True`` than mask CC-components based on their stack.
            (default: ``False``)
        """

        # Check input data consistency
        if shifts is not None:
            assert len(shifts) == len(uvfits_files)
        if shifts_errors is not None:
            assert len(shifts_errors) == len(uvfits_files)
        if path_to_clean_script is None:
            raise Exception("Specify path to CLEAN script!")

        self.stokes = ("I", "Q", "U")
        if use_V:
            self.stokes = ("I", "Q", "U", "V")
        absent_uvfits_files = list()
        for uvfits_file in uvfits_files:
            if not os.path.exists(uvfits_file):
                absent_uvfits_files.append(uvfits_file)
                print("No UVFITS file is found: ", uvfits_file)
        for uvfits_file in absent_uvfits_files:
            uvfits_files.remove(uvfits_file)
        self.uvfits_files = uvfits_files
        self.n_data = len(self.uvfits_files)
        self.shifts = shifts
        self.box_files = box_files
        self.shifts_errors = shifts_errors
        self.mapsize_clean = mapsize_clean
        self.beam = beam
        self._npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsize_clean[1]**2)

        self.path_to_clean_script = path_to_clean_script
        self.n_epochs_not_masked_min = n_epochs_not_masked_min
        self.n_epochs_not_masked_min_std = n_epochs_not_masked_min_std

        self.omit_residuals = omit_residuals
        self.do_smooth = do_smooth

        self.mask_cc_models_using_their_stack = mask_cc_models_using_their_stack

        if working_dir is None:
            working_dir = os.getcwd()
        elif not os.path.exists(working_dir):
            os.makedirs(working_dir)
        self.working_dir = working_dir

        # Paths to FITS files with clean maps (the same parameters) for each Stokes.
        self.ccfits_files = dict()
        for stokes in self.stokes:
            self.ccfits_files[stokes] = list()
        for stokes in self.stokes:
            for i in range(len(uvfits_files)):
                self.ccfits_files[stokes].append(os.path.join(self.working_dir,
                                                              "cc_{}_{}.fits".format(stokes, str(i+1).zfill(3))))

        # Paths to difmap-format files with clean components for each Stokes.
        self.difmap_files = dict()
        for stokes in self.stokes:
            self.difmap_files[stokes] = list()
        for stokes in self.stokes:
            for i in range(len(uvfits_files)):
                self.difmap_files[stokes].append(os.path.join(self.working_dir,
                                                              "cc_{}_{}.mdl".format(stokes, str(i+1).zfill(3))))

        # List of dictionaries with keys - Stokes parameters and values -
        # instances of ``Model`` class with ``stokes`` attribute and ``ft(uv)`` method
        self.cc_models = len(uvfits_files)*[None]

        # Paths to FITS files with residuals images (the same parameters) for each Stokes.
        self.residuals_fits_files = dict()
        for stokes in self.stokes:
            self.residuals_fits_files[stokes] = list()
        for stokes in self.stokes:
            for i in range(len(uvfits_files)):
                self.residuals_fits_files[stokes].append(os.path.join(self.working_dir,
                                                                      "residuals_{}_{}.fits".format(stokes, str(i+1).zfill(3))))

        # Paths to FITS files with beam-convolved CCs for each Stokes.
        self.cconly_fits_files = dict()
        for stokes in self.stokes:
            self.cconly_fits_files[stokes] = list()
        for stokes in self.stokes:
            for i in range(len(uvfits_files)):
                self.cconly_fits_files[stokes].append(os.path.join(self.working_dir,
                                                                   "cconly_{}_{}.fits".format(stokes, str(i+1).zfill(3))))

        # Containers for full stack CLEAN images
        self.stack_images = dict()
        for stokes in ("I", "Q", "U", "RPPOL", "PPOL", "PANG", "FPOL",
                       "PPOL2", "FPOL2", "PANG2", "NEPOCHS", "PANGSTD",
                       "FPOLSTD"):
            self.stack_images[stokes] = None
        if use_V:
            self.stack_images["V"] = None

        self.per_epoch_P_mask = list()
        self.per_epoch_I_mask = list()

        # Containers for full stack CC-only images
        self.cconly_stack_images = dict()
        for stokes in ("I", "Q", "U", "PPOL", "PANG", "FPOL"):
            self.cconly_stack_images[stokes] = None
        if use_V:
            self.cconly_stack_images["V"] = None

        # Containers for full stack residual images
        self.stack_residuals_images = dict()
        for stokes in ("I", "Q", "U", "RPPOL", "PPOL", "PANG", "FPOL",
                       "PPOL2", "FPOL2", "PANG2", "NEPOCHS", "PANGSTD",
                       "FPOLSTD"):
            self.stack_residuals_images[stokes] = None
        if use_V:
            self.stack_residuals_images["V"] = None

        self._image_ctor_params = dict()
        self._clean_original_data_with_the_same_params()

        # Optionally filter CC for creating artificial uv-data
        mask = None
        if mask_cc_models_using_their_stack:
            mask = self.create_mask_from_stacked_cc()
        self.fill_cc_models(mask=mask)

        if create_stacks:
            # Need this first if omit_residuals = True
            self.create_stack_residual_images()
            self.create_stack_images()
            self.create_stack_CConly_images()

        # Containers for CLEAN images
        self.images = dict()
        for stokes in self.stokes:
            self.images[stokes] = list()
        self.pol_mask = None
        self.i_mask = None

    def _clean_original_data_with_the_same_params(self):
        print("Cleaning uv-data with the same parameters: mapsize = {}, restore beam = {}".format(self.mapsize_clean, self.beam))
        for i, uvfits_file in enumerate(self.uvfits_files):

            if self.box_files is not None:
                box_file = self.box_files[i]
                print("Using CLEAN box file: ", box_file)
            else:
                box_file = None

            if self.shifts is not None:
                shift = self.shifts[i]
                if self.shifts_errors is not None:
                    bmaj, bmin, bpa = self.shifts_errors[i]
                    bpa = np.deg2rad(bpa)
                    print("Adding core shift uncertainty using error ellipse with bmaj, bmin, bpa = {}, {}, {}...".format(bmaj, bmin, bpa))
                    delta_x = np.random.normal(0, bmaj, size=1)[0]
                    delta_y = np.random.normal(0, bmin, size=1)[0]
                    # ``bpa`` goes from North clockwise => bpa = 0 means maximal
                    # shifts at DEC direction (``delta_y`` should be maximal)
                    bpa += np.pi/2
                    delta_x_rot = delta_x*np.cos(bpa) + delta_y*np.sin(bpa)
                    delta_y_rot = -delta_x*np.sin(bpa) + delta_y*np.cos(bpa)
                    shift = (shift[0] + delta_x_rot, shift[1] + delta_y_rot)
                print("Cleaning {} with applied shift = {}...".format(uvfits_file, shift))
            elif self.shifts is None and self.shifts_errors is not None:
                bmaj, bmin, bpa = self.shifts_errors[i]
                bpa = np.deg2rad(bpa)
                print("Adding core shift uncertainty using error ellipse with bmaj, bmin, bpa = {}, {}, {}...".format(bmaj, bmin, bpa))
                delta_x = np.random.normal(0, bmaj, size=1)[0]
                delta_y = np.random.normal(0, bmin, size=1)[0]
                # ``bpa`` goes from North clockwise => bpa = 0 means maximal
                # shifts at DEC direction (``delta_y`` should be maximal)
                bpa += np.pi / 2
                delta_x_rot = delta_x*np.cos(bpa) + delta_y*np.sin(bpa)
                delta_y_rot = -delta_x*np.sin(bpa) + delta_y*np.cos(bpa)
                shift = (delta_x_rot, delta_y_rot)
                print("Cleaning {} with applied shift = {}...".format(uvfits_file, shift))
            else:
                shift = None
                print("Cleaning {} with no shift applied...".format(uvfits_file))

            for stokes in self.stokes:
                print("Stokes {}".format(stokes))
                # Here shifts are not applied for saving difmap model file!!!
                clean_difmap(fname=uvfits_file, outfname="cc_{}_{}.fits".format(stokes, str(i+1).zfill(3)),
                             stokes=stokes, outpath=self.working_dir, beam_restore=self.beam,
                             mapsize_clean=self.mapsize_clean, shift=shift,
                             path_to_script=self.path_to_clean_script, window_file=box_file, box_rms_factor=4,
                             show_difmap_output=False, omit_residuals=self.omit_residuals,
                             do_smooth=self.do_smooth, dmap="residuals_{}_{}.fits".format(stokes, str(i+1).zfill(3)),
                             dfm_model="cc_{}_{}.mdl".format(stokes, str(i+1).zfill(3)))

        image = create_clean_image_from_fits_file(os.path.join(self.working_dir, "cc_I_001.fits"))
        self._image_ctor_params["imsize"] = image.imsize
        self._image_ctor_params["pixsize"] = image.pixsize
        self._image_ctor_params["pixref"] = image.pixref
        self._image_ctor_params["freq"] = image.freq
        self._image_ctor_params["pixrefval"] = image.pixrefval

    def create_mask_from_stacked_cc(self, cc_conv_cutoff_mjy=0.0075, kern_width=10):
        """
        Create mask on stacked Stokes I CC-components.
        cutoff: 0.0075 for 2251+158, 0.00025 for 0212+735
        """
        return get_mask_for_ccstack(self.ccfits_files["I"], cc_conv_cutoff_mjy,
                                    kern_width, imsize=self.mapsize_clean[0])

    def fill_cc_models(self, mask=None):
        for i in range(len(self.uvfits_files)):
            self.cc_models[i] = dict()
            for stk in self.stokes:
                if mask is not None:
                    filter_CC(self.ccfits_files[stk][i], mask)
                model = create_model_from_fits_file(self.ccfits_files[stk][i])
                self.cc_models[i][stk] = model

    # Must be done after ordinary stacks are constructed because we need
    # per-epoch masks!
    def create_stack_CConly_images(self):
        print("Convolving CC with beam and creating residual-less images")
        for stk in self.stokes:
            for i in range(len(self.uvfits_files)):
                if self.shifts is not None:
                    shift = self.shifts[i]
                else:
                    shift = None
                # If CCs were filtered, than we need to update difmap model
                # files using also shifts (difmap doesn't write shifts when
                # saving models)
                if self.mask_cc_models_using_their_stack:
                    CCFITS_to_difmap(self.ccfits_files[stk][i], self.difmap_files[stk][i], shift=shift)
                convert_difmap_model_file_to_CCFITS(self.difmap_files[stk][i], stk, self.mapsize_clean, self.beam,
                                                    self.uvfits_files[i], self.cconly_fits_files[stk][i], shift=shift)

        print("Creating I cc-only stack")
        ipol_image = Image()
        ipol_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="I")
        i_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.cconly_fits_files["I"]]
        ipol_image.image = np.sum([i_image.image for i_image in i_images], axis=0) / \
                           self.n_data
        self.cconly_stack_images["I"] = ipol_image

        print("Creating Q cc-only stack")
        q_image = Image()
        q_image._construct(imsize=self._image_ctor_params["imsize"],
                           pixsize=self._image_ctor_params["pixsize"],
                           pixref=self._image_ctor_params["pixref"],
                           freq=self._image_ctor_params["freq"],
                           pixrefval=self._image_ctor_params["pixrefval"],
                           stokes="Q")
        q_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.cconly_fits_files["Q"]]
        q_image.image = np.sum([q_image.image for q_image in q_images], axis=0) / \
                        self.n_data
        self.cconly_stack_images["Q"] = q_image

        print("Creating U cc-only stack")
        u_image = Image()
        u_image._construct(imsize=self._image_ctor_params["imsize"],
                           pixsize=self._image_ctor_params["pixsize"],
                           pixref=self._image_ctor_params["pixref"],
                           freq=self._image_ctor_params["freq"],
                           pixrefval=self._image_ctor_params["pixrefval"],
                           stokes="U")
        u_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.cconly_fits_files["U"]]
        u_image.image = np.sum([u_image.image for u_image in u_images], axis=0) \
                        / self.n_data
        self.cconly_stack_images["U"] = u_image

        if "V" in self.stokes:
            print("Creating V cc-only stack")
            v_image = Image()
            v_image._construct(imsize=self._image_ctor_params["imsize"],
                               pixsize=self._image_ctor_params["pixsize"],
                               pixref=self._image_ctor_params["pixref"],
                               freq=self._image_ctor_params["freq"],
                               pixrefval=self._image_ctor_params["pixrefval"],
                               stokes="V")
            v_images = [create_image_from_fits_file(ccfits_file) for ccfits_file
                        in self.cconly_fits_files["V"]]
            v_image.image = np.sum([v_image.image for v_image in v_images], axis=0) \
                            / self.n_data
            self.cconly_stack_images["V"] = v_image

        print("Creating PPOL cc-only stack")
        ppol_image = Image()
        ppol_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="PPOL")
        ppol_array = np.hypot(self.cconly_stack_images["Q"].image,
                              self.cconly_stack_images["U"].image)
        ppol_image.image = ppol_array
        self.cconly_stack_images["PPOL"] = ppol_image

        print("Creating FPOL cc-only stack")
        fpol_image = Image()
        fpol_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="FPOL")
        fpol_array = ppol_array/self.cconly_stack_images["I"].image
        fpol_image.image = fpol_array
        self.cconly_stack_images["FPOL"] = fpol_image

        print("Creating PANG cc-only stack")
        pang_image = Image()
        pang_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="FPOL")
        pang_array = 0.5*np.arctan2(self.cconly_stack_images["U"].image,
                                    self.cconly_stack_images["Q"].image)
        pang_image.image = pang_array
        self.cconly_stack_images["PANG"] = pang_image


        print("Creating PPOL2 & FPOL2 cc-only stack")
        ppol2_image = Image()
        ppol2_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="PPOL2")
        # This will be list of PPOL, PANG and FPOL arrays masked according to
        # there's own epoch polarization mask
        ppol2_arrays = list()
        fpol2_arrays = list()
        pang2_arrays = list()

        for i, i_image, q_image, u_image in zip(range(len(i_images)), i_images, q_images, u_images):
            ppol2_array = np.ma.array(np.hypot(q_image.image, u_image.image), mask=self.per_epoch_P_mask[i])
            ppol2_arrays.append(ppol2_array)
            fpol2_arrays.append(ppol2_array/i_image.image)
            pang2_array = 0.5*np.arctan2(u_image.image, q_image.image)
            pang2_array = np.ma.array(pang2_array, mask=self.per_epoch_P_mask[i])
            pang2_arrays.append(pang2_array)

        ppol2_image.image = stat_of_masked(ppol2_arrays, stat="median",
                                           n_epochs_not_masked_min=self.n_epochs_not_masked_min)
        self.cconly_stack_images["PPOL2"] = ppol2_image


        fpol2_image = Image()
        fpol2_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="FPOL2")
        fpol2_image.image = stat_of_masked(fpol2_arrays, stat="median",
                                           n_epochs_not_masked_min=self.n_epochs_not_masked_min)
        self.cconly_stack_images["FPOL2"] = fpol2_image


        print("Creating STD FPOL cc-only image")
        stdfpol_image = Image()
        stdfpol_image._construct(imsize=self._image_ctor_params["imsize"],
                                 pixsize=self._image_ctor_params["pixsize"],
                                 pixref=self._image_ctor_params["pixref"],
                                 freq=self._image_ctor_params["freq"],
                                 pixrefval=self._image_ctor_params["pixrefval"],
                                 stokes="FPOLSTD")
        stdfpol_image.image = stat_of_masked(fpol2_arrays, stat="std",
                                             n_epochs_not_masked_min=self.n_epochs_not_masked_min_std)
        self.cconly_stack_images["FPOLSTD"] = stdfpol_image

        print("Creating STD PPOL2 cc-only image")
        stdppol_image = Image()
        stdppol_image._construct(imsize=self._image_ctor_params["imsize"],
                                 pixsize=self._image_ctor_params["pixsize"],
                                 pixref=self._image_ctor_params["pixref"],
                                 freq=self._image_ctor_params["freq"],
                                 pixrefval=self._image_ctor_params["pixrefval"],
                                 stokes="PPOLSTD")
        stdppol_image.image = stat_of_masked(ppol2_arrays, stat="std",
                                             n_epochs_not_masked_min=self.n_epochs_not_masked_min_std)
        self.cconly_stack_images["PPOLSTD"] = stdppol_image







    def create_stack_residual_images(self):
        print("Creating I residual stack")
        ipol_image = Image()
        ipol_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="I")
        i_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.residuals_fits_files["I"]]
        ipol_image.image = np.sum([i_image.image for i_image in i_images], axis=0) / \
                           self.n_data
        self.stack_residuals_images["I"] = ipol_image

        print("Creating Q residual stack")
        q_image = Image()
        q_image._construct(imsize=self._image_ctor_params["imsize"],
                           pixsize=self._image_ctor_params["pixsize"],
                           pixref=self._image_ctor_params["pixref"],
                           freq=self._image_ctor_params["freq"],
                           pixrefval=self._image_ctor_params["pixrefval"],
                           stokes="Q")
        q_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.residuals_fits_files["Q"]]
        q_image.image = np.sum([q_image.image for q_image in q_images], axis=0) / \
                        self.n_data
        self.stack_residuals_images["Q"] = q_image

        print("Creating U residual stack")
        u_image = Image()
        u_image._construct(imsize=self._image_ctor_params["imsize"],
                           pixsize=self._image_ctor_params["pixsize"],
                           pixref=self._image_ctor_params["pixref"],
                           freq=self._image_ctor_params["freq"],
                           pixrefval=self._image_ctor_params["pixrefval"],
                           stokes="U")
        u_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.residuals_fits_files["U"]]
        u_image.image = np.sum([u_image.image for u_image in u_images], axis=0) \
                        / self.n_data
        self.stack_residuals_images["U"] = u_image

        if "V" in self.stokes:
            print("Creating V residual stack")
            v_image = Image()
            v_image._construct(imsize=self._image_ctor_params["imsize"],
                               pixsize=self._image_ctor_params["pixsize"],
                               pixref=self._image_ctor_params["pixref"],
                               freq=self._image_ctor_params["freq"],
                               pixrefval=self._image_ctor_params["pixrefval"],
                               stokes="V")
            v_images = [create_image_from_fits_file(ccfits_file) for ccfits_file
                        in self.residuals_fits_files["V"]]
            v_image.image = np.sum([v_image.image for v_image in v_images], axis=0) \
                            / self.n_data
            self.stack_residuals_images["V"] = v_image

    def create_stack_images(self):
        print("Creating I stack")
        ipol_image = Image()
        ipol_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="I")
        i_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.ccfits_files["I"]]
        ipol_image.image = np.sum([i_image.image for i_image in i_images], axis=0) /\
                           self.n_data
        self.stack_images["I"] = ipol_image

        print("Creating Q stack")
        q_image = Image()
        q_image._construct(imsize=self._image_ctor_params["imsize"],
                           pixsize=self._image_ctor_params["pixsize"],
                           pixref=self._image_ctor_params["pixref"],
                           freq=self._image_ctor_params["freq"],
                           pixrefval=self._image_ctor_params["pixrefval"],
                           stokes="Q")
        q_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.ccfits_files["Q"]]
        q_image.image = np.sum([q_image.image for q_image in q_images], axis=0) /\
                    self.n_data
        self.stack_images["Q"] = q_image

        print("Creating U stack")
        u_image = Image()
        u_image._construct(imsize=self._image_ctor_params["imsize"],
                           pixsize=self._image_ctor_params["pixsize"],
                           pixref=self._image_ctor_params["pixref"],
                           freq=self._image_ctor_params["freq"],
                           pixrefval=self._image_ctor_params["pixrefval"],
                           stokes="U")
        u_images = [create_image_from_fits_file(ccfits_file) for ccfits_file in
                    self.ccfits_files["U"]]
        u_image.image = np.sum([u_image.image for u_image in u_images], axis=0)\
                        / self.n_data
        self.stack_images["U"] = u_image

        if "V" in self.stokes:
            print("Creating V stack")
            v_image = Image()
            v_image._construct(imsize=self._image_ctor_params["imsize"],
                               pixsize=self._image_ctor_params["pixsize"],
                               pixref=self._image_ctor_params["pixref"],
                               freq=self._image_ctor_params["freq"],
                               pixrefval=self._image_ctor_params["pixrefval"],
                               stokes="V")
            v_images = [create_image_from_fits_file(ccfits_file) for ccfits_file
                        in self.ccfits_files["V"]]
            v_image.image = np.sum([v_image.image for v_image in v_images], axis=0)\
                            / self.n_data
            self.stack_images["V"] = v_image

        print("Creating raw PPOL stack")
        ppol_image = Image()
        ppol_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="RPPOL")
        ppol_array = np.hypot(self.stack_images["Q"].image,
                              self.stack_images["U"].image)
        ppol_image.image = ppol_array
        self.stack_images["RPPOL"] = ppol_image

        print("Creating bias-corrected PPOL stack")
        ppol_image = Image()
        ppol_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="PPOL")
        if not self.omit_residuals:
            ppol_array = correct_ppol_bias(ipol_image.image, ppol_array,
                                           q_image.image, u_image.image,
                                           self._npixels_beam)

        ppol_image.image = ppol_array
        self.stack_images["PPOL"] = ppol_image

        print("Creating FPOL stack")
        fpol_image = Image()
        fpol_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="FPOL")
        fpol_array = ppol_array/self.stack_images["I"].image
        fpol_image.image = fpol_array
        self.stack_images["FPOL"] = fpol_image

        print("Creating PANG stack")
        pang_image = Image()
        pang_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="PANG")
        pang_array = 0.5*np.arctan2(self.stack_images["U"].image,
                                    self.stack_images["Q"].image)
        pang_image.image = pang_array
        self.stack_images["PANG"] = pang_image

        print("Creating PPOL2 stack")
        ppol2_image = Image()
        ppol2_image._construct(imsize=self._image_ctor_params["imsize"],
                              pixsize=self._image_ctor_params["pixsize"],
                              pixref=self._image_ctor_params["pixref"],
                              freq=self._image_ctor_params["freq"],
                              pixrefval=self._image_ctor_params["pixrefval"],
                              stokes="PPOL2")
        # This will be list of PPOL, PANG and FPOL arrays masked according to
        # there's own epoch polarization mask
        ppol2_arrays = list()
        fpol2_arrays = list()
        pang2_arrays = list()

        for i, i_image, q_image, u_image in zip(range(len(i_images)), i_images, q_images, u_images):

            if self.omit_residuals:
                n_sigma = 1
                residual_images = {"I": create_image_from_fits_file(self.residuals_fits_files["I"][i]).image,
                                   "Q": create_image_from_fits_file(self.residuals_fits_files["Q"][i]).image,
                                   "U": create_image_from_fits_file(self.residuals_fits_files["U"][i]).image}
            else:
                n_sigma = 3
                residual_images = None

            ppol2_mask_dict, ppol_quantile = pol_mask({"I": i_image.image, "Q": q_image.image, "U": u_image.image},
                                                      self._npixels_beam, n_sigma=n_sigma, return_quantile=True,
                                                      residual_images=residual_images)
            # Mask before correction for bias
            ppol2_array = np.ma.array(np.hypot(q_image.image, u_image.image), mask=ppol2_mask_dict["P"])
            if not self.omit_residuals:
                ppol2_array = correct_ppol_bias(i_image.image, ppol2_array, q_image.image, u_image.image, self._npixels_beam)
            ppol2_arrays.append(ppol2_array)

            fpol2_arrays.append(ppol2_array/i_image.image)

            pang2_array = 0.5*np.arctan2(u_image.image, q_image.image)
            pang2_array = np.ma.array(pang2_array, mask=ppol2_mask_dict["P"])
            pang2_arrays.append(pang2_array)

            self.per_epoch_P_mask.append(ppol2_mask_dict["P"])
            self.per_epoch_I_mask.append(ppol2_mask_dict["I"])

        ppol2_image.image = stat_of_masked(ppol2_arrays, stat="median",
                                           n_epochs_not_masked_min=self.n_epochs_not_masked_min)
        self.stack_images["PPOL2"] = ppol2_image

        print("Creating # epochs not masked image")
        nepochs_image = Image()
        nepochs_image._construct(imsize=self._image_ctor_params["imsize"],
                                 pixsize=self._image_ctor_params["pixsize"],
                                 pixref=self._image_ctor_params["pixref"],
                                 freq=self._image_ctor_params["freq"],
                                 pixrefval=self._image_ctor_params["pixrefval"],
                                 stokes="NEPOCHS")
        nepochs_image.image = image_of_nepochs_not_masked(ppol2_arrays)
        self.stack_images["NEPOCHS"] = nepochs_image

        print("Creating PANG2 stack")
        pang2_image = Image()
        pang2_image._construct(imsize=self._image_ctor_params["imsize"],
                               pixsize=self._image_ctor_params["pixsize"],
                               pixref=self._image_ctor_params["pixref"],
                               freq=self._image_ctor_params["freq"],
                               pixrefval=self._image_ctor_params["pixrefval"],
                               stokes="PANG2")
        pang2_image.image = stat_of_masked(pang2_arrays, stat="scipy_circmean",
                                           n_epochs_not_masked_min=self.n_epochs_not_masked_min)
        self.stack_images["PANG2"] = pang2_image

        print("Creating STD PANG image")
        std_pang_image = Image()
        std_pang_image._construct(imsize=self._image_ctor_params["imsize"],
                               pixsize=self._image_ctor_params["pixsize"],
                               pixref=self._image_ctor_params["pixref"],
                               freq=self._image_ctor_params["freq"],
                               pixrefval=self._image_ctor_params["pixrefval"],
                               stokes="PANGSTD")
        std_pang_image.image = stat_of_masked(pang2_arrays, stat="scipy_circstd",
                                              n_epochs_not_masked_min=self.n_epochs_not_masked_min_std)
        self.stack_images["PANGSTD"] = std_pang_image

        print("Creating FPOL2 stack")
        fpol2_image = Image()
        fpol2_image._construct(imsize=self._image_ctor_params["imsize"],
                               pixsize=self._image_ctor_params["pixsize"],
                               pixref=self._image_ctor_params["pixref"],
                               freq=self._image_ctor_params["freq"],
                               pixrefval=self._image_ctor_params["pixrefval"],
                               stokes="FPOL2")
        fpol2_image.image = stat_of_masked(fpol2_arrays, stat="median",
                                           n_epochs_not_masked_min=self.n_epochs_not_masked_min)
        self.stack_images["FPOL2"] = fpol2_image

        print("Creating STD FPOL image")
        stdfpol_image = Image()
        stdfpol_image._construct(imsize=self._image_ctor_params["imsize"],
                                 pixsize=self._image_ctor_params["pixsize"],
                                 pixref=self._image_ctor_params["pixref"],
                                 freq=self._image_ctor_params["freq"],
                                 pixrefval=self._image_ctor_params["pixrefval"],
                                 stokes="FPOLSTD")
        stdfpol_image.image = stat_of_masked(fpol2_arrays, stat="std",
                                             n_epochs_not_masked_min=self.n_epochs_not_masked_min_std)
        self.stack_images["FPOLSTD"] = stdfpol_image

        print("Creating STD PPOL2 image")
        stdppol_image = Image()
        stdppol_image._construct(imsize=self._image_ctor_params["imsize"],
                                 pixsize=self._image_ctor_params["pixsize"],
                                 pixref=self._image_ctor_params["pixref"],
                                 freq=self._image_ctor_params["freq"],
                                 pixrefval=self._image_ctor_params["pixrefval"],
                                 stokes="PPOLSTD")
        stdppol_image.image = stat_of_masked(ppol2_arrays, stat="std",
                                             n_epochs_not_masked_min=self.n_epochs_not_masked_min_std)
        self.stack_images["PPOLSTD"] = stdppol_image

        if self.omit_residuals:
            residual_images = {stokes: self.stack_residuals_images[stokes].image for stokes in ("I", "Q", "U")}
            n_sigma = 1
        else:
            residual_images = None
            n_sigma = 4

        pang_mask_dict, ppol_quantile = pol_mask({stokes: self.stack_images[stokes].image for stokes in
                                                 ("I", "Q", "U")}, self._npixels_beam, n_sigma=n_sigma, return_quantile=True,
                                                 residual_images=residual_images)
        self.stack_images["P_mask"] = pang_mask_dict["P"]
        self.stack_images["I_mask"] = pang_mask_dict["I"]
        self.stack_images["P_quantile"] = ppol_quantile

    def plot_stack_images(self, save_fn, outdir=None):

        if outdir is None:
            outdir = self.working_dir

        ipol_image = self.stack_images["I"]
        ipol_dimage = self.stack_residuals_images["I"]
        ipol_cconly_image = self.cconly_stack_images["I"]
        ppol_image = self.stack_images["PPOL"]
        ppol_cconly_image = self.cconly_stack_images["PPOL"]
        qpol_dimage = self.stack_residuals_images["Q"]
        upol_dimage = self.stack_residuals_images["U"]
        fpol_image = self.stack_images["FPOL"]
        fpol_cconly_image = self.cconly_stack_images["FPOL"]
        pang_image = self.stack_images["PANG"]
        pang_cconly_image = self.cconly_stack_images["PANG"]
        ppol2_image = self.stack_images["PPOL2"]
        nepochs_image = self.stack_images["NEPOCHS"]
        fpol2_image = self.stack_images["FPOL2"]
        pang2_image = self.stack_images["PANG2"]
        std_pang_image = self.stack_images["PANGSTD"]
        std_fpol_image = self.stack_images["FPOLSTD"]
        std_ppol_image = self.stack_images["PPOLSTD"]
        ppol_quantile = self.stack_images["P_quantile"]
        ipol_mask = self.stack_images["I_mask"]
        ppol_mask = self.stack_images["P_mask"]

        if not self.omit_residuals:
            std = find_image_std(ipol_image.image, beam_npixels=self._npixels_beam)
        else:
            # Estimate std using residuals images
            std = np.median([mad_std(ipol_dimage.image), mad_std(qpol_dimage.image), mad_std(upol_dimage.image)])
        blc, trc = find_bbox(ipol_image.image, level=4*std, min_maxintensity_mjyperbeam=6*std,
                             min_area_pix=4*self._npixels_beam, delta=10)

        # iplot expects AIPS-like behaviour of BLC, TRC
        if blc[0] == 0:
            blc = (1, blc[1])
        if blc[1] == 0:
            blc = (blc[0], 1)

        try:

            print("Plotting stack images with blc={}, trc={}".format(blc, trc))
            # I (1 contour) + P (color) + EVPA (vector)
            fig = iplot(ppol_image.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=ppol_quantile, blc=blc, trc=trc,
                        close=False, contour_color='black',
                        plot_colorbar=False)
            # Add IPOL single contour and vectors of PANG with colors of PPOL
            fig = iplot(contours=ipol_image.image, vectors=pang_image.image,
                        x=ipol_image.x, y=ipol_image.y, vinc=4,  contour_linewidth=0.25,
                        vectors_mask=ppol_mask, abs_levels=[3*std], blc=blc, trc=trc,
                        beam=self.beam, close=False, show_beam=True, show=True,
                        contour_color='black', fig=fig, vector_color="black", plot_colorbar=False)
            axes = fig.get_axes()[0]
            axes.invert_xaxis()
            fig.savefig(os.path.join(outdir, "{}_ppol.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            if "V" in self.stokes:
                vpol_image = self.stack_images["V"]
                max_abs_v = np.ma.max(np.ma.abs(np.ma.array(1000*vpol_image.image, mask=self.i_mask)))
                v_std = mad_std(np.ma.array(1000*vpol_image.image, mask=~self.i_mask).compressed())
                max_snr = max_abs_v/v_std
                fig = iplot(ipol_image.image, 1000*vpol_image.image/v_std, x=ipol_image.x, y=ipol_image.y,
                            min_abs_level=3*std, colors_mask=ipol_mask, blc=blc, trc=trc,
                            beam=self.beam, close=False, colorbar_label=r"$\frac{V}{\sigma_{V}}$",
                            show_beam=True, show=True, cmap='bwr', color_clim=[-max_snr, max_snr],
                            contour_color='black', plot_colorbar=True, contour_linewidth=0.25)
                fig.savefig(os.path.join(outdir, "{}_vpol.png".format(save_fn)), dpi=600, bbox_inches="tight")
                plt.close()

            # Add IPOL single contour and colors of FPOL with colorbar
            # max_fpol_range, _ = choose_range_from_positive_tailed_distribution(np.ma.array(fpol_image.image, mask=pang_mask).compressed())
            # fpol_mask = np.logical_or(pang_mask, fpol_image.image > max_fpol_range)
            fig = iplot(ipol_image.image, fpol_image.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=ppol_mask, color_clim=[0, 0.7], blc=blc, trc=trc,
                        beam=self.beam, close=False, colorbar_label="m", show_beam=True, show=True,
                        cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                        contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_fpol.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            fig = iplot(ppol2_image.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=ppol_quantile, blc=blc, trc=trc,
                        close=False, contour_color='black',
                        plot_colorbar=False)
            fig = iplot(contours=ipol_image.image, vectors=pang2_image.image,
                        x=ipol_image.x, y=ipol_image.y, vinc=4,  contour_linewidth=0.25,
                        vectors_mask=pang2_image.image.mask, abs_levels=[3*std], blc=blc, trc=trc,
                        beam=self.beam, close=False, show_beam=True, show=True,
                        contour_color='black', fig=fig, vector_color="black", plot_colorbar=False)
            axes = fig.get_axes()[0]
            axes.invert_xaxis()
            fig.savefig(os.path.join(outdir, "{}_ppol2.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            # max_fpol_range, _ = choose_range_from_positive_tailed_distribution(np.ma.array(fpol_image.image, mask=pang_mask).compressed())
            # fpol_mask = np.logical_or(pang_mask, fpol_image.image > max_fpol_range)
            fig = iplot(contours=ipol_image.image, colors=nepochs_image.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=nepochs_image.image.mask, color_clim=None, blc=blc, trc=trc,
                        beam=self.beam, close=False, colorbar_label="# nepochs", show_beam=True, show=True,
                        cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                        contour_linewidth=0.25, n_discrete_colors=np.max(nepochs_image.image.flatten()))
            fig.savefig(os.path.join(outdir, "{}_nepochs.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            # max_fpol_range, _ = choose_range_from_positive_tailed_distribution(np.ma.array(fpol_image.image, mask=pang_mask).compressed())
            # fpol_mask = np.logical_or(pang_mask, fpol_image.image > max_fpol_range)
            fig = iplot(contours=ipol_image.image, colors=fpol2_image.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=fpol2_image.image.mask, color_clim=[0, 0.7], blc=blc, trc=trc,
                        beam=self.beam, close=False, colorbar_label="m", show_beam=True, show=True,
                        cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                        contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_fpol2.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            fig = iplot(contours=ipol_image.image, colors=std_ppol_image.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=std_ppol_image.image.mask, color_clim=None, blc=blc, trc=trc,
                        beam=self.beam, close=False, colorbar_label=r"$\sigma_{P}$", show_beam=True, show=True,
                        cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                        contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_ppolstd.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            fig = iplot(contours=ipol_image.image, colors=std_fpol_image.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=std_fpol_image.image.mask, color_clim=None, blc=blc, trc=trc,
                        beam=self.beam, close=False, colorbar_label=r"$\sigma_{m}$", show_beam=True, show=True,
                        cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                        contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_fpolstd.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            # max_fpol_range, _ = choose_range_from_positive_tailed_distribution(np.ma.array(fpol_image.image, mask=pang_mask).compressed())
            # fpol_mask = np.logical_or(pang_mask, fpol_image.image > max_fpol_range)
            fig = iplot(contours=ipol_image.image, colors=np.rad2deg(std_pang_image.image), x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=std_pang_image.image.mask, color_clim=None, blc=blc, trc=trc,
                        beam=self.beam, close=False, colorbar_label=r"$\sigma_{\rm EVPA},$ $^{\circ}$", show_beam=True, show=True,
                        cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                        contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_pangstd.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            print("Plotting CC-only stack images with blc={}, trc={}".format(blc, trc))
            # I (1 contour) + P (color) + EVPA (vector)
            fig = iplot(ppol_cconly_image.image, x=ipol_cconly_image.x, y=ipol_cconly_image.y,
                        min_abs_level=ppol_quantile, blc=blc, trc=trc,
                        close=False, contour_color='black',
                        plot_colorbar=False)
            # Add IPOL single contour and vectors of PANG with colors of PPOL
            fig = iplot(contours=ipol_cconly_image.image, vectors=pang_cconly_image.image,
                        x=ipol_cconly_image.x, y=ipol_cconly_image.y, vinc=4,  contour_linewidth=0.25,
                        vectors_mask=ppol_mask, abs_levels=[3*std], blc=blc, trc=trc,
                        beam=self.beam, close=False, show_beam=True, show=True,
                        contour_color='black', fig=fig, vector_color="black", plot_colorbar=False)
            axes = fig.get_axes()[0]
            axes.invert_xaxis()
            fig.savefig(os.path.join(outdir, "{}_cconly_ppol.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            if "V" in self.stokes:
                vpol_cconly_image = self.cconly_stack_images["V"]
                max_abs_v = np.ma.max(np.ma.abs(np.ma.array(1000*vpol_cconly_image.image, mask=self.i_mask)))
                max_snr = max_abs_v/v_std
                fig = iplot(ipol_cconly_image.image, 1000*vpol_cconly_image.image/v_std,
                            x=ipol_cconly_image.x, y=ipol_cconly_image.y,
                            min_abs_level=3*std, colors_mask=ipol_mask, blc=blc, trc=trc,
                            beam=self.beam, close=False, colorbar_label=r"$\frac{V}{\sigma_{V}}$",
                            show_beam=True, show=True, cmap='bwr', color_clim=[-max_snr, max_snr],
                            contour_color='black', plot_colorbar=True, contour_linewidth=0.25)
                fig.savefig(os.path.join(outdir, "{}_cconly_vpol.png".format(save_fn)), dpi=600, bbox_inches="tight")
                plt.close()

            # Add IPOL single contour and colors of FPOL with colorbar
            # max_fpol_range, _ = choose_range_from_positive_tailed_distribution(np.ma.array(fpol_image.image, mask=pang_mask).compressed())
            # fpol_mask = np.logical_or(pang_mask, fpol_image.image > max_fpol_range)
            fig = iplot(ipol_cconly_image.image, fpol_cconly_image.image,
                        x=ipol_cconly_image.x, y=ipol_cconly_image.y,
                        min_abs_level=3*std, colors_mask=ppol_mask, color_clim=[0, 0.7], blc=blc, trc=trc,
                        beam=self.beam, close=False, colorbar_label="m", show_beam=True, show=True,
                        cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                        contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_cconly_fpol.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            blc_all = (1, 1)
            trc_all = (self.mapsize_clean[0], self.mapsize_clean[0])

            print("Plotting residuals stack images with blc={}, trc={}".format(blc, trc))

            fig = iplot(ipol_image.image, 1000*ipol_dimage.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=None, blc=blc_all, trc=trc_all,
                        beam=self.beam, close=False, colorbar_label=r"$I_{\rm resid}$, mJy/beam",
                        show_beam=True, show=True, cmap='bwr', color_clim=[-3000*std, 3000*std],
                        contour_color='black', plot_colorbar=True, contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_ipol_residuals.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            fig = iplot(ipol_image.image, 1000*qpol_dimage.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=None, blc=blc_all, trc=trc_all,
                        beam=self.beam, close=False, colorbar_label=r"$Q_{\rm resid}$, mJy/beam",
                        show_beam=True, show=True, cmap='bwr', color_clim=[-3000*std, 3000*std],
                        contour_color='black', plot_colorbar=True, contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_qpol_residuals.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

            fig = iplot(ipol_image.image, 1000*upol_dimage.image, x=ipol_image.x, y=ipol_image.y,
                        min_abs_level=3*std, colors_mask=None, blc=blc_all, trc=trc_all,
                        beam=self.beam, close=False, colorbar_label=r"$U_{\rm resid}$, mJy/beam",
                        show_beam=True, show=True, cmap='bwr', color_clim=[-3000*std, 3000*std],
                        contour_color='black', plot_colorbar=True, contour_linewidth=0.25)
            fig.savefig(os.path.join(outdir, "{}_upol_residuals.png".format(save_fn)), dpi=600, bbox_inches="tight")
            plt.close()

        except:
            pass

    def save_stack_images(self, save_fn, outdir=None):
        save_dict = dict()

        if outdir is None:
            outdir = self.working_dir

        for stokes in self.stack_images:
            # Mask can't be np.nan for int64
            if stokes == "NEPOCHS":
                save_dict.update({stokes: np.ma.filled(self.stack_images[stokes].image, 0)})
            # These are not masked, but masks for I & PPOL are saved and
            # masks with other parameters can be recovered using I, Q, U, RPPOL
            # (last is raw PPOL w/o bias correction)
            elif stokes in ("I", "Q", "U", "V", "RPPOL", "PPOL", "PANG", "FPOL"):
                save_dict.update({stokes: self.stack_images[stokes].image})
            # Others (scalar averaged) have masks
            elif stokes in ("PPOL2", "FPOL2", "PANG2", "PPOLSTD", "FPOLSTD", "PANGSTD"):
                save_dict.update({stokes: np.ma.filled(self.stack_images[stokes].image, np.nan)})
            elif stokes in ("I_mask", "P_mask"):
                save_dict.update({stokes: self.stack_images[stokes]})
            elif stokes == "P_quantile":
                pass
            else:
                raise Exception("This stokes ({}) is not supposed to be here!".format(stokes))

        np.savez_compressed(os.path.join(outdir, save_fn+"_stack.npz"),
                            **save_dict)

    def save_cconly_stack_images(self, save_fn, outdir=None):
        save_dict = dict()

        if outdir is None:
            outdir = self.working_dir

        for stokes in self.cconly_stack_images:
            # These are not masked, but masks for I & PPOL are saved and
            # masks with other parameters can be recovered using I, Q, U, RPPOL
            # (last is raw PPOL w/o bias correction)
            if stokes in ("I", "Q", "U", "V", "PPOL", "FPOL", "PANG"):
                save_dict.update({stokes: self.cconly_stack_images[stokes].image})
            # These are masked but try the same to save in npz
            elif stokes in ("PPOL2", "FPOL2", "PPOLSTD", "FPOLSTD"):
                save_dict.update({stokes: self.cconly_stack_images[stokes].image})
            else:
                raise Exception("This stokes ({}) is not supposed to be here!".format(stokes))

        np.savez_compressed(os.path.join(outdir, save_fn+"_cconly_stack.npz"),
                            **save_dict)

    def save_stack_images_in_fits(self, save_fn, outdir=None):

        if outdir is None:
            outdir = self.working_dir

        hdr = pf.open(self.ccfits_files["I"][0])[0].header

        for stokes in self.stack_images:
            # Mask can't be np.nan for int64
            if stokes == "NEPOCHS":
                hdu = pf.PrimaryHDU(data=np.ma.filled(self.stack_images[stokes].image, 0), header=hdr)
            # These are not masked, but masks for I & PPOL are saved and
            # masks with other parameters can be recovered using I, Q, U, RPPOL
            # (last is raw PPOL w/o bias correction)
            elif stokes in ("I", "Q", "U", "V", "RPPOL", "PPOL", "PANG", "FPOL"):
                hdu = pf.PrimaryHDU(data=self.stack_images[stokes].image, header=hdr)
            # Others (scalar averaged) have masks
            elif stokes in ("PPOL2", "FPOL2", "PANG2", "PPOLSTD", "FPOLSTD", "PANGSTD"):
                hdu = pf.PrimaryHDU(data=np.ma.filled(self.stack_images[stokes].image, np.nan), header=hdr)
            elif stokes in ("I_mask", "P_mask"):
                hdu = pf.PrimaryHDU(data=np.array(self.stack_images[stokes], dtype=int),
                                    header=hdr)
            elif stokes == "P_quantile":
                continue
            else:
                raise Exception("This stokes ({}) is not supposed to be here!".format(stokes))

            hdu.writeto(os.path.join(outdir, "{}_{}.fits".format(save_fn, stokes)), output_verify='ignore')

    def save_cconly_stack_images_in_fits(self, save_fn, outdir=None):

        if outdir is None:
            outdir = self.working_dir

        hdr = pf.open(self.ccfits_files["I"][0])[0].header

        for stokes in self.cconly_stack_images:
            # These are not masked, but masks for I & PPOL are saved and
            # masks with other parameters can be recovered using I, Q, U, RPPOL
            # (last is raw PPOL w/o bias correction)
            if stokes in ("I", "Q", "U", "V", "PPOL", "FPOL", "PANG"):
                hdu = pf.PrimaryHDU(data=self.cconly_stack_images[stokes].image, header=hdr)
            # Others (scalar averaged) have masks
            elif stokes in ("PPOL2", "FPOL2", "PPOLSTD", "FPOLSTD"):
                hdu = pf.PrimaryHDU(data=np.ma.filled(self.cconly_stack_images[stokes].image, np.nan), header=hdr)
            else:
                raise Exception("This stokes ({}) is not supposed to be here!".format(stokes))

            hdu.writeto(os.path.join(outdir, "{}_{}.fits".format(save_fn, stokes)), output_verify='ignore')

    def remove_cc_fits(self):
        """
        Remove FITS files with CC for all Stokes.
        """
        for stokes in self.stokes:
            for cc_fits in self.ccfits_files[stokes]:
                os.unlink(cc_fits)

    def move_cc_fits(self, destination_directory):
        """
        Rename FITS file with CC for all Stokes for later use.
        """
        for stokes in self.stokes:
            for cc_fits in self.ccfits_files[stokes]:
                prev_dir, name = os.path.split(cc_fits)
                shutil.move(cc_fits, os.path.join(destination_directory, name))
