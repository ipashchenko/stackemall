import os
import json
import glob
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import numpy as np
import astropy.io.fits as pf
from stack_utils import (parse_source_list, convert_mojave_epoch, choose_mapsize,
                         find_image_std, find_bbox, stat_of_masked,
                         choose_range_from_positive_tailed_distribution,
                         get_beam_info_by_dec, get_inner_jet_PA)
from create_artificial_data import (ArtificialDataCreator, rename_mc_stack_files)
from stack import Stack
from stack_utils import pol_mask, correct_ppol_bias
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 've/vlbi_errors')
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from image import plot as iplot


def move_result_files_to_jet(source, calculon_dir, jet_dir):
    """
    Move result files from CALCULON to jet.

    This can be used to move some resulting files to specified place.

    :param source:
        Name of the source. Corresponding directory will be created if necessary.
    :param calculon_dir:
        Directory on CALCULON with result files.
    :param jet_dir:
        Directory on jet where to move files.
    """
    if not os.path.exists(os.path.join(jet_dir, source)):
        os.mkdir(os.path.join(jet_dir, source))
    files = list()
    # Pictures of original stacks
    for item in ("ppol", "fpol", "ppol2", "fpol2", "pangstd", "fpolstd", "nepochs"):
        files.append("{}_original_{}.png".format(source, item))

    if os.path.exists(os.path.join(calculon_dir, "{}_original_vpol.png".format(source))):
        files.append("{}_original_vpol.png".format(source))

    # Pictures of errors
    for item in ("ipol", "ppol", "fpol", "pang", "ppol2", "fpol2", "pang2", "pangstd", "fpolstd"):
        files.append("{}_{}_errors.png".format(source, item))
    # FITS files with errors of stacks
    for stokes in ("I", "PPOL", "PANG", "FPOL", "PPOL2", "FPOL2", "PANG2", "FPOLSTD", "PANGSTD"):
        files.append("{}_{}_stack_error.fits".format(source, stokes))
    # npz file with errors of stacks
    files.append("{}_stack_errors.npz".format(source))

    # Original stacks
    files.append("{}_original_stack.npz".format(source))
    for stokes in ("I", "Q", "U", "RPPOL", "PPOL", "PANG", "FPOL", "PPOL2", "FPOL2", "PANG2", "FPOLSTD", "PANGSTD", "NEPOCHS"):
        files.append("{}_original_stack_{}.fits".format(source, stokes))
    files.append("{}_original_stack_I_mask.fits".format(source))
    files.append("{}_original_stack_P_mask.fits".format(source))

    if os.path.exists(os.path.join(calculon_dir, "{}_original_stack_V.fits".format(source))):
        files.append("{}_original_stack_V.fits".format(source))

    # Biases
    files.append("{}_stack_biases.npz".format(source))
    for stokes in ("I", "PPOL", "FPOL"):
        files.append("{}_{}_stack_bias.fits".format(source, stokes))
    for stokes in ("ipol", "ppol", "fpol"):
        files.append("{}_{}_bias.png".format(source, stokes))

    for file in files:
        shutil.move(os.path.join(calculon_dir, file), os.path.join(jet_dir, source, file))


class Simulation(object):
    def __init__(self, source, n_mc, common_mapsize_clean, common_beam,
                 source_epoch_core_offset_file, working_dir,
                 path_to_clean_script, shifts_errors_ell_bmaj,
                 shifts_errors_ell_bmin, shifts_errors_PA_file,
                 model_core_shifts_errors=True,
                 remove_artificial_uvfits_files=True,
                 create_original_V_stack=False,
                 path_to_uvfits_files="/mnt/jet1/yyk/VLBI/2cmVLBA/data",
                 omit_residuals=False, do_smooth=True):
        """
        :param source:
            String B1950 name of the source.
        :param n_mc:
            Number of realizations.
        :param common_mapsize_clean:
            Tuple of common image size and pixel size (mas).
        :param common_beam:
            Tuple of bmaj[mas], bmin[mas], bpa[deg] for common restoring beam.
        :param source_epoch_core_offset_file:
            File with source, epoch, core offset values.
        :param working_dir:
            Directory to store results.
        :param path_to_clean_script:
            Path to D. Homan difmap cleaning script.
        :param shifts_errors_ell_bmaj:
            Major axis of core shift error ellipse (mas).
        :param shifts_errors_ell_bmin:
            Minor axis of core shift error ellipse (mas).
        :param shifts_errors_PA_file:
            File with PA of inner jet for each source (possible epoch).
        :param model_core_shifts_errors: (optional)
            Boolean. Use parameters of error ellipse specified in previous 3
            arguments to model error of core offsets? (default: ``True``)
        :param remove_artificial_uvfits_files: (optional)
            Boolean. Remove created artificial UVFITS files? (default: ``True``)
        :param create_original_V_stack: (optional)
            Boolean. Create stacks of Stokes V? (default: ``False``)
        :param path_to_uvfits_files: (optional)
            Directory with UVFITS files. Individual files must be
            path_to_uvfits_files/source/epoch/source.u.epoch.uvf
        """
        self.source = source
        self.n_mc = n_mc
        self.common_mapsize_clean = common_mapsize_clean
        self.common_beam = common_beam
        self._npixels_beam = np.pi*common_beam[0]*common_beam[1]/common_mapsize_clean[1]**2
        self.working_dir = working_dir
        self.path_to_clean_script = path_to_clean_script
        self.omit_residuals = omit_residuals
        self.do_smooth = do_smooth
        self.uvfits_files = list()
        self.shifts = list()
        self.shifts_errors = list()
        self.model_core_shift_errors = model_core_shifts_errors
        df = parse_source_list(source_epoch_core_offset_file, source=source)
        df = df.drop_duplicates()
        for index, row in df.iterrows():
            epoch = convert_mojave_epoch(row['epoch'])
            self.shifts.append((row['shift_ra'], row['shift_dec']))
            uvfits_file = "{}/{}/{}/{}.u.{}.uvf".format(path_to_uvfits_files, source, epoch, source, epoch)
            self.uvfits_files.append(uvfits_file)
            # TODO: If per-epoch core shift errors are needed then change
            #  implementation of this function
            pa = get_inner_jet_PA(source, epoch, shifts_errors_PA_file)
            self.shifts_errors.append((shifts_errors_ell_bmaj, shifts_errors_ell_bmin, pa))

        # Template header used for saving results in FITS format
        self.hdr = None
        # Template image
        self.some_image = None
        self.remove_artificial_uvfits_files = remove_artificial_uvfits_files
        self.create_original_V_stack = create_original_V_stack
        # Original stack
        self.original_stack = None

    def create_artificial_uvdata(self, sigma_scale_amplitude, noise_scale,
                                 sigma_evpa_deg, VLBA_residual_Dterms_file=None,
                                 noise_from_V=True):
        if VLBA_residual_Dterms_file is not None:
            with open(VLBA_residual_Dterms_file, "r") as fo:
                d_term = json.load(fo)
        else:
            d_term = None

        for i, uvfits_file, shift in zip(range(len(self.uvfits_files)), self.uvfits_files, self.shifts):
            print("Creating {} artificial data sets from {} with applied shift = {}".format(n_mc, uvfits_file, shift))
            # Using already CLEANed models (when stack was created) for simulations
            # Shifts are accounted for here! (CC-models have core at (0,0)!)
            models = self.original_stack.cc_models[i]
            # No need to shift!
            shift = None
            creator = ArtificialDataCreator(uvfits_file, self.path_to_clean_script, self.common_mapsize_clean,
                                            self.common_beam, shift=shift, models=models, working_dir=self.working_dir,
                                            noise_from_V=noise_from_V)
            creator.mc_create_uvfits(n_mc=self.n_mc, d_term=d_term, sigma_scale_amplitude=sigma_scale_amplitude,
                                     noise_scale=noise_scale, sigma_evpa=sigma_evpa_deg,
                                     constant_dterm_amplitude=True,
                                     ignore_cross_hands=False)
            creator.remove_cc_fits()
        rename_mc_stack_files(self.working_dir)

    def create_original_stack(self, n_epochs_not_masked_min, n_epochs_not_masked_min_std):
        stack = Stack(self.uvfits_files, self.common_mapsize_clean, self.common_beam,
                      working_dir=self.working_dir, create_stacks=True,
                      shifts=self.shifts, path_to_clean_script=self.path_to_clean_script,
                      n_epochs_not_masked_min=n_epochs_not_masked_min,
                      n_epochs_not_masked_min_std=n_epochs_not_masked_min_std,
                      use_V=self.create_original_V_stack, omit_residuals=self.omit_residuals,
                      do_smooth=self.do_smooth)
        stack.save_stack_images("{}_original".format(self.source),
                                outdir=self.working_dir)
        stack.save_cconly_stack_images("{}_original".format(self.source),
                                       outdir=self.working_dir)
        stack.plot_stack_images("{}_original".format(self.source),
                                outdir=self.working_dir)
        stack.save_stack_images_in_fits("{}_original_stack".format(self.source),
                                        outdir=self.working_dir)
        stack.save_cconly_stack_images_in_fits("{}_original_cconly_stack".format(self.source),
                                               outdir=self.working_dir)
        self.hdr = pf.open(stack.ccfits_files["I"][0])[0].header
        self.some_image = create_clean_image_from_fits_file(stack.ccfits_files["I"][0])
        # Remove CLEAN FITS-files
        stack.remove_cc_fits()
        self.original_stack = stack

    def create_artificial_stacks(self, n_epochs_not_masked_min, n_epochs_not_masked_min_std):
        # Create images for artificial stacks
        if self.model_core_shift_errors:
            shifts_errors = self.shifts_errors
        else:
            shifts_errors = None
        for i in range(self.n_mc):
            data_dir = os.path.join(self.working_dir, "artificial_{}".format(str(i + 1).zfill(3)))
            uvfits_files = sorted(glob.glob(os.path.join(data_dir, "*uvf")))
            # Shifts are already accounted for (inserted while creating original stack and CC-models!)
            stack = Stack(uvfits_files, self.common_mapsize_clean, self.common_beam,
                          path_to_clean_script=self.path_to_clean_script,
                          shifts=None, shifts_errors=shifts_errors,
                          working_dir=data_dir, create_stacks=True,
                          n_epochs_not_masked_min=n_epochs_not_masked_min,
                          n_epochs_not_masked_min_std=n_epochs_not_masked_min_std,
                          omit_residuals=self.omit_residuals, do_smooth=self.do_smooth)
            stack.save_stack_images("{}_mc_images_{}".format(self.source, str(i + 1).zfill(3)),
                                    outdir=self.working_dir)
            # stack.save_cconly_stack_images("{}_mc_images_{}".format(self.source, str(i + 1).zfill(3)),
            #                                outdir=self.working_dir)
            # TODO: Do we need FITS files of artificial stacks?
            # stack.save_stack_images_in_fits(str(i+1).zfill(3))

            # Remove CLEAN FITS-files
            # stack.remove_cc_fits()

            # Move CC FITS files for later use
            cc_save_dir = os.path.join(self.working_dir, "CC_{}".format(str(i+1).zfill(3)))
            if not os.path.exists(cc_save_dir):
                os.mkdir(cc_save_dir)
            stack.move_cc_fits(cc_save_dir)

            if self.remove_artificial_uvfits_files:
                # Remove artificial data files
                for uvfits_file in uvfits_files:
                    os.unlink(uvfits_file)

    def create_individual_epoch_error_images(self, n_realizations_not_masked_min):
        for i_epoch in range(len(self.uvfits_files)):
            epoch_errors_dict = dict()
            ipol_arrays = list()
            ppol_arrays = list()
            fpol_arrays = list()
            pang_arrays = list()
            for i_real in range(self.n_mc):
                i_cc_fits_file = os.path.join(self.working_dir,
                                              "CC_{}".format(str(i_real+1).zfill(3)),
                                              "cc_{}_{}.fits".format("I", str(i_epoch+1).zfill(3)))
                q_cc_fits_file = os.path.join(self.working_dir,
                                              "CC_{}".format(str(i_real+1).zfill(3)),
                                              "cc_{}_{}.fits".format("Q", str(i_epoch+1).zfill(3)))
                u_cc_fits_file = os.path.join(self.working_dir,
                                              "CC_{}".format(str(i_real+1).zfill(3)),
                                              "cc_{}_{}.fits".format("U", str(i_epoch+1).zfill(3)))
                i_image = create_image_from_fits_file(i_cc_fits_file)
                q_image = create_image_from_fits_file(q_cc_fits_file)
                u_image = create_image_from_fits_file(u_cc_fits_file)

                ppol_mask_dict, ppol_quantile = pol_mask({"I": i_image.image, "Q": q_image.image, "U": u_image.image},
                                                         self._npixels_beam, n_sigma=3, return_quantile=True)
                ipol_array = np.ma.array(i_image.image, mask=ppol_mask_dict["I"])
                ipol_arrays.append(ipol_array)

                # Mask before correction for bias
                ppol_array = np.ma.array(np.hypot(q_image.image, u_image.image), mask=ppol_mask_dict["P"])
                ppol_array = correct_ppol_bias(i_image.image, ppol_array, q_image.image, u_image.image, self._npixels_beam)
                ppol_arrays.append(ppol_array)

                fpol_arrays.append(ppol_array/i_image.image)

                pang_array = 0.5 * np.arctan2(u_image.image, q_image.image)
                pang_array = np.ma.array(pang_array, mask=ppol_mask_dict["P"])
                pang_arrays.append(pang_array)

            # Create error images for given epoch
            std_ipol = stat_of_masked(ipol_arrays, stat="std", n_epochs_not_masked_min=n_realizations_not_masked_min)
            std_ppol = stat_of_masked(ppol_arrays, stat="std", n_epochs_not_masked_min=n_realizations_not_masked_min)
            std_fpol = stat_of_masked(fpol_arrays, stat="std", n_epochs_not_masked_min=n_realizations_not_masked_min)
            std_pang = stat_of_masked(fpol_arrays, stat="scipy_circstd", n_epochs_not_masked_min=n_realizations_not_masked_min)
            std_dict = {"IPOL": std_ipol, "PPOL": std_ppol, "FPOL": std_fpol, "PANG": std_pang}

            # Save it to FITS
            save_dir = os.path.join(self.working_dir, "epoch_errors_{}".format(str(i_epoch+1).zfill(3)))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for stokes in ("IPOL", "PPOL", "FPOL", "PANG"):
                hdu = pf.PrimaryHDU(data=np.ma.filled(std_dict[stokes], np.nan), header=self.hdr)
                epoch_errors_dict[stokes] = np.ma.filled(std_dict[stokes], np.nan)
                hdu.writeto(os.path.join(save_dir, "{}_{}_epoch_errors.fits".format(self.source, stokes)), output_verify='ignore')
                np.savez_compressed(os.path.join(save_dir, "{}_epoch_errors.npz".format(self.source)),
                                    **epoch_errors_dict)

        # Remove directories with CC FITS files
        for i_real in range(self.n_mc):
            shutil.rmtree(os.path.join(self.working_dir, "CC_{}".format(str(i_real+1).zfill(3))))

    def create_errors_images(self, create_pictures=True):

        some_image = self.some_image
        beam = self.common_beam
        original_images = np.load(os.path.join(self.working_dir, "{}_original_stack.npz".format(self.source)))
        original_cconly_images = np.load(os.path.join(self.working_dir, "{}_original_cconly_stack.npz".format(self.source)))

        errors_dict = dict()
        biases_dict = dict()
        for stokes in ("I", "PPOL", "PANG", "FPOL", "PPOL2", "FPOL2", "PANG2", "PPOLSTD", "FPOLSTD", "PANGSTD"):
            mc_images = list()
            for i in range(self.n_mc):
                npz = np.load(os.path.join(self.working_dir, "{}_mc_images_{}_stack.npz".format(self.source, str(i + 1).zfill(3))))
                array = npz[stokes]
                # This stacks are not masked => use trivial mask with zeros
                if stokes in ("I", "PPOL", "PANG", "FPOL"):
                    array = np.ma.array(array, mask=np.zeros(array.shape, dtype=bool))
                elif stokes in ("PPOL2", "FPOL2", "PANG2", "PPOLSTD", "FPOLSTD", "PANGSTD"):
                    # Masked array with masked values having nans
                    array = np.ma.array(array, mask=np.isnan(array))
                else:
                    raise Exception("{} no allowed, smth is going wrong!".format(stokes))
                mc_images.append(array)

            # Find errors
            if stokes not in ("PANG", "PANG2"):
                std = stat_of_masked(mc_images, stat="std",
                                     n_epochs_not_masked_min=n_epochs_not_masked_min_std)
            else:
                std = stat_of_masked(mc_images, stat="scipy_circstd",
                                     n_epochs_not_masked_min=n_epochs_not_masked_min_std)

            hdu = pf.PrimaryHDU(data=np.ma.filled(std, np.nan), header=self.hdr)
            errors_dict[stokes] = np.ma.filled(std, np.nan)
            hdu.writeto(os.path.join(self.working_dir, "{}_{}_stack_error.fits".format(self.source, stokes)), output_verify='ignore')

            # Find biases
            if stokes in ("I", "PPOL", "FPOL"):
                mean = stat_of_masked(mc_images, stat="mean",
                                      n_epochs_not_masked_min=n_epochs_not_masked_min_std)
                # FIXME: Here must be beam-convolved CC-images (that are used to build artificial data)!
                bias = mean - original_cconly_images[stokes]
                biases_dict[stokes] = np.ma.filled(bias, np.nan)
                hdu = pf.PrimaryHDU(data=np.ma.filled(bias, np.nan), header=self.hdr)
                hdu.writeto(os.path.join(self.working_dir, "{}_{}_stack_bias.fits".format(self.source, stokes)), output_verify='ignore')

        np.savez_compressed(os.path.join(self.working_dir, "{}_stack_errors.npz".format(self.source)),
                            **errors_dict)
        np.savez_compressed(os.path.join(self.working_dir, "{}_stack_biases.npz".format(self.source)),
                            **biases_dict)

        # Remove directories with artificial files optionally
        if self.remove_artificial_uvfits_files:
            for i in range(self.n_mc):
                data_dir = os.path.join(self.working_dir, "artificial_{}".format(str(i + 1).zfill(3)))
                shutil.rmtree(data_dir)

        if not create_pictures:
            return

        # Create pictures of errors
        # Get noise and boxes estimates from original I stack
        std = find_image_std(original_images["I"], beam_npixels=self._npixels_beam)
        blc, trc = find_bbox(original_images["I"], level=4*std,
                             min_maxintensity_mjyperbeam=6*std,
                             min_area_pix=4*self._npixels_beam, delta=10)

        # I
        error = errors_dict["I"]
        # Use pre-computed and saved mask for I in addition to the mask of the
        # obtained error itself
        error = np.ma.array(error, mask=original_images["I_mask"])
        fig = iplot(original_images["I"], 1000*error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{I}$, mJy/bm", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ipol_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # PPOL (bias-corrected)
        error = errors_dict["PPOL"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        fig = iplot(original_images["I"], 1000*error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{P}$, mJy/bm", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ppol_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # PANG
        error = errors_dict["PANG"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        highest = np.rad2deg(highest)
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=[0, highest],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{\rm EVPA}$, $ ^{\circ}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_pang_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # FPOL
        error = errors_dict["FPOL"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=[0, highest],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{m}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_fpol_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


        # PPOL2
        error = errors_dict["PPOL2"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        # highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], 1000*error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{P2}$, mJy/bm", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ppol2_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # PANG2
        error = errors_dict["PANG2"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        highest = np.rad2deg(highest)
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=[0, highest],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{\rm EVPA2}$, $ ^{\circ}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_pang2_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # FPOL2
        error = errors_dict["FPOL2"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=[0, highest],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{m2}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_fpol2_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # STDPANG2
        error = errors_dict["PANGSTD"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        highest = np.rad2deg(highest)
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=[0, highest],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{\sigma_{\rm EVPA2}}$, $ ^{\circ}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_pangstd_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # STDFPOL2
        error = errors_dict["FPOLSTD"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=[0, highest],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{\sigma_{m2}}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_fpolstd_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # STDPPOL2
        error = errors_dict["PPOLSTD"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        # highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], 1000*error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{\sigma_{P2}}$, mJy/beam", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ppolstd_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # FIXME: This is wrong measure of variability
        # # Original std_{FPOL} "/" error of std_{FPOL}. Values > 3 imply
        # # significant variability (intrinsic, not due to noise)
        # error = errors_dict["FPOLSTD"]
        # error = np.ma.array(error, mask=original_images["P_mask"])
        # # highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        # fig = iplot(original_images["I"], self.original_stack.stack_images["FPOLSTD"].image/error,
        #             x=some_image.x, y=some_image.y,
        #             min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
        #             blc=blc, trc=trc, beam=beam, close=True,
        #             colorbar_label=r"$ \frac{\sigma_{m}}{\sigma_{\sigma_{m}}}$", show_beam=True,
        #             show=True, cmap='nipy_spectral_r', contour_color='black',
        #             plot_colorbar=True, contour_linewidth=0.25)
        # fig.savefig(os.path.join(self.working_dir, "{}_significance_of_fpol_variability.png".format(self.source)),
        #             dpi=300, bbox_inches="tight")
        # plt.close(fig)


        bias = biases_dict["I"]
        bias = np.ma.array(bias, mask=original_images["I_mask"])
        max_bias_value = 1000*np.nanmax(np.abs(bias))
        fig = iplot(original_images["I"], 1000*bias, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=bias.mask, color_clim=[-max_bias_value, max_bias_value],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$b_{I}$, mJy/bm", show_beam=True,
                    show=True, cmap='bwr', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ipol_bias.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        bias = biases_dict["PPOL"]
        bias = np.ma.array(bias, mask=original_images["P_mask"])
        max_bias_value = 1000*np.nanmax(np.abs(bias))
        fig = iplot(original_images["I"], 1000*bias, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=bias.mask, color_clim=[-max_bias_value, max_bias_value],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$b_{P}$, mJy/bm", show_beam=True,
                    show=True, cmap='bwr', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ppol_bias.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        bias = biases_dict["FPOL"]
        bias = np.ma.array(bias, mask=original_images["P_mask"])
        max_bias_value = 0.2
        fig = iplot(original_images["I"], bias, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=bias.mask, color_clim=[-max_bias_value, max_bias_value],
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$b_{m}$", show_beam=True,
                    show=True, cmap='bwr', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_fpol_bias.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        raise Exception("Specify source as positional argument")
    source = sys.argv[1]

    # Number of realizations
    n_mc = 30

    omit_residuals = False
    do_smooth = True

    # Remove created artificial UVFITS files to save disk space?
    remove_artificial_uvfits_files = True

    # Common map parameters
    common_mapsize_clean = choose_mapsize(source)
    beam_size = get_beam_info_by_dec(source)
    common_beam = (beam_size, beam_size, 0)

    # File with source, epoch, core offsets
    source_epoch_core_offset_file = "core_offsets.txt"

    # Directory to save intermediate and final results
    # results_dir = "/mnt/storage/ilya/MOJAVE_pol_stacking/run_1_full_rms"
    results_dir = "/mnt/storage/ilya/MOJAVE_pol_stacking/bias_right"
    working_dir = os.path.join(results_dir, source)
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    # Path to Dan Homan CLEAN-ing script
    path_to_clean_script = "final_clean_rms"

    # Estimate thermal noise from Stokes V or use successive difference approach?
    noise_from_V = False

    # Residual uncertainty in the scale of the gain amplitudes. Set to ``None``
    # if not necessary to model this.
    sigma_scale_amplitude = 0.035

    # Scale to thermal noise estimated from data (1.0 => keep those found in
    # data)
    noise_scale = 1.0

    # Absolute EVPA calibration uncertainty (see MOJAVE VIII paper). Set to
    # ``None`` to skip modelling of this error.
    sigma_evpa_deg = 3.0

    # Model uncertainty of core offsets?
    model_core_shifts_errors = False
    # Error of the core shift (mas)
    shifts_errors_ell_bmaj = 0.05
    shifts_errors_ell_bmin = 0.05/3
    # File with position angles of the inner jet (possibly per-epoch)
    shifts_errors_PA_file = "PA_inner_jet.txt"

    # File with D-terms residuals for VLBA & Eff. Set to ``None`` if no residual
    # D-term modelling is needed.
    VLBA_residual_Dterms_file = "VLBA_EB_Y_residuals_D.json"

    # Number of non-masked epochs in pixel to consider when calculating means.
    n_epochs_not_masked_min = 1
    # Number of non-masked epochs in pixel to consider when calculating errors
    # or stds of PANG, FPOL.
    n_epochs_not_masked_min_std = 5
    # Number of non-masked realizations in pixel to consider when calculating
    # errors for individual epochs maps.
    n_realizations_not_masked_min = 5

    path_to_uvfits_files = "/mnt/jet1/yyk/VLBI/2cmVLBA/data"

    simulation = Simulation(source, n_mc, common_mapsize_clean, common_beam,
                            source_epoch_core_offset_file, working_dir,
                            path_to_clean_script, shifts_errors_ell_bmaj,
                            shifts_errors_ell_bmin, shifts_errors_PA_file,
                            model_core_shifts_errors=model_core_shifts_errors,
                            remove_artificial_uvfits_files=remove_artificial_uvfits_files,
                            create_original_V_stack=False,
                            path_to_uvfits_files=path_to_uvfits_files,
                            omit_residuals=omit_residuals, do_smooth=do_smooth)
    simulation.create_original_stack(n_epochs_not_masked_min, n_epochs_not_masked_min_std)
    simulation.create_artificial_uvdata(sigma_scale_amplitude, noise_scale,
                                        sigma_evpa_deg, VLBA_residual_Dterms_file,
                                        noise_from_V)
    simulation.create_artificial_stacks(n_epochs_not_masked_min, n_epochs_not_masked_min_std)
    simulation.create_errors_images()
    if not omit_residuals:
        simulation.create_individual_epoch_error_images(n_realizations_not_masked_min)

    npz_files = glob.glob(os.path.join(working_dir, "*mc_images*stack.npz"))
    for npz_file in npz_files:
        os.unlink(npz_file)
