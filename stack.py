import matplotlib
matplotlib.use("Agg")
import os
import sys
import numpy as np
from stack_utils import (pol_mask, stat_of_masked, find_image_std, find_bbox,
                         correct_ppol_bias, image_of_nepochs_not_masked)
sys.path.insert(0, 've/vlbi_errors')
from spydiff import clean_difmap
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from image import Image
from image import plot as iplot
import matplotlib.pyplot as plt
from astropy.io import fits as pf
from astropy.stats import mad_std


class Stack(object):
    def __init__(self, uvfits_files, mapsize_clean, beam, path_to_clean_script,
                 shifts=None, shifts_ell_std=None, working_dir=None,
                 create_stacks=True, n_epochs_not_masked_min=1,
                 n_epochs_not_masked_min_std=5, use_V=False):
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
        :param shifts_ell_std: (optional)
            Iterable of stds of 2D Gaussian distribution (maj[mas], min[mas],
            bpa[deg]) to use to model error in the derived core shifts. If
            ``None`` then do not model this error. (default: ``None``)
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

        """

        # Check input data consistency
        if shifts is not None:
            assert len(shifts) == len(uvfits_files)
        if shifts_ell_std is not None:
            assert len(shifts_ell_std) == len(uvfits_files)
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
        self.shifts_ell_std = shifts_ell_std
        self.mapsize_clean = mapsize_clean
        self.beam = beam
        self._npixels_beam = np.pi*beam[0]*beam[1]/mapsize_clean[1]**2

        self.path_to_clean_script = path_to_clean_script
        self.n_epochs_not_masked_min = n_epochs_not_masked_min
        self.n_epochs_not_masked_min_std = n_epochs_not_masked_min_std

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

        # Containers for full stack images
        self.stack_images = dict()
        for stokes in ("I", "Q", "U", "RPPOL", "PPOL", "PANG", "FPOL",
                       "PPOL2", "FPOL2", "PANG2", "NEPOCHS", "PANGSTD",
                       "FPOLSTD"):
            self.stack_images[stokes] = None
        if use_V:
            self.stack_images["V"] = None

        self._image_ctor_params = dict()
        self._clean_original_data_with_the_same_params()
        if create_stacks:
            self.create_stack_images()

        # Containers for CLEAN images
        self.images = dict()
        for stokes in self.stokes:
            self.images[stokes] = list()
        self.pol_mask = None
        self.i_mask = None

    def _clean_original_data_with_the_same_params(self):
        print("Cleaning uv-data with the same parameters: mapsize = {}, restore beam = {}".format(self.mapsize_clean, self.beam))
        for i, uvfits_file in enumerate(self.uvfits_files):

            if self.shifts is not None:
                shift = self.shifts[i]
                if self.shifts_ell_std is not None:
                    bmaj, bmin, bpa = self.shifts_ell_std[i]
                    bpa = np.deg2rad(bpa)
                    print("Adding core shift uncertainty...")
                    delta_x = np.random.normal(0, bmaj, size=1)[0]
                    delta_y = np.random.normal(0, bmin, size=1)[0]
                    # ``bpa`` goes from North clockwise => bpa = 0 means maximal
                    # shifts at DEC direction (``delta_y`` should be maximal)
                    bpa += np.pi/2
                    delta_x_rot = abs(delta_x*np.cos(bpa) - delta_y*np.sin(bpa))
                    delta_y_rot = abs(delta_x*np.sin(bpa) + delta_y*np.cos(bpa))
                    shift = (shift[0] + delta_x_rot, shift[1] + delta_y_rot)
                print("Cleaning {} with applied shift = {}...".format(uvfits_file, shift))
            elif self.shifts is None and self.shifts_ell_std is not None:
                print("Adding core shift uncertainty...")
                bmaj, bmin, bpa = self.shifts_ell_std[i]
                bpa = np.deg2rad(bpa)
                print("Adding core shift uncertainty...")
                delta_x = np.random.normal(0, bmaj, size=1)[0]
                delta_y = np.random.normal(0, bmin, size=1)[0]
                # ``bpa`` goes from North clockwise => bpa = 0 means maximal
                # shifts at DEC direction (``delta_y`` should be maximal)
                bpa += np.pi / 2
                delta_x_rot = abs(delta_x*np.cos(bpa) - delta_y*np.sin(bpa))
                delta_y_rot = abs(delta_x*np.sin(bpa) + delta_y*np.cos(bpa))
                shift = (delta_x_rot, delta_y_rot)
                print("Cleaning {} with applied shift = {}...".format(uvfits_file, shift))
            else:
                shift = None
                print("Cleaning {} with no shift applied...".format(uvfits_file))

            for stokes in self.stokes:
                print("Stokes {}".format(stokes))
                clean_difmap(fname=uvfits_file, outfname="cc_{}_{}.fits".format(stokes, str(i+1).zfill(3)),
                             stokes=stokes, outpath=self.working_dir, beam_restore=self.beam,
                             mapsize_clean=self.mapsize_clean, shift=shift,
                             path_to_script=self.path_to_clean_script,
                             show_difmap_output=False)

        image = create_clean_image_from_fits_file(os.path.join(self.working_dir, "cc_I_001.fits"))
        self._image_ctor_params["imsize"] = image.imsize
        self._image_ctor_params["pixsize"] = image.pixsize
        self._image_ctor_params["pixref"] = image.pixref
        self._image_ctor_params["freq"] = image.freq
        self._image_ctor_params["pixrefval"] = image.pixrefval

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
        # This will be list of PPOL arrays masked according to it's own epoch
        # polarization mask
        ppol2_arrays = list()

        for i_image, q_image, u_image in zip(i_images, q_images, u_images):
            ppol2_mask_dict, ppol_quantile = pol_mask({"I": i_image.image, "Q": q_image.image, "U": u_image.image},
                                                      self._npixels_beam, n_sigma=3, return_quantile=True)
            # Mask before correction for bias
            ppol2_array = np.ma.array(np.hypot(q_image.image, u_image.image), mask=ppol2_mask_dict["P"])
            ppol2_array = correct_ppol_bias(i_image.image, ppol2_array, q_image.image, u_image.image, self._npixels_beam)
            ppol2_arrays.append(ppol2_array)
        ppol2_image.image = stat_of_masked(ppol2_arrays, stat="mean",
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
        # This will be list of PANG arrays masked according to it's own epoch
        # polarization mask
        pang2_arrays = list()
        for i_image, q_image, u_image in zip(i_images, q_images, u_images):
            pang2_mask_dict, ppol_quantile = pol_mask({"I": i_image.image, "Q": q_image.image, "U": u_image.image},
                                                      self._npixels_beam, n_sigma=3, return_quantile=True)
            pang2_array = 0.5*np.arctan2(u_image.image, q_image.image)
            pang2_array = np.ma.array(pang2_array, mask=pang2_mask_dict["P"])
            pang2_arrays.append(pang2_array)

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
        # This will be list of FPOL arrays masked according to it's own epoch
        # polarization mask
        fpol2_arrays = list()
        for q_image, u_image, i_image in zip(q_images, u_images, i_images):
            ppol_mask_dict, ppol_quantile = pol_mask({"I": i_image.image, "Q": q_image.image, "U": u_image.image},
                                                     self._npixels_beam, n_sigma=3, return_quantile=True)
            ppol_array = np.hypot(q_image.image, u_image.image)
            # First mask
            ppol_array = np.ma.array(ppol_array, mask=ppol_mask_dict["P"])
            # Then correct bias
            ppol_array = correct_ppol_bias(i_image.image, ppol_array, q_image.image, u_image.image,
                                           self._npixels_beam)
            fpol2_arrays.append(ppol_array/i_image.image)

        fpol2_image.image = stat_of_masked(fpol2_arrays, stat="mean",
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

        pang_mask_dict, ppol_quantile = pol_mask({stokes: self.stack_images[stokes].image for stokes in
                                                 ("I", "Q", "U")}, self._npixels_beam, n_sigma=3, return_quantile=True)
        self.stack_images["P_mask"] = pang_mask_dict["P"]
        self.stack_images["I_mask"] = pang_mask_dict["I"]
        self.stack_images["P_quantile"] = ppol_quantile

    def plot_stack_images(self, save_fn, outdir=None):

        if outdir is None:
            outdir = self.working_dir

        ipol_image = self.stack_images["I"]
        ppol_image = self.stack_images["PPOL"]
        fpol_image = self.stack_images["FPOL"]
        pang_image = self.stack_images["PANG"]
        ppol2_image = self.stack_images["PPOL2"]
        nepochs_image = self.stack_images["NEPOCHS"]
        fpol2_image = self.stack_images["FPOL2"]
        pang2_image = self.stack_images["PANG2"]
        std_pang_image = self.stack_images["PANGSTD"]
        std_fpol_image = self.stack_images["FPOLSTD"]
        ppol_quantile = self.stack_images["P_quantile"]
        ipol_mask = self.stack_images["I_mask"]
        ppol_mask = self.stack_images["P_mask"]

        std = find_image_std(ipol_image.image, beam_npixels=self._npixels_beam)
        blc, trc = find_bbox(ipol_image.image, level=4*std, min_maxintensity_mjyperbeam=6*std,
                             min_area_pix=4*self._npixels_beam, delta=10)

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
            elif stokes in ("PPOL2", "FPOL2", "PANG2", "FPOLSTD", "PANGSTD"):
                save_dict.update({stokes: np.ma.filled(self.stack_images[stokes].image, np.nan)})
            elif stokes in ("I_mask", "P_mask"):
                save_dict.update({stokes: self.stack_images[stokes]})
            elif stokes == "P_quantile":
                pass
            else:
                raise Exception("This stokes ({}) is not supposed to be here!".format(stokes))

        np.savez_compressed(os.path.join(outdir, save_fn+"_stack.npz"),
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
            elif stokes in ("PPOL2", "FPOL2", "PANG2", "FPOLSTD", "PANGSTD"):
                hdu = pf.PrimaryHDU(data=np.ma.filled(self.stack_images[stokes].image, np.nan), header=hdr)
            elif stokes in ("I_mask", "P_mask"):
                hdu = pf.PrimaryHDU(data=np.array(self.stack_images[stokes], dtype=int),
                                    header=hdr)
            elif stokes == "P_quantile":
                continue
            else:
                raise Exception("This stokes ({}) is not supposed to be here!".format(stokes))

            hdu.writeto(os.path.join(outdir, "{}_{}.fits".format(save_fn, stokes)))

    def remove_cc_fits(self):
        """
        Remove FITS files with CC for all Stokes.
        """
        for stokes in self.stokes:
            for cc_fits in self.ccfits_files[stokes]:
                os.unlink(cc_fits)

