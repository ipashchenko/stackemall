import os
import json
import glob
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
import astropy.io.fits as pf
from stack_utils import (parse_source_list, convert_mojave_epoch, choose_mapsize,
                         find_image_std, find_bbox, choose_range_from_positive_tailed_distribution)
from create_artificial_data import (ArtificialDataCreator, rename_mc_stack_files)
from stack import Stack
from stack_utils import (create_mean_image, create_std_image)
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from from_fits import create_clean_image_from_fits_file
from image import plot as iplot


def move_result_files_to_jet(source, calculon_dir, jet_dir):
    """
    Move result files from CALCULON to jet

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
    # Pictures of errors
    for item in ("ipol", "ppol", "fpol", "pang", "ppol2", "fpol2", "pang2", "pangstd", "fpolstd"):
        files.append("{}_{}_errors.png".format(source, item))
    # FITS files with errors of stacks
    for stokes in ("I", "PPOL", "PANG", "FPOL", "PPOL2", "FPOL2", "PANG2", "STDFPOL", "STDPANG"):
        files.append("{}_{}_stack_error.fits".format(source, stokes))
    # npz file with errors of stacks
    files.append("{}_stack_errors.npz".format(source))

    # Original stacks
    files.append("{}_original_images_stack.npz".format(source))
    for stokes in ("I", "PPOL", "PANG", "FPOL", "PPOL2", "FPOL2", "PANG2", "STDFPOL", "STDPANG", "NEPOCHS"):
        files.append("{}_{}_original_images_stack.fits".format(source, stokes))

    for file in files:
        shutil.move(os.path.join(calculon_dir, file), os.path.join(jet_dir, source, file))


class Simulation(object):
    def __init__(self, source, n_mc, common_mapsize_clean, common_beam,
                 source_epoch_core_offset_file, working_dir,
                 path_to_clean_script):
        self.source = source
        self.n_mc = n_mc
        self.common_mapsize_clean = common_mapsize_clean
        self.common_beam = common_beam
        self._npixels_beam = np.pi*common_beam[0]*common_beam[1]/common_mapsize_clean[1]**2
        self.working_dir = working_dir
        self.path_to_clean_script = path_to_clean_script
        self.uvfits_files = list()
        self.shifts = list()
        df = parse_source_list(source_epoch_core_offset_file, source=source)
        df = df.drop_duplicates()
        for index, row in df.iterrows():
            epoch = convert_mojave_epoch(row['epoch'])
            self.shifts.append((row['shift_ra'], row['shift_dec']))
            uvfits_file = "/mnt/jet1/yyk/VLBI/2cmVLBA/data/{}/{}/{}.u.{}.uvf".format(source, epoch, source, epoch)
            self.uvfits_files.append(uvfits_file)
        # Template header used for saving results in FITS format
        self.hdr = None
        # Template image
        self.some_image = None

    def create_artificial_uvdata(self, sigma_scale_amplitude, noise_scale,
                                 sigma_evpa_deg, VLBA_residual_Dterms_file):
        with open(VLBA_residual_Dterms_file, "r") as fo:
            d_term = json.load(fo)

        for uvfits_file, shift in zip(self.uvfits_files, self.shifts):
            print("Creating artificial data for {} & correcting shift = {}".format(uvfits_file, shift))
            creator = ArtificialDataCreator(uvfits_file, self.path_to_clean_script, self.common_mapsize_clean,
                                            self.common_beam, shift=shift, working_dir=self.working_dir)
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
                      n_epochs_not_masked_min_std=n_epochs_not_masked_min_std)
        stack.save_stack_images("{}_original_images_stack.npz".format(self.source),
                                outdir=self.working_dir)
        stack.save_stack_images_in_fits("{}_original_images".format(self.source),
                                        outdir=self.working_dir)
        self.hdr = pf.open(stack.ccfits_files["I"][0])[0].header
        self.some_image = create_clean_image_from_fits_file(stack.ccfits_files["I"][0])
        # Remove CLEAN FITS-files
        stack.remove_cc_fits()

    def create_artificial_stacks(self, n_epochs_not_masked_min, n_epochs_not_masked_min_std):
        # Create images for artificial stacks
        for i in range(self.n_mc):
            data_dir = os.path.join(self.working_dir, "artificial_{}".format(str(i + 1).zfill(3)))
            uvfits_files = sorted(glob.glob(os.path.join(data_dir, "*uvf")))
            # Shifts are already inserted in artificial data
            stack = Stack(uvfits_files, self.common_mapsize_clean, self.common_beam,
                          path_to_clean_script=self.path_to_clean_script,
                          shifts=None, working_dir=data_dir, create_stacks=True,
                          n_epochs_not_masked_min=n_epochs_not_masked_min,
                          n_epochs_not_masked_min_std=n_epochs_not_masked_min_std)
            stack.save_stack_images("{}_mc_images_{}".format(self.source, str(i + 1).zfill(3)),
                                    outdir=self.working_dir)
            stack.save_stack_images_in_fits(str(i+1).zfill(3))
            # Remove CLEAN FITS-files
            stack.remove_cc_fits()
            # Remove artificial data files
            for uvfits_file in uvfits_files:
                os.unlink(uvfits_file)

    def create_erros_images(self, create_pictures=True):

        some_image = self.some_image
        beam = self.common_beam

        errors_dict = dict()
        for stokes in ("I", "PPOL", "PANG", "FPOL", "PPOL2", "FPOL2", "PANG2", "STDFPOL", "STDPANG"):
            mc_images = list()
            for i in range(self.n_mc):
                npz = np.load(os.path.join(self.working_dir, "{}_mc_images_{}_stack.npz".format(self.source, str(i + 1).zfill(3))))
                array = npz[stokes]
                # This stacks are not masked => use trivial mask with zeros
                if stokes in ("I", "PPOL", "PANG", "FPOL"):
                    array = np.ma.array(array, mask=np.zeros(array.shape, dtype=bool))
                elif stokes in ("PPOL2", "FPOL2", "PANG2", "STDFPOL", "STDPANG"):
                    # Masked array with masked values having nans
                    array = np.ma.array(array, mask=np.isnan(array))
                else:
                    raise Exception("{} no allowed, smth is going wrong!".format(stokes))
                mc_images.append(array)
            std = create_std_image(mc_images, n_epochs_not_masked_min=n_epochs_not_masked_min_std)
            hdu = pf.PrimaryHDU(data=np.ma.filled(std, np.nan), header=self.hdr)
            errors_dict[stokes] = np.ma.filled(std, np.nan)
            hdu.writeto(os.path.join(self.working_dir, "{}_{}_stack_error.fits".format(self.source, stokes)))
        np.savez_compressed(os.path.join(self.working_dir, "{}_stack_errors.npz".format(self.source)),
                            **errors_dict)

        if not create_pictures:
            return

        # Create pictures of errors
        original_images = np.load(os.path.join(self.working_dir, "{}_original_images_stack.npz".format(self.source)))
        # Get noise and boxes estimates from original I stack
        std = find_image_std(original_images["I"], beam_npixels=self._npixels_beam)
        blc, trc = find_bbox(original_images["I"], level=4*std,
                             min_maxintensity_mjyperbeam=4*std,
                             min_area_pix=10*self._npixels_beam, delta=10)

        # I
        error = errors_dict["I"]
        # Use pre-computed and saved mask for I in addition to the mask of the
        # obtained error itself
        error = np.ma.array(error, mask=original_images["I_mask"])
        fig = iplot(original_images["I"], 1000*error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=self.common_beam, close=False,
                    colorbar_label=r"$\sigma_{\rm I}$, mJy/bm", show_beam=True,
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
                    blc=blc, trc=trc, beam=beam, close=False,
                    colorbar_label=r"$\sigma_{\rm P}$, mJy/bm", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ppol_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # PANG
        error = errors_dict["PANG"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=[0, np.rad2deg(highest)],
                    blc=blc, trc=trc, beam=beam, close=False,
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
                    blc=blc, trc=trc, beam=beam, close=False,
                    colorbar_label=r"$\sigma_{m}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_fpol_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


        # PPOL2
        error = errors_dict["PPOL2"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=[0, highest],
                    blc=blc, trc=trc, beam=beam, close=False,
                    colorbar_label=r"$\sigma_{P2}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ppol2_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # PANG2
        error = errors_dict["PANG2"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=[0, np.rad2deg(highest)],
                    blc=blc, trc=trc, beam=beam, close=False,
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
                    blc=blc, trc=trc, beam=beam, close=False,
                    colorbar_label=r"$\sigma_{m2}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_fpol2_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # STDPANG
        error = errors_dict["STDPANG"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=[0, np.rad2deg(highest)],
                    blc=blc, trc=trc, beam=beam, close=False,
                    colorbar_label=r"$\sigma_{\sigma_{\rm EVPA}}$, $ ^{\circ}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_stdpang_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # STDFPOL
        error = errors_dict["STDFPOL"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=[0, highest],
                    blc=blc, trc=trc, beam=beam, close=False,
                    colorbar_label=r"$\sigma_{\sigma_{m}}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_stdfpol_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":

    # source = sys.argv[1]
    source = "0006+061"
    n_mc = 3
    common_mapsize_clean = choose_mapsize(source)
    # TODO: Get info on all beams
    common_beam = (0.8, 0.8, 0)
    source_epoch_core_offset_file = "/home/ilya/github/stackemall/core_offsets.txt"
    working_dir = "/home/ilya/github/stackemall/data/"
    path_to_clean_script = "/home/ilya/github/stackemall/final_clean"

    sigma_scale_amplitude = 0.035
    noise_scale = 1.0
    sigma_evpa_deg = 2.0
    VLBA_residual_Dterms_file = "/home/ilya/github/stackemall/VLBA_EB_residuals_D.json"

    n_epochs_not_masked_min = 1
    n_epochs_not_masked_min_std = 5

    jet_dir = "/mnt/jet1/ilya/MOJAVE_pol_stacking"

    simulation = Simulation(source, n_mc, common_mapsize_clean, common_beam,
                            source_epoch_core_offset_file, working_dir,
                            path_to_clean_script=path_to_clean_script)
    simulation.create_original_stack(n_epochs_not_masked_min, n_epochs_not_masked_min_std)
    simulation.create_artificial_uvdata(sigma_scale_amplitude, noise_scale,
                                        sigma_evpa_deg, VLBA_residual_Dterms_file)
    simulation.create_artificial_stacks(n_epochs_not_masked_min, n_epochs_not_masked_min_std)
    move_result_files_to_jet(source, working_dir, jet_dir)
