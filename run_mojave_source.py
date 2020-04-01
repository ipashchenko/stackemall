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
                         find_image_std, find_bbox, stat_of_masked,
                         choose_range_from_positive_tailed_distribution,
                         get_beam_info)
from create_artificial_data import (ArtificialDataCreator, rename_mc_stack_files)
from stack import Stack
import matplotlib.pyplot as plt
import sys
# FIXME: Substitute with your local path
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
                 path_to_clean_script, remove_artificial_uvfits_files=True,
                 create_original_V_stack=False):
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
        self.remove_artificial_uvfits_files = remove_artificial_uvfits_files
        self.create_original_V_stack = create_original_V_stack

    def create_artificial_uvdata(self, sigma_scale_amplitude, noise_scale,
                                 sigma_evpa_deg, VLBA_residual_Dterms_file):
        with open(VLBA_residual_Dterms_file, "r") as fo:
            d_term = json.load(fo)

        for uvfits_file, shift in zip(self.uvfits_files, self.shifts):
            print("Creating {} artificial data sets from {} with applied shift = {}".format(n_mc, uvfits_file, shift))
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
                      n_epochs_not_masked_min_std=n_epochs_not_masked_min_std,
                      use_V=self.create_original_V_stack)
        stack.save_stack_images("{}_original".format(self.source),
                                outdir=self.working_dir)
        stack.plot_stack_images("{}_original".format(self.source),
                                outdir=self.working_dir)
        stack.save_stack_images_in_fits("{}_original_stack".format(self.source),
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
            # TODO: Do we need FITS files of artificial stacks?
            # stack.save_stack_images_in_fits(str(i+1).zfill(3))
            # Remove CLEAN FITS-files
            stack.remove_cc_fits()

            if self.remove_artificial_uvfits_files:
                # Remove artificial data files
                for uvfits_file in uvfits_files:
                    os.unlink(uvfits_file)

    def create_errors_images(self, create_pictures=True):

        some_image = self.some_image
        beam = self.common_beam
        original_images = np.load(os.path.join(self.working_dir, "{}_original_stack.npz".format(self.source)))

        errors_dict = dict()
        biases_dict = dict()
        for stokes in ("I", "PPOL", "PANG", "FPOL", "PPOL2", "FPOL2", "PANG2", "FPOLSTD", "PANGSTD"):
            mc_images = list()
            for i in range(self.n_mc):
                npz = np.load(os.path.join(self.working_dir, "{}_mc_images_{}_stack.npz".format(self.source, str(i + 1).zfill(3))))
                array = npz[stokes]
                # This stacks are not masked => use trivial mask with zeros
                if stokes in ("I", "PPOL", "PANG", "FPOL"):
                    array = np.ma.array(array, mask=np.zeros(array.shape, dtype=bool))
                elif stokes in ("PPOL2", "FPOL2", "PANG2", "FPOLSTD", "PANGSTD"):
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
            hdu.writeto(os.path.join(self.working_dir, "{}_{}_stack_error.fits".format(self.source, stokes)))

            # Find biases
            if stokes in ("I", "PPOL", "FPOL"):
                mean = stat_of_masked(mc_images, stat="mean",
                                      n_epochs_not_masked_min=n_epochs_not_masked_min_std)
                bias = mean - original_images[stokes]
                biases_dict[stokes] = np.ma.filled(bias, np.nan)
                hdu = pf.PrimaryHDU(data=np.ma.filled(bias, np.nan), header=self.hdr)
                hdu.writeto(os.path.join(self.working_dir, "{}_{}_stack_bias.fits".format(self.source, stokes)))

        np.savez_compressed(os.path.join(self.working_dir, "{}_stack_errors.npz".format(self.source)),
                            **errors_dict)
        np.savez_compressed(os.path.join(self.working_dir, "{}_stack_biases.npz".format(self.source)),
                            **biases_dict)

        # Remove directories with artificial files optionally
        if self.remove_artificial_uvfits_files:
            for i in range(self.n_mc):
                data_dir = os.path.join(self.working_dir, "artificial_{}".format(str(i + 1).zfill(3)))
                os.rmdir(data_dir)

        if not create_pictures:
            return

        # Create pictures of errors
        # Get noise and boxes estimates from original I stack
        std = find_image_std(original_images["I"], beam_npixels=self._npixels_beam)
        blc, trc = find_bbox(original_images["I"], level=4*std,
                             min_maxintensity_mjyperbeam=4*std,
                             min_area_pix=2*self._npixels_beam, delta=10)

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
        # highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=None,
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
        fig = iplot(original_images["I"], error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{P2}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_ppol2_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # PANG2
        error = errors_dict["PANG2"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        # highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=None,
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
        # highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{m2}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_fpol2_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # STDPANG
        error = errors_dict["PANGSTD"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        # highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], np.rad2deg(error), x=some_image.x, y=some_image.y,
                    min_abs_level=3 * std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{\sigma_{\rm EVPA}}$, $ ^{\circ}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_pangstd_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # STDFPOL
        error = errors_dict["FPOLSTD"]
        error = np.ma.array(error, mask=original_images["P_mask"])
        # highest, frac = choose_range_from_positive_tailed_distribution(error.compressed())
        fig = iplot(original_images["I"], error, x=some_image.x, y=some_image.y,
                    min_abs_level=3*std, colors_mask=error.mask, color_clim=None,
                    blc=blc, trc=trc, beam=beam, close=True,
                    colorbar_label=r"$\sigma_{\sigma_{m}}$", show_beam=True,
                    show=True, cmap='nipy_spectral_r', contour_color='black',
                    plot_colorbar=True, contour_linewidth=0.25)
        fig.savefig(os.path.join(self.working_dir, "{}_fpolstd_errors.png".format(self.source)),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

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
        max_bias_value = np.nanmax(np.abs(bias))
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

    # Directory on calculon (jet mirror) to save results
    jet_dir = "/mnt/jet1/ilya/MOJAVE_pol_stacking"
    # Create it if not exists
    if not os.path.exists(jet_dir):
        os.mkdir(jet_dir)

    if len(sys.argv) == 1:
        raise Exception("Specify source as positional argument")
    source = sys.argv[1]
    n_mc = 30
    remove_artificial_uvfits_files = True
    common_mapsize_clean = choose_mapsize(source)
    common_beam = get_beam_info(source)
    # File with source, epoch, core offsets
    source_epoch_core_offset_file = "core_offsets.txt"
    # Directory to save intermediate results
    working_dir = "data/{}".format(source)
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    # Path to Dan Homan CLEAN-ing script
    path_to_clean_script = "final_clean"

    sigma_scale_amplitude = 0.035
    noise_scale = 1.0
    # MOJAVE VIII
    sigma_evpa_deg = 3.0
    # File with D-terms residuals for VLBA & Eff.
    VLBA_residual_Dterms_file = "VLBA_EB_residuals_D.json"

    n_epochs_not_masked_min = 1
    n_epochs_not_masked_min_std = 5

    simulation = Simulation(source, n_mc, common_mapsize_clean, common_beam,
                            source_epoch_core_offset_file, working_dir,
                            path_to_clean_script=path_to_clean_script,
                            remove_artificial_uvfits_files=remove_artificial_uvfits_files,
                            create_original_V_stack=False)
    simulation.create_original_stack(n_epochs_not_masked_min, n_epochs_not_masked_min_std)
    simulation.create_artificial_uvdata(sigma_scale_amplitude, noise_scale,
                                        sigma_evpa_deg, VLBA_residual_Dterms_file)
    simulation.create_artificial_stacks(n_epochs_not_masked_min, n_epochs_not_masked_min_std)
    simulation.create_errors_images()
    move_result_files_to_jet(source, working_dir, jet_dir)

    npz_files = glob.glob(os.path.join(working_dir, "*mc_images*stack.npz"))
    for npz_file in npz_files:
        os.unlink(npz_file)
