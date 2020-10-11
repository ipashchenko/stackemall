import os
import json
import numpy as np
from astropy.io import fits as pf
import astropy.units as u
import matplotlib.pyplot as plt
from create_artificial_data import ArtificialDataCreator
import sys
sys.path.insert(0, '/home/ilya/github/jetpol/scripts')
from jet_image import JetImage, TwinJetImage
sys.path.insert(0, '/home/ilya/github/easy_jet')
from jetmodel import JetModelZoom
sys.path.insert(0, 've/vlbi_errors')
from from_fits import (create_image_from_fits_file,
                       create_clean_image_from_fits_file)
from image import plot as iplot
from spydiff import clean_difmap
from stack_utils import (pol_mask, correct_ppol_bias, find_image_std, find_bbox,
                         stat_of_masked, convolve_with_beam)


working_dir = "/home/ilya/github/stackemall/data/check/"
uvfits_file = "/home/ilya/github/stackemall/data/check/template.uvf"
# HDR for some template image
tempalte_ccfits_file = "/home/ilya/github/stackemall/data/check/template_cc.fits"

hdr = pf.open(tempalte_ccfits_file)[0].header
template_ccimage = create_clean_image_from_fits_file(tempalte_ccfits_file)
VLBA_residual_Dterms_file = "/home/ilya/github/stackemall/VLBA_EB_Y_residuals_D.json"
with open(VLBA_residual_Dterms_file, "r") as fo:
    d_term = json.load(fo)
path_to_clean_script = "/home/ilya/github/stackemall/external_scripts/final_clean_rms"
mapsize_clean = (512, 0.1)
beam = (hdr["BMAJ"]*(u.deg).to(u.mas), hdr["BMAJ"]*(u.deg).to(u.mas), 0)
npixels_beam = np.pi*beam[0]*beam[1]/(mapsize_clean[1]**2*4*np.log(2))
noise_from_V = True
n_mc = 2
sigma_scale_amplitude = 0.035
noise_scale = 1.0
sigma_evpa_deg = 3.0
n_realizations_not_masked_min = 1
# Iterable of the parameters of 2D Gaussian distribution (maj[mas], min[mas],
# bpa[deg]) to use to model error in the derived core shifts. If ``None`` then
# do not model this error. (default: ``None``)
shifts_errors = None


#
# jetpol_run_directory = "/home/ilya/github/jetpol/cmake-build-release"
# stokes = ("I", "Q", "U", "V")
# # FIXME: Substitute with values used in radiative transfer
# jms = [JetImage(z=0.31, n_along=1000, n_across=400, lg_pixel_size_mas_min=-4, lg_pixel_size_mas_max=-2, jet_side=True,
#                 rot=0.0) for _ in stokes]
# cjms = [JetImage(z=0.31, n_along=1000, n_across=400, lg_pixel_size_mas_min=-4, lg_pixel_size_mas_max=-2, jet_side=False,
#                  rot=0.0) for _ in stokes]
# for i, stk in enumerate(stokes):
#     jms[i].load_image_stokes(stk, "{}/jet_image_{}.txt".format(jetpol_run_directory, stk.lower()))
#     cjms[i].load_image_stokes(stk, "{}/cjet_image_{}.txt".format(jetpol_run_directory, stk.lower()))
# # List of models (for J & CJ) for all stokes
# js = [TwinJetImage(jms[i], cjms[i]) for i in range(len(stokes))]
# models_dict = {stk: TwinJetImage(jms[i], cjms[i]) for i, stk in zip((0, 1, 2, 3), stokes)}




lg_pixel_size_mas_min = np.log10(0.01)
lg_pixel_size_mas_max = np.log10(0.01)
n_along = 2000
n_across = 2000
# along_size_mas = np.sum(np.logspace(lg_pixel_size_mas_min,
#                                     lg_pixel_size_mas_max,
#                                     n_along))

jm_i = JetModelZoom(15.4 * u.GHz, 0.0165, n_along, n_across,
                    lg_pixel_size_mas_min,
                    lg_pixel_size_mas_max,
                    central_vfield=True, stokes="I")
jm_i.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                              np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                              np.log(0.000001)]))

jm_q = JetModelZoom(15.4 * u.GHz, 0.0165, n_along, n_across,
                    lg_pixel_size_mas_min,
                    lg_pixel_size_mas_max,
                    central_vfield=True, stokes="Q", ft_scale_factor=0.2)
jm_q.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                              np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                              np.log(0.000001)]))

jm_u = JetModelZoom(15.4 * u.GHz, 0.0165, n_along, n_across,
                    lg_pixel_size_mas_min,
                    lg_pixel_size_mas_max,
                    central_vfield=True, stokes="U", ft_scale_factor=0.0)
jm_u.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                              np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                              np.log(0.000001)]))

models_dict = {stk: jm for jm, stk in zip((jm_i, jm_q, jm_u,), ("I", "Q", "U"))}





lg_pixel_size_mas_min = np.log10(0.1)
lg_pixel_size_mas_max = np.log10(0.1)
n_along = 512
n_across = 512

# Models with CC image resolution
jm_i_im = JetModelZoom(15.4 * u.GHz, 0.0165, n_along, n_across,
                    lg_pixel_size_mas_min,
                    lg_pixel_size_mas_max,
                    central_vfield=True, stokes="I")
jm_i_im.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                              np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                              np.log(0.000001)]))

jm_q_im = JetModelZoom(15.4 * u.GHz, 0.0165, n_along, n_across,
                    lg_pixel_size_mas_min,
                    lg_pixel_size_mas_max,
                    central_vfield=True, stokes="Q", ft_scale_factor=0.2)
jm_q_im.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                              np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                              np.log(0.000001)]))

jm_u_im = JetModelZoom(15.4 * u.GHz, 0.0165, n_along, n_across,
                    lg_pixel_size_mas_min,
                    lg_pixel_size_mas_max,
                    central_vfield=True, stokes="U", ft_scale_factor=0.0)
jm_u_im.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                              np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                              np.log(0.000001)]))

i_model_image = jm_i_im.image()
print("Flux density (Jy/pix) in model image = ", i_model_image.sum())
q_model_image = 0.2*jm_q_im.image()
u_model_image = 0.0*jm_u_im.image()
i_model_image = np.concatenate((np.zeros((512, 256)), i_model_image[:, :256]), axis=1).T[::-1, :]
q_model_image = np.concatenate((np.zeros((512, 256)), q_model_image[:, :256]), axis=1).T[::-1, :]
u_model_image = np.concatenate((np.zeros((512, 256)), u_model_image[:, :256]), axis=1).T[::-1, :]
# ccimage = create_clean_image_from_fits_file(tempalte_ccfits_file)
# beam = ccimage.beam
# FIXME: BPA in rad when obtained from ve!
# Using circular beam from the start
# beam = (beam[0], beam[1], np.rad2deg(beam[2]))
pixsize_mas = 0.1
# TODO: Shoul I just divide convolution with non-normalized beam on beam area instead of normalizing beam?
beam_size = np.pi * beam[0] * beam[1] / (pixsize_mas ** 2 * 4 * np.log(2))
convolved_i_model_image = convolve_with_beam(i_model_image, beam, pixsize_mas, normalize_beam=True)
print("Flux density (Jy/pix) in convolved model image = ", convolved_i_model_image.sum())
convolved_q_model_image = convolve_with_beam(q_model_image, beam, pixsize_mas, normalize_beam=True)
convolved_u_model_image = convolve_with_beam(u_model_image, beam, pixsize_mas, normalize_beam=True)
convolved_p_model_image = np.hypot(convolved_q_model_image, convolved_u_model_image)
convolved_m_model_image = convolved_p_model_image/convolved_i_model_image

creator = ArtificialDataCreator(uvfits_file, path_to_clean_script, mapsize_clean,
                                beam, shift=None, working_dir=working_dir,
                                noise_from_V=noise_from_V, models=models_dict)
creator.mc_create_uvfits(n_mc=n_mc, d_term=d_term, sigma_scale_amplitude=sigma_scale_amplitude,
                         noise_scale=noise_scale, sigma_evpa=sigma_evpa_deg,
                         constant_dterm_amplitude=True,
                         ignore_cross_hands=False)


# bmaj, bmin, bpa = shifts_errors
# bpa = np.deg2rad(bpa)
# print("Adding core shift uncertainty using error ellipse with bmaj, bmin, bpa = {}, {}, {}...".format(bmaj, bmin, bpa))


for i in range(n_mc):
    # delta_x = np.random.normal(0, bmaj, size=1)[0]
    # delta_y = np.random.normal(0, bmin, size=1)[0]
    # # ``bpa`` goes from North clockwise => bpa = 0 means maximal
    # # shifts at DEC direction (``delta_y`` should be maximal)
    # bpa += np.pi / 2
    # delta_x_rot = delta_x * np.cos(bpa) + delta_y * np.sin(bpa)
    # delta_y_rot = -delta_x * np.sin(bpa) + delta_y * np.cos(bpa)
    # shift = (delta_x_rot, delta_y_rot)
    shift = None
    print("Cleaning {} with applied shift = {}...".format("artificial_{}_template.uvf".format(str(i + 1).zfill(3)), shift))

    for stk in ("I", "Q", "U"):
        print("Stokes {}".format(stk))
        clean_difmap(fname="artificial_{}_template.uvf".format(str(i + 1).zfill(3)), outfname="cc_{}_{}.fits".format(stk, str(i + 1).zfill(3)),
                     path=working_dir, stokes=stk, outpath=working_dir,
                     mapsize_clean=mapsize_clean, shift=shift,
                     path_to_script=path_to_clean_script, beam_restore=beam,
                     show_difmap_output=False)


epoch_errors_dict = dict()
ipol_arrays = list()
ppol_arrays = list()
fpol_arrays = list()
pang_arrays = list()
blc = None
trc = None
for i in range(n_mc):
    i_cc_fits_file = os.path.join(working_dir,
                                  "cc_{}_{}.fits".format("I", str(i + 1).zfill(3)))
    q_cc_fits_file = os.path.join(working_dir,
                                  "cc_{}_{}.fits".format("Q", str(i + 1).zfill(3)))
    u_cc_fits_file = os.path.join(working_dir,
                                  "cc_{}_{}.fits".format("U", str(i + 1).zfill(3)))
    i_image = create_image_from_fits_file(i_cc_fits_file)
    q_image = create_image_from_fits_file(q_cc_fits_file)
    u_image = create_image_from_fits_file(u_cc_fits_file)

    ppol_mask_dict, ppol_quantile = pol_mask({"I": i_image.image, "Q": q_image.image, "U": u_image.image},
                                             npixels_beam, n_sigma=3, return_quantile=True)
    ipol_array = np.ma.array(i_image.image, mask=ppol_mask_dict["I"])
    ipol_arrays.append(ipol_array)

    # Mask before correction for bias
    ppol_array = np.ma.array(np.hypot(q_image.image, u_image.image), mask=ppol_mask_dict["P"])
    ppol_array = correct_ppol_bias(i_image.image, ppol_array, q_image.image, u_image.image, npixels_beam)
    ppol_arrays.append(ppol_array)
    fpol_array = ppol_array / i_image.image
    fpol_arrays.append(fpol_array)

    pang_array = 0.5 * np.arctan2(u_image.image, q_image.image)
    pang_array = np.ma.array(pang_array, mask=ppol_mask_dict["P"])
    pang_arrays.append(pang_array)

    if blc is None:
        std = find_image_std(ipol_array, beam_npixels=npixels_beam)
        blc, trc = find_bbox(ipol_array, level=4*std, min_maxintensity_mjyperbeam=6*std,
                             min_area_pix=4*npixels_beam, delta=0)

    ppol_mask_dict, ppol_quantile = pol_mask({"I": ipol_array, "Q": q_image.image, "U": u_image.image},
                                              npixels_beam, n_sigma=4, return_quantile=True)

    fig = iplot(convolved_i_model_image*beam_size, convolved_i_model_image*beam_size, x=template_ccimage.x, y=template_ccimage.y,
                min_abs_level=3 * std, colors_mask=ppol_mask_dict["I"], color_clim=None,
                blc=blc, trc=trc, beam=beam, close=False, colorbar_label=r"$I_{\rm mod,conv}$, Jy/bm", show_beam=True, show=True,
                cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25)
    fig.savefig(os.path.join(working_dir, "{}_ipol_convoled_model.png".format(i)), dpi=600, bbox_inches="tight")
    plt.close()

    fig = iplot(ipol_array, ipol_array, x=template_ccimage.x, y=template_ccimage.y,
                min_abs_level=3 * std, colors_mask=ppol_mask_dict["I"], color_clim=None,
                blc=blc, trc=trc, beam=beam, close=False, colorbar_label=r"$I_{\rm obs}$, Jy/bm", show_beam=True, show=True,
                cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25)
    fig.savefig(os.path.join(working_dir, "{}_ipol.png".format(i)), dpi=600, bbox_inches="tight")
    plt.close()

    idiff = ipol_array-convolved_i_model_image*beam_size
    max_idiff = 1000*np.max(np.abs(idiff))
    fig = iplot(ipol_array, 1000*idiff, x=template_ccimage.x, y=template_ccimage.y,
                min_abs_level=3 * std, colors_mask=ppol_mask_dict["I"], color_clim=[-max_idiff, max_idiff],
                blc=blc, trc=trc, beam=beam, close=False, colorbar_label=r"$I_{\rm obs} - I_{\rm mod,conv}$, mJy/bm", show_beam=True, show=True,
                cmap='bwr', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25)
    fig.savefig(os.path.join(working_dir, "{}_ipol_diff_convoled_model_ipol.png".format(i)), dpi=600, bbox_inches="tight")
    plt.close()

    fig = iplot(ipol_array, convolved_m_model_image, x=template_ccimage.x, y=template_ccimage.y,
                min_abs_level=3 * std, colors_mask=ppol_mask_dict["P"], color_clim=[0, 0.4], blc=blc, trc=trc,
                beam=beam, close=False, colorbar_label=r"$m_{\rm mod,conv}$", show_beam=True, show=True,
                cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25)
    fig.savefig(os.path.join(working_dir, "{}_fpol_convolved_model.png".format(i)), dpi=600, bbox_inches="tight")
    plt.close()

    fig = iplot(ipol_array, fpol_array, x=template_ccimage.x, y=template_ccimage.y,
                min_abs_level=3 * std, colors_mask=ppol_mask_dict["P"], color_clim=[0, 0.7], blc=blc, trc=trc,
                beam=beam, close=False, colorbar_label="m", show_beam=True, show=True,
                cmap='nipy_spectral_r', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25)
    fig.savefig(os.path.join(working_dir, "{}_fpol.png".format(i)), dpi=600, bbox_inches="tight")
    plt.close()

    mdiff = fpol_array - convolved_m_model_image
    max_mdiff = np.max(np.abs(mdiff))
    fig = iplot(ipol_array, mdiff, x=template_ccimage.x, y=template_ccimage.y,
                min_abs_level=3 * std, colors_mask=ppol_mask_dict["P"],
                # color_clim=[-max_mdiff, max_mdiff],
                color_clim=[-0.3, 0.3],
                blc=blc, trc=trc, beam=beam, close=False, colorbar_label=r"$m_{\rm obs} - m_{\rm mod,conv}$", show_beam=True, show=True,
                cmap='bwr', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25)
    fig.savefig(os.path.join(working_dir, "{}_fpol_diff_convolved_model_fpol.png".format(i)), dpi=600, bbox_inches="tight")
    plt.close()

# Create error images for given epoch
std_ipol = stat_of_masked(ipol_arrays, stat="std", n_epochs_not_masked_min=n_realizations_not_masked_min)
std_ppol = stat_of_masked(ppol_arrays, stat="std", n_epochs_not_masked_min=n_realizations_not_masked_min)
std_fpol = stat_of_masked(fpol_arrays, stat="std", n_epochs_not_masked_min=n_realizations_not_masked_min)
std_pang = stat_of_masked(fpol_arrays, stat="scipy_circstd", n_epochs_not_masked_min=n_realizations_not_masked_min)
fpol_mean = stat_of_masked(fpol_arrays, stat="mean", n_epochs_not_masked_min=n_realizations_not_masked_min)
ipol_mean = stat_of_masked(ipol_arrays, stat="mean", n_epochs_not_masked_min=n_realizations_not_masked_min)
fpol_bias = fpol_mean - 0.2
fig = iplot(ipol_array, fpol_bias, x=template_ccimage.x, y=template_ccimage.y,
            min_abs_level=3 * std, colors_mask=ppol_mask_dict["P"], color_clim=[-0.2, 0.2], blc=blc, trc=trc,
            beam=beam, close=False, colorbar_label="FPOL bias", show_beam=True, show=True,
            cmap='bwr', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25)
fig.savefig(os.path.join(working_dir, "{}_fpolbias_4sigma.png".format(i)), dpi=600, bbox_inches="tight")
plt.close()

fpol_bias = fpol_mean - convolved_m_model_image
fig = iplot(ipol_array, fpol_bias, x=template_ccimage.x, y=template_ccimage.y,
            min_abs_level=3 * std, colors_mask=ppol_mask_dict["P"], color_clim=[-0.2, 0.2], blc=blc, trc=trc,
            beam=beam, close=False, colorbar_label="FPOL bias conv.", show_beam=True, show=True,
            cmap='bwr', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25)
fig.savefig(os.path.join(working_dir, "{}_fpolbiasconv_4sigma.png".format(i)), dpi=600, bbox_inches="tight")
plt.close()

std_dict = {"IPOL": std_ipol, "PPOL": std_ppol, "FPOL": std_fpol, "PANG": std_pang}

# Save it to FITS
save_dir = os.path.join(working_dir, "epoch_errors")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for stokes in ("IPOL", "PPOL", "FPOL", "PANG"):
    hdu = pf.PrimaryHDU(data=np.ma.filled(std_dict[stokes], np.nan), header=hdr)
    epoch_errors_dict[stokes] = np.ma.filled(std_dict[stokes], np.nan)
    hdu.writeto(os.path.join(save_dir, "{}_errors.fits".format(stokes)), output_verify='ignore')
    np.savez_compressed(os.path.join(save_dir, "errors.npz"),
                        **epoch_errors_dict)