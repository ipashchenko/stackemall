import os
import numpy as np
import sys
import astropy.units as u
import glob
sys.path.insert(0, '/home/ilya/github/easy_jet')
from jetmodel import JetModelZoom
sys.path.insert(0, '/home/ilya/github/agn_abc')
from data import Data
from uv_data import downscale_uvdata_by_freq
from stack import Stack
from stack_utils import get_beam_info_by_dec


real_dir = "/home/ilya/github/stackemall/data/bk/real"
real_uvfits_files = glob.glob(os.path.join(real_dir, "*.uvf"))
art_dir = "/home/ilya/github/stackemall/data/bk/art"
lg_pixel_size_mas_min = np.log10(0.0005)
lg_pixel_size_mas_max = np.log10(0.15)
n_along = 1200

for real_uvfits_file in real_uvfits_files:
    print(real_uvfits_file)
    data = Data(real_uvfits_file, use_V_for_noise=True)
    along_size_mas = np.sum(np.logspace(lg_pixel_size_mas_min,
                                        lg_pixel_size_mas_max,
                                        n_along))

    jm_i = JetModelZoom(15.4 * u.GHz, 0.0165, 1200, 100,
                      lg_pixel_size_mas_min,
                      lg_pixel_size_mas_max,
                      central_vfield=True, stokes="I")
    jm_i.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                                np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                                np.log(0.000001)]))

    jm_q = JetModelZoom(15.4 * u.GHz, 0.0165, 1200, 100,
                      lg_pixel_size_mas_min,
                      lg_pixel_size_mas_max,
                      central_vfield=True, stokes="Q", ft_scale_factor=0.2)
    jm_q.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                                np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                                np.log(0.000001)]))

    jm_u = JetModelZoom(15.4 * u.GHz, 0.0165, 1200, 100,
                      lg_pixel_size_mas_min,
                      lg_pixel_size_mas_max,
                      central_vfield=True, stokes="U", ft_scale_factor=0.0)
    jm_u.set_params_vec(np.array([0.0, 0.0, 0.0, 38 * u.deg.to(u.rad), 3.9149 * u.deg.to(u.rad), -2.1052, 8.672,
                                np.log(1.90256), np.log(0.95306), np.log(1.9061), np.log(2 * 1.21 + 1),
                                np.log(0.000001)]))

    data.substitute_with_models([jm_i, jm_q, jm_u])
    # data._uvdata.scale_cross_hands(0.2)
    data.add_original_noise(scale=1)
    downscale_by_freq = downscale_uvdata_by_freq(data._uvdata)
    uvfname = os.path.split(real_uvfits_file)[-1]
    data.save(os.path.join(art_dir, uvfname), downscale_by_freq=downscale_by_freq)


beam_size = get_beam_info_by_dec("0212+735")
common_beam = (beam_size, beam_size, 0)
# Number of non-masked epochs in pixel to consider when calculating means.
n_epochs_not_masked_min = 1
# Number of non-masked epochs in pixel to consider when calculating errors
# or stds of PANG, FPOL.
n_epochs_not_masked_min_std = 5

uvfits_files = glob.glob(os.path.join(art_dir, "*.uvf"))

stack = Stack(uvfits_files, (512, 0.1), common_beam,
              working_dir=art_dir, create_stacks=True,
              shifts=None, path_to_clean_script="external_scripts/final_clean_rms",
              n_epochs_not_masked_min=n_epochs_not_masked_min,
              n_epochs_not_masked_min_std=n_epochs_not_masked_min_std,
              use_V=False)
stack.plot_stack_images("bk_original", outdir=art_dir)