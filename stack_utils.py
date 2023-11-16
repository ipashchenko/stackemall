import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import pandas as pd
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from scipy.stats import percentileofscore, scoreatpercentile
from scipy.stats import circstd, circmean
import astropy.io.fits as pf
from astropy.wcs import WCS
from astropy.stats import mad_std
from astropy.convolution import convolve, Gaussian2DKernel
from astropy import units as u
from pycircstat import mean, std
import sys
sys.path.insert(0, 've/vlbi_errors')
from from_fits import create_clean_image_from_fits_file



# def parse_std_file(fn, bad_sources_fn=None):
#     bad_sources = list()
#     with open(fn, "r") as fo:
#         lines = fo.readlines()
#     for line in lines:
#         if line.startswith("#"):
#             continue
#         splitted = line.split()
#         source = splitted[0]
#         i_rms_ratio = float(splitted[2])
#         p_rms_ratio = float(splitted[4])
#         if i_rms_ratio > 1 or p_rms_ratio > 1:
#             bad_sources.append(source)
#     if bad_sources_fn is not None:
#         with open(bad_sources_fn, "w") as fo:
#             for source in bad_sources:
#                 fo.write(source + "\n")
#     return bad_sources



def get_mask_for_ccstack(iccfiles, cc_conv_cutoff_mjy=0.0075, kern_width=10,
                         imsize=512):
    """
    Get mask containing CC of stack.
    :param iccfiles:
        Iterable of CC-files.
    :param cc_conv_cutoff_mjy:
    :param kern_width:
    :return:
        Masked numpy 2D array.
    """
    # iccfiles = glob.glob(os.path.join(cc_dir, "cc_I_*.fits"))
    iccimages = [create_clean_image_from_fits_file(ccfile) for ccfile in iccfiles]
    icconly = np.mean([ccimage.cc for ccimage in iccimages], axis=0)
    cconly_conv = convolve(1000*icconly, Gaussian2DKernel(x_stddev=kern_width))
    signal = cconly_conv > cc_conv_cutoff_mjy
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=cconly_conv)
    max_prop = sorted(props, key=lambda x: x.area, reverse=True)[0]
    mask = np.zeros((imsize, imsize), dtype=bool)
    bbox = max_prop.bbox
    mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = max_prop.filled_image
    return mask


def filter_CC(ccfits, mask, outname=None, plotsave_fn=None):
    """
    Filter CC from FITS file using specified mask. Only CC that fall into the
    mask are kept. Write resulting set of CC to new (or old) CC FITS.

    :param ccfits:
        CC FITS file.
    :param mask:
        Mask where the region of source flux being True.
    :param outname: (optional)
        CC FITS name to write the resulting CCs. If ``None`` than rewrite.
        (default: ``None``)
    :param plotsave_fn: (optional)
        File to save plot. If ``None`` than do not plot. (default: ``None``)
    """
    hdus = pf.open(ccfits)
    hdus.verify("silentfix")
    data = hdus[1].data
    data_ = data.copy()
    deg2mas = u.deg.to(u.mas)

    header = pf.getheader(ccfits)
    imsize = header["NAXIS1"]
    wcs = WCS(header)
    # Ignore FREQ, STOKES - only RA, DEC matters here
    wcs = wcs.celestial

    # Make offset coordinates
    wcs.wcs.crval = 0, 0
    wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    wcs.wcs.cunit = 'deg', 'deg'

    xs = list()
    ys = list()
    xs_del = list()
    ys_del = list()
    fs_del = list()
    for flux, x_orig, y_orig in zip(data['FLUX'], data['DELTAX'], data['DELTAY']):
        x, y = wcs.world_to_array_index(x_orig*u.deg, y_orig*u.deg)

        if x >= imsize:
            x = imsize - 1
        if y >= imsize:
            y = imsize - 1
        if mask[x, y]:
            # Keep this component
            xs.append(x)
            ys.append(y)
        else:
            # Remove row from rec_array
            xs_del.append(x_orig)
            ys_del.append(y_orig)
            fs_del.append(flux)

    for (x, y, f) in zip(xs_del, ys_del, fs_del):
        local_mask = ~np.logical_and(np.logical_and(data_["DELTAX"] == x, data_["DELTAY"] == y),
                                     data_["FLUX"] == f)
        data_ = data_.compress(local_mask, axis=0)
    print("Deleted {} components".format(len(xs_del)))

    if plotsave_fn is not None:
        a = data_['DELTAX']*deg2mas
        b = data_['DELTAY']*deg2mas
        a_all = data['DELTAX']*deg2mas
        b_all = data['DELTAY']*deg2mas

        fig, axes = plt.subplots(1, 1)
        # im = axes.scatter(a, b, c=np.log10(1000*data_["FLUX"]), vmin=0, s=1, cmap="jet")
        axes.scatter(a_all, b_all, color="gray", alpha=0.25, s=2)
        axes.scatter(a, b, color="red", alpha=0.5, s=1)
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.00)
        # cb = fig.colorbar(im, cax=cax)
        # cb.set_label("CC Flux, Jy")
        axes.invert_xaxis()
        axes.set_aspect("equal")
        axes.set_xlabel("RA, mas")
        axes.set_ylabel("DEC, mas")
        plt.savefig(plotsave_fn, bbox_inches="tight", dpi=300)
        plt.close()

    hdus[1].data = data_
    hdus[1].header["NAXIS2"] = len(data_)
    if outname is None:
        outname = ccfits
    hdus.writeto(outname, overwrite=True)


def parse_bad_epochs_file(fn):
    bad_epochs_dict = dict()
    with open(fn, "r") as fo:
        lines = fo.readlines()
        for line in lines:
            splitted = line.split()
            source = splitted[0]
            epoch = splitted[2]
            if source in bad_epochs_dict:
                bad_epochs_dict[source].append(epoch)
            else:
                bad_epochs_dict[source] = list()
                bad_epochs_dict[source].append(epoch)
    return bad_epochs_dict


# TODO: If per-epoch core shift errors are needed then change
#  implementation of this function
def get_inner_jet_PA(source, epoch, inner_jet_PA_file):
    df = pd.read_csv(inner_jet_PA_file, delim_whitespace=True, names=["source", "pa_comp", "sigma_pa_comp", "pa_ridge"])
    return df.query("source == @source")["pa_ridge"].values[0]


def get_sources_with_failed_jobs(logfile):
    failed_sources = list()
    df = pd.read_csv(logfile, sep="\t")
    df = df.sort_values(["Seq", "Starttime"])
    gb = df.groupby("Command")
    for state, frame in gb:
        source = state.split(" ")[-1]
        last_exit_val = frame.tail(1)["Exitval"].values[0]
        if last_exit_val == 1:
            failed_sources.append(source)
    return failed_sources


def remove_dirs_with_failed_jobs(logfile, working_dir="/mnt/storage/ilya/MOJAVE_pol_stacking", test=True):
    failed_sources = get_sources_with_failed_jobs(logfile)
    for source in failed_sources:
        dir_to_rm = os.path.join(working_dir, source)
        print("Removig directory ", dir_to_rm)
        try:
            if test:
                print("TEST: Removing ", dir_to_rm)
            else:
                shutil.rmtree(dir_to_rm)
        except FileNotFoundError:
            print("No directory {} found! Skipping.".format(dir_to_rm))


def get_beam_info_by_dec(source):
    df = pd.read_csv("source_dec.txt", delim_whitespace=True,
                     names=["source", "dec"], dtype={"source": str, "dec": float})
    dec = df.query("source == @source")["dec"].values[0]
    bmaj = 1.28295977 - 8.95027412e-03*dec - 7.91363153e-05*dec**2 + 1.24419018e-06*dec**3
    bmin = 0.52192393 + 1.00730852e-03*dec + 8.88395448e-06*dec**2 - 5.57102780e-08*dec**3
    return np.round(0.5*(bmaj+bmin), 2)


def get_beam_info(source):
    beam_file = "/mnt/jet1/pushkarev/Stacking_P/5_epochs_sample_deep_clean/{}/beam_median".format(source)
    bmaj, bmin, bpa = np.loadtxt(beam_file)
    assert bmaj == bmin
    assert bpa == 0
    return bmaj, bmin, bpa


# Note that this is real map sizes. Difmap expects twice than this!
def choose_mapsize(source):
    if source in ("0219+428",
                  "0429+415",
                  "0333+321",
                  "0519+011",
                  "0805-077",
                  "0820+560",
                  "0859-140",
                  "1228+126",
                  "1458+718",
                  "1514+004",
                  "1514-241",
                  "1641+399",
                  "1730-130",
                  "1807+698",
                  "1914-194",
                  "1928+738"):
        mapsize_clean = 1024
    elif source == "1345+125":
        mapsize_clean = 2048
    else:
        mapsize_clean = 512
    return mapsize_clean, 0.1


def convert_mojave_epoch(epoch):
    year = epoch.split('-')[0]
    month = epoch.split('-')[1]
    day = epoch.split('-')[2]
    return "{}_{}_{}".format(year, month, day)


def convert_mojave_epoch_inverse(epoch):
    year = epoch.split('_')[0]
    month = epoch.split('_')[1]
    day = epoch.split('_')[2]
    return "{}-{}-{}".format(year, month, day)


def get_epochs(target_source, source_list):
    epochs = list()
    df = pd.read_csv(source_list, sep="   |  | ", names=["source", "epoch", "shift_ra", "shift_dec"],
                     engine="python")
    for index, row in df.iterrows():
        source = row['source']
        if source != target_source:
            continue
        epoch = convert_mojave_epoch(row['epoch'])
        epochs.append(epoch)
    return epochs


def parse_source_list(source_list, source=None):
    df = pd.read_csv(source_list, sep="   |  | ", names=["source", "epoch", "shift_ra", "shift_dec"],
                     engine="python")
    if source is not None:
        df = df.query("source == @source")
    return df.drop_duplicates()


def check_bbox(blc, trc, image_size):
    """
    :note:
        This can make quadratic image rectangular.
    """
    # If some bottom corner coordinate become negative
    blc = list(blc)
    trc = list(trc)
    if blc[0] < 0:
        blc[0] = 0
    if blc[1] < 0:
        blc[1] = 0
    # If some top corner coordinate become large than image size
    if trc[0] > image_size:
        delta = abs(trc[0]-image_size)
        blc[0] -= delta
        # Check if shift have not made it negative
        if blc[0] < 0 and trc[0] > image_size:
            blc[0] = 0
        trc[0] -= delta
    if trc[1] > image_size:
        delta = abs(trc[1]-image_size)
        blc[1] -= delta
        # Check if shift have not made it negative
        if blc[1] < 0 and trc[1] > image_size:
            blc[1] = 0
        trc[1] -= delta
    return tuple(blc), tuple(trc)


def find_bbox(array, level, min_maxintensity_mjyperbeam, min_area_pix,
              delta=0.):
    """
    Find bounding box for part of image containing source.

    :param array:
        Numpy 2D array with image.
    :param level:
        Level at which threshold image in image units.
    :param min_maxintensity_mjyperbeam:
        Minimum of the maximum intensity in the region to include.
    :param min_area_pix:
        Minimum area for region to include.
    :param delta: (optional)
        Extra space to add symmetrically [pixels]. (default: ``0``)
    :return:
        Tuples of BLC & TRC.

    :note:
        This is BLC, TRC for numpy array (i.e. transposed source map as it
        conventionally seen on VLBI maps).
    """
    signal = array > level
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=array)

    signal_props = list()
    for prop in props:
        if prop.max_intensity > min_maxintensity_mjyperbeam/1000 and prop.area > min_area_pix:
            signal_props.append(prop)

    # Sometimes no regions are found. In that case return full image
    if not signal_props:
        return (0, 0,), (array.shape[1], array.shape[1],)

    blcs = list()
    trcs = list()

    for prop in signal_props:
        bbox = prop.bbox
        blc = (int(bbox[1]), int(bbox[0]))
        trc = (int(bbox[3]), int(bbox[2]))
        blcs.append(blc)
        trcs.append(trc)

    min_blc_0 = min([blc[0] for blc in blcs])
    min_blc_1 = min([blc[1] for blc in blcs])
    max_trc_0 = max([trc[0] for trc in trcs])
    max_trc_1 = max([trc[1] for trc in trcs])
    blc_rec = (min_blc_0-delta, min_blc_1-delta,)
    trc_rec = (max_trc_0+delta, max_trc_1+delta,)

    blc_rec_ = blc_rec
    trc_rec_ = trc_rec
    blc_rec_, trc_rec_ = check_bbox(blc_rec_, trc_rec_, array.shape[0])

    # Enlarge 10% each side
    delta_ra = abs(trc_rec[0]-blc_rec[0])
    delta_dec = abs(trc_rec[1]-blc_rec[1])
    blc_rec = (blc_rec[0] - int(0.1*delta_ra), blc_rec[1] - int(0.1*delta_dec))
    trc_rec = (trc_rec[0] + int(0.1*delta_ra), trc_rec[1] + int(0.1*delta_dec))

    blc_rec, trc_rec = check_bbox(blc_rec, trc_rec, array.shape[0])

    return blc_rec, trc_rec


def find_image_std(image_array, beam_npixels):
    # Robustly estimate image pixels std
    std = mad_std(image_array)

    # Find preliminary bounding box
    blc, trc = find_bbox(image_array, level=4*std,
                         min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=2*beam_npixels,
                         delta=0)

    # Now mask out source emission using found bounding box and estimate std
    # more accurately
    mask = np.zeros(image_array.shape)
    mask[blc[1]: trc[1], blc[0]: trc[0]] = 1
    outside_icn = np.ma.array(image_array, mask=mask)
    return mad_std(outside_icn)


def find_iqu_image_std(i_image_array, q_image_array, u_image_array, beam_npixels):
    # Robustly estimate image pixels std
    std = mad_std(i_image_array)

    # Find preliminary bounding box
    blc, trc = find_bbox(i_image_array, level=4*std,
                         min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=2*beam_npixels,
                         delta=0)

    # Now mask out source emission using found bounding box and estimate std
    # more accurately
    mask = np.zeros(i_image_array.shape)
    mask[blc[1]: trc[1], blc[0]: trc[0]] = 1
    outside_icn = np.ma.array(i_image_array, mask=mask)
    outside_qcn = np.ma.array(q_image_array, mask=mask)
    outside_ucn = np.ma.array(u_image_array, mask=mask)
    return {"I": mad_std(outside_icn), "Q": mad_std(outside_qcn), "U": mad_std(outside_ucn)}


def correct_ppol_bias(ipol_array, ppol_array, q_array, u_array, beam_npixels):
    std_dict = find_iqu_image_std(ipol_array, q_array, u_array, beam_npixels)
    rms = 0.5*(std_dict["Q"] + std_dict["U"])
    snr = ppol_array / rms
    factor = 1-1/snr**2
    factor[factor < 0] = 0
    return ppol_array*np.sqrt(factor)


# TODO: Add restriction on spatial closeness of the outliers to include them in the range
def choose_range_from_positive_tailed_distribution(data, min_fraction=95):
    """
    Suitable for PANG and FPOL maps.

    :param data:
        Array of values in masked region. Only small fraction (in positive side
        tail) is supposed to be noise.
    :param min_fraction: (optional)
        If no gaps in data distribution than choose this fraction range (in
        percents). (default: ``95``)
    :return:
    """
    mstd = mad_std(np.ma.array(data, mask=np.isnan(data)))
    min_fraction_range = scoreatpercentile(data, min_fraction)
    hp_indexes = np.argsort(data)[::-1][np.argsort(np.diff(np.sort(data)[::-1]))]
    diff = None
    for ind in hp_indexes:
        hp = data[ind]
        try:
            hp_low = np.sort(data)[hp - np.sort(data) > 0][-1]
        except IndexError:
            return min_fraction_range, 95
        diff = hp - hp_low
        frac = percentileofscore(data, hp_low)
        if diff < mstd/2 and frac < 95:
            break
    # Somehow diff haven't been computed
    if diff is None:
        return min_fraction_range, 95
    if diff > mstd/2:
        return min_fraction_range, 95
    else:
        return hp_low, frac


def pol_mask(stokes_image_dict, beam_pixels, n_sigma=2., return_quantile=False,
             residual_images=None, use_both_I_and_P_for_P_mask=True):
    """
    Find mask using stokes 'I' map and 'PPOL' map using specified number of
    sigma.

    :param stokes_image_dict:
        Dictionary with keys - stokes, values - arrays with images.
    :param beam_pixels:
        Number of pixels in beam.
    :param n_sigma: (optional)
        Number of sigma to consider for stokes 'I' and 'PPOL'. 1, 2 or 3.
        (default: ``2``)
    :param residual_images: (optional)
        Dictionary with residual images for each Stokes. If not ``None`` than
        use this images to estimate std. (default: ``None``)
    :return:
        Dictionary with Boolean array of masks and P quantile (optionally).
    """
    quantile_dict = {1: 0.6827, 2: 0.9545, 3: 0.9973, 4: 0.99994}
    if residual_images is None:
        rms_dict = find_iqu_image_std(*[stokes_image_dict[stokes] for stokes in ('I', 'Q', 'U')],  beam_pixels)
    else :
        rms_dict = {stokes: mad_std(residual_images[stokes]) for stokes in ("I", "Q", "U")}

    qu_rms = np.mean([rms_dict[stoke] for stoke in ('Q', 'U')])
    ppol_quantile = qu_rms * np.sqrt(-np.log((1. - quantile_dict[n_sigma]) ** 2.))
    i_cs_mask = stokes_image_dict['I'] < n_sigma * rms_dict['I']
    ppol_cs_image = np.hypot(stokes_image_dict['Q'], stokes_image_dict['U'])
    ppol_cs_mask = ppol_cs_image < ppol_quantile

    if use_both_I_and_P_for_P_mask:
        mask_dict = {"I": i_cs_mask, "P": np.logical_or(i_cs_mask, ppol_cs_mask)}
    else:
        mask_dict = {"I": i_cs_mask, "P": ppol_cs_mask}

    if not return_quantile:
        return mask_dict
    else:
        return mask_dict, ppol_quantile


def gen_mask(marrays, n_epochs_not_masked_min=1):
    """
    Using several masked arrays find mask based on minimal number of unmasked
    pixels in stack of pixels.

    :param marrays:
        Iterable of masked arrays.
    :param n_epochs_not_masked_min: (default: ``1``)
        Integer. Minimal number of unmasked pixels in stack of pixels for not
        including this stack pixel in resulting mask.
    :return:
        Masked numpy array.
    """
    # Find union of masks
    masks = [~marray.mask for marray in marrays]
    # Now each mask has 1 where pixel is not masked
    masks = [np.array(mask, dtype=int) for mask in masks]
    # In each pixel number of epochs when pixel is not masked
    gen_mask_sums = np.sum(masks, axis=0)
    gen_mask = gen_mask_sums >= n_epochs_not_masked_min
    # Masked if pixel is masked in each epoch
    gen_mask = ~gen_mask
    return gen_mask


def image_of_nepochs_not_masked(marrays):
    """
    Create image of number of individual arrays pixels in stack pixel that are
    not masked.

    :param marrays:
        Iterable of masked arrays.
    :return:
        Numpy array of ints.
    """
    masks = [~marray.mask for marray in marrays]
    # Now each mask has 1 where pixel is not masked
    masks = [np.array(mask, dtype=int) for mask in masks]
    # In each pixel number of epochs when pixel is not masked
    gen_mask_sums = np.sum(masks, axis=0)
    mask = gen_mask_sums == 0
    return np.ma.array(gen_mask_sums, mask=mask)


def stat_of_masked(marrays, stat="mean", n_epochs_not_masked_min=1):
    """
    Calculate specified statistic over several images masked with different
    individual masks.

    :param marrays:
        Iterable of individual masked arrays.
    :param stat:
        Statistics to compute.
    :param n_epochs_not_masked_min: (default: ``1``)
        Minimum number of unmasked pixels in stack of pixels to consider while
        calculating statistic. For scatter-concerned statistic (e.g. ``std``)
        number > 1 is required.
    :return:
        Masked array.
    """
    statistics = ("mean", "median", "std", "circmean", "circstd", "scipy_circmean", "scipy_circstd")
    if stat not in statistics:
        raise Exception("stat must be among {}!".format(statistics))
    # General mask based on number of unmasked pixels in stack of pixels.
    general_mask = gen_mask(marrays, n_epochs_not_masked_min)
    # Union general mask and individual array masks
    new_marrays = list()
    for marray in marrays:
        old_mask = np.ma.getmask(marray)
        array = np.ma.getdata(marray)
        new_mask = np.logical_or(old_mask, general_mask)
        new_marrays.append(np.ma.array(array, mask=new_mask))
    marrays_stacked = np.ma.dstack(new_marrays)

    if stat == "mean":
        result = np.ma.mean(marrays_stacked, axis=2)

    elif stat == "median":
        result = np.ma.median(marrays_stacked, axis=2)

    elif stat == "std":
        result = np.ma.std(marrays_stacked, axis=2)

    elif stat == "circmean":
        result = mean(2*marrays_stacked, axis=2)/2

    elif stat == "circstd":
        result = np.sqrt(std(2*marrays_stacked, axis=2)/2)

    elif stat == "scipy_circstd":
        result = np.nan*np.ones(new_marrays[0].shape, dtype=float)
        for (x, y), mvalue in np.ndenumerate(general_mask):
            if mvalue:
                continue
            else:
                data = marrays_stacked[x, y, :]
                result[x, y] = circstd(data.compressed(), low=-np.pi/2, high=np.pi/2)
        result = np.ma.array(result, mask=general_mask)

    elif stat == "scipy_circmean":
        marrays_stacked = np.ma.dstack(new_marrays)
        result = np.nan*np.ones(new_marrays[0].shape, dtype=float)
        for (x, y), mvalue in np.ndenumerate(general_mask):
            if mvalue:
                continue
            else:
                data = marrays_stacked[x, y, :]
                result[x, y] = circmean(data.compressed(), low=-np.pi/2, high=np.pi/2)
        result = np.ma.array(result, mask=general_mask)

    else:
        raise Exception

    return result
