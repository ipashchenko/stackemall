import os
from stack_utils import (parse_source_list, convert_mojave_epoch,
                         choose_mapsize, get_beam_info_by_dec)
from stack import Stack


source = "0202-172"
results_dir = "/mnt/storage/ilya/MOJAVE_pol_stacking/compare"
working_dir = os.path.join(results_dir, source)
if not os.path.exists(working_dir):
    os.mkdir(working_dir)

path_to_uvfits_files = "/mnt/jet1/yyk/VLBI/2cmVLBA/data"
source_epoch_core_offset_file = "core_offsets.txt"
path_to_clean_script = "final_clean"
common_mapsize_clean = choose_mapsize(source)
beam_size = get_beam_info_by_dec(source)
common_beam = (beam_size, beam_size, 0)

# Number of non-masked epochs in pixel to consider when calculating means.
n_epochs_not_masked_min = 1
# Number of non-masked epochs in pixel to consider when calculating errors
# or stds of PANG, FPOL.
n_epochs_not_masked_min_std = 5

shifts = list()
uvfits_files = list()

df = parse_source_list(source_epoch_core_offset_file, source=source)
df = df.drop_duplicates()
for index, row in df.iterrows():
    epoch = convert_mojave_epoch(row['epoch'])
    shifts.append((row['shift_ra'], row['shift_dec']))
    uvfits_file = "{}/{}/{}/{}.u.{}.uvf".format(path_to_uvfits_files, source, epoch, source, epoch)
    uvfits_files.append(uvfits_file)


stack = Stack(uvfits_files, common_mapsize_clean, common_beam,
              working_dir=working_dir, create_stacks=False,
              shifts=shifts, path_to_clean_script=path_to_clean_script,
              n_epochs_not_masked_min=n_epochs_not_masked_min,
              n_epochs_not_masked_min_std=n_epochs_not_masked_min_std,
              use_V=False)
