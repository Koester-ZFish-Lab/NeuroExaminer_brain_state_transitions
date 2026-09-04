from pathlib import Path
import numpy as np
import nrrd
import h5py
import os
import re
from ANTs_helper_functions import compute_volume_registration, apply_volume_registration_to_stack


# changes only here in this section
data_path = '/Volumes/raid_126TB/Mikrofluidik/2025/aligned_ants_selected'  # path to motion corrected files
# resolution of the images
dx = 0.7188675 * 2
dy = 0.7188675 * 2
dz = 9.9992850
reference_brain_path = Path('/Volumes/raid_126TB/H2B_GCaMP6s_new')
reference_brain_name = 'Elavl3-H2BRFP_z_brain.nrrd'
name_text_file = 'ANTs_List.txt'  # for storage of processed fish
Path(f'{data_path}/{name_text_file}').touch()


for folder, subfolders, files in os.walk(f'{data_path}'):
    for file in files:
        if re.search(r'aligned.hdf5', file):

            filepath_aligned = Path(os.sep.join([folder, file]))
            root_path = filepath_aligned.parent
            stack_name = filepath_aligned.stem

            if not os.path.exists(f'{root_path}/zbrain_registered'):
                os.mkdir(f'{root_path}/zbrain_registered')

            with open(f'{data_path}/{name_text_file}', 'r') as ANTs_filelist:
                text = ANTs_filelist.read()

            if re.search(stack_name, text):
                print(f'\n{stack_name} is already registered')
            else:
                print(f'\nCurrently working on {stack_name}')

                f_hdf5 = h5py.File(str(root_path / f'{stack_name}.hdf5'), 'r')
                stack = np.array(f_hdf5['TZYX'], dtype=np.uint8).T
                short = stack[:, :, :, 0:100]  # better for registration when it's not so blurry
                mean_over_time = np.nanmean(short, axis=3)
                mean_over_time = mean_over_time.astype(np.uint8)

                options = {'type': 'uint8',
                           'encoding': 'raw',
                           'endian': 'big',
                           'dimension': 3,
                           'sizes': mean_over_time.shape,
                           'space dimension': 3,
                           'space directions': [[dx, 0, 0], [0, dy, 0], [0, 0, dz]],
                           'space units': ['microns', 'microns', 'microns']}

                nrrd.write(str(root_path / f'{stack_name}_time_averaged.nrrd'), mean_over_time, options)
                f_hdf5.close()

                compute_volume_registration(
                    source_stack_path=root_path / f'{stack_name}_time_averaged.nrrd',
                    target_stack_path=reference_brain_path / f'{reference_brain_name}',
                    registration_files_prefix=root_path / 'zbrain_registered' / f'{stack_name}_time_averaged_to_Elavl3-H2BRFP')

                # Note: change 8bit/16bit depending on template images!
                apply_volume_registration_to_stack(
                    registration_files_prefix_list=[
                        root_path / 'zbrain_registered' / f'{stack_name}_time_averaged_to_Elavl3-H2BRFP'],
                    source_stack_path=root_path / f'{stack_name}_time_averaged.nrrd',
                    target_stack_path=reference_brain_path / f'{reference_brain_name}',
                    output_stack_path=root_path / 'zbrain_registered' / f'{stack_name}_time_averaged_registered.nrrd',
                    bit='uint8')

                with open(f'{data_path}/{name_text_file}', 'a+') as ANTs_filelist:
                    ANTs_filelist.write(f'{stack_name}\n')
