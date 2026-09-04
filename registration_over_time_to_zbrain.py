from pathlib import Path
import numpy as np
import nrrd
import h5py
import re
import os
import tempfile
import multiprocessing as mp
from ANTs_helper_functions import apply_volume_registration_to_stack


data_path = '/Volumes/raid_126TB/Mikrofluidik/2025/aligned_ants_selected'
reference_brain_path = Path('/Volumes/raid_126TB/H2B_GCaMP6s_new')
reference_brain_name = 'Elavl3-H2BRFP_z_brain.nrrd'
name_text_file = 'ANTs_over_time.txt'  # for storage of processed fish
Path(f'{data_path}/{name_text_file}').touch()
if not os.path.exists(f'{data_path}/zbrain_registered/over_time'):
    os.makedirs(f'{data_path}/zbrain_registered/over_time')

# get all the filepaths of the files to be registered
filepaths = []
for folder, subfolders, files in os.walk(f'{data_path}'):
    for file in files:
        if re.search(r'aligned.hdf5', file):  # only apply it to the aligned files

            filepath_aligned = Path(os.sep.join([folder, file]))
            filepaths.append(filepath_aligned)

# resolutions of the size-reduced motion aligned lightsheet stack
dx = 0.7188675 * 2
dy = 0.7188675 * 2
dz = 9.9992850


# define a function for opening a file and performing the ANTs registration
def apply_ants_registration(filepath):

    root_path = filepath.parent
    stack_name = filepath.stem
    # pre_stack = stack_name.replace('post', 'pre')

    with open(f'{data_path}/{name_text_file}', 'r') as ants_filelist:
        text = ants_filelist.read()

    if re.search(stack_name, text):
        print(f'\n{stack_name} is already registered\n')
    else:
        print(f'\nCurrently working on {stack_name}\n')

        f_hdf5_motion_aligned = h5py.File(filepath, 'r')  # read the aligned hdf5 file

        t_size = f_hdf5_motion_aligned["TZYX"].shape[0]  # number of time steps in that file

        # Create a new hdf5 file with same number of time steps
        filepath_output = root_path / 'zbrain_registered' / 'over_time' / f'{stack_name}_registered.hdf5'
        f_hdf5_z_brain_registered = h5py.File(filepath_output, 'w')
        dset = f_hdf5_z_brain_registered.create_dataset("TZYX", (t_size, 138, 1406, 621), dtype='uint8',
                                                        compression='gzip')

        # convert each time point in the hdf5 file to XYZ and save to a temporary nrrd file
        with tempfile.TemporaryDirectory() as tmpdir:
            for t in range(t_size):
                # print(f'\ntime step {t}')
                stack = np.array(f_hdf5_motion_aligned["TZYX"][t], dtype=np.uint8).transpose((2, 1, 0))

                options = {'type': 'uint8',
                           'encoding': 'raw',
                           'endian': 'big',
                           'dimension': 3,
                           'sizes': stack.shape,
                           'space dimension': 3,
                           'space directions': [[dx, 0, 0], [0, dy, 0], [0, 0, dz]],
                           'space units': ['microns', 'microns', 'microns']}

                nrrd.write(tmpdir+'/temp_stack.nrrd', stack, options)  # save only as a temporary file

                # Note: change 8bit/16bit depending on template images!
                # use transform files of pre stack for pre and post because they are usually better
                # this is probably due to the clearer image of the average pre stack
                # and it should work because pre and post are both motion aligned to the average pre stack
                apply_volume_registration_to_stack(
                    registration_files_prefix_list=[root_path / 'zbrain_registered'
                                                    / f'{stack_name}_time_averaged_to_Elavl3-H2BRFP'],
                    source_stack_path=Path(tmpdir+'/temp_stack.nrrd'),
                    target_stack_path=reference_brain_path / f'{reference_brain_name}',
                    output_stack_path=Path(tmpdir+'/temp_stack_registered.nrrd'), bit='uint8')

                # Read the registered data
                readdata, header = nrrd.read(tmpdir+'/temp_stack_registered.nrrd')
                readdata = readdata.astype(np.uint8)

                # Place it at the right place into the hdf5 file created above and bring it back to ZYX
                dset[t, :, :, :] = readdata.transpose((2, 1, 0))

        f_hdf5_motion_aligned.close()
        f_hdf5_z_brain_registered.close()

        with open(f'{data_path}/{name_text_file}', 'a+') as ants_filelist:
            ants_filelist.write(f'{stack_name}\n')


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    pool.map(apply_ants_registration, filepaths)
