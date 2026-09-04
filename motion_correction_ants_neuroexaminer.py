from pathlib import Path
import os
import re
import datetime
import math
import numpy as np
import nrrd
import h5py
import nibabel as nib
from ANTs_helper_functions import run_linux_command


data_path = '/Volumes/raid_126TB/Mikrofluidik/2025'
ants_bin_path = '/Users/koester_lab/install/bin'
name_text_file = 'Motion_Correction.txt'  # for storage of processed fish
Path(f'{data_path}/{name_text_file}').touch()  # create empty text file
aligned_folder = 'aligned_ants'
acceptable_length_per_stack = 1109  # for more time points file will be split to reduce memory pressure


def split_stack(hdf5_file, number_of_parts, length_of_parts):

    print(f'\nSplitting stack into {number_of_parts} parts')
    print(datetime.datetime.now())

    rootpath = hdf5_file.parent
    stackname = hdf5_file.stem

    hdf5file = h5py.File(hdf5_file, 'r')['TZYX']

    start = 0
    for n in range(number_of_parts):
        end = start + length_of_parts
        part = hdf5file[start:end, :, :, :]

        target_filepath = rootpath / f'{stackname}_part{n+1}.hdf5'
        with h5py.File(target_filepath, 'w') as hdf5_short:
            hdf5_short.create_dataset('TZYX', part.shape, data=part, dtype='uint8', compression='gzip')

        start += length_of_parts


def combine_parts(aligned_files_prefix, number_of_parts):
    all_parts = []

    for n in range(number_of_parts):
        part, header = nrrd.read(str(aligned_files_prefix) + f'_part{n+1}_aligned.nrrd')
        all_parts.append(part)

    complete = np.concatenate(all_parts, axis=3)
    complete = complete.T  # for hdf5 t,z,y,x, instead of x,y,z,t as in nrrd

    target_filepath = str(aligned_files_prefix) + '_aligned.hdf5'
    with h5py.File(target_filepath, 'w') as hdf5:
        hdf5.create_dataset('TZYX', complete.shape, data=complete, dtype='uint8', compression='gzip')


def convert_hdf5_to_nii(hdf5_file):

    print('\nConverting HDF5 file to NIfTI file')
    print(datetime.datetime.now())

    rootpath = hdf5_file.parent
    stackname = hdf5_file.stem

    f_hdf5 = h5py.File(hdf5_file, 'r')
    stack = np.array(f_hdf5["TZYX"], dtype=np.uint8).T
    short = np.array(f_hdf5["TZYX"][0:10, :, :, :], dtype=np.uint8).T  # to make a reference

    img = nib.Nifti1Image(stack, np.eye(4))
    # we need to provide an image coordinate transform (affine) and choose the identity matrix to not change anything
    img_short = nib.Nifti1Image(short, np.eye(4))

    nib.save(img, str(rootpath / f'{stackname}.nii.gz'))
    nib.save(img_short, str(rootpath / f'{stackname}_short.nii.gz'))

    f_hdf5.close()


def convert_nii_to_nrrd(nii_file):

    print('\nConverting NIfTI file to nrrd')
    print(datetime.datetime.now())

    rootpath = nii_file.parent
    stackname = Path(nii_file.stem)  # wegen .nii.gz zweimal stem erforderlich
    stackname = stackname.stem

    nii_image = nib.load(nii_file)
    data = nii_image.get_fdata()
    data = data.astype(np.uint8)

    dx = 0.7188675 * 2
    dy = 0.7188675 * 2
    dz = 9.9992850

    options = {'type': 'uint8',
               'encoding': 'raw',
               'endian': 'big',
               'dimension': 4,
               'sizes': data.shape,
               'space dimension': 3,
               'space directions': [[dx, 0, 0], [0, dy, 0], [0, 0, dz]],
               'space units': ['microns', 'microns', 'microns']}

    nrrd.write(str(rootpath / f"{stackname}.nrrd"), data, options)


def convert_nii_to_hdf5(nii_file):

    print('\nConverting NIfTI file to hdf5')
    print(datetime.datetime.now())

    rootpath = nii_file.parent
    stackname = Path(nii_file.stem)  # wegen .nii.gz zweimal stem erforderlich
    stackname = stackname.stem

    nii_image = nib.load(nii_file)
    data = nii_image.get_fdata()
    data = data.T  # for hdf5 t,z,y,x, instead of x,y,z,t as in nrrd and nii
    data = data.astype(np.uint8)

    target_filepath = (str(rootpath / f'{stackname}.hdf5'))
    with h5py.File(target_filepath, 'w') as hdf5:
        hdf5.create_dataset('TZYX', data.shape, data=data, dtype='uint8', compression='gzip')


def compute_mean_stack(source_stack_path, output_stack_path):

    print('\nComputing mean of the stack')
    print(datetime.datetime.now())

    commands_average = [f'{ants_bin_path}/antsMotionCorr',
                        '-d', '3',
                        '-a', f'{source_stack_path}',  # average the time series
                        '-o', f'{output_stack_path}']

    run_linux_command(commands_average)


def compute_motion_correction_3d(moving_image, fixed_image, registration_prefix):

    print('\nNow computing ANTs Motion Correction in 3D')
    print(datetime.datetime.now())

    commands_list = [f'{ants_bin_path}/antsMotionCorr',
                     '-d', '3',   # dimensionality: 3D
                     '-n', '10',  # number of images used to construct the template image
                     '-m', f'GC[{fixed_image},{moving_image},1,32,Regular,0.25]',
                     # image metric (GC global correlation, CC neighbourhood cross correlation, MI Mutual Information)
                     '-t', 'Affine[0.1]',  # [gradient step]
                     '-i', '200x100',   # number of iterations
                     '-u', '1',       # use a fixed reference image
                     '-e', '1',       # use scales estimator to control optimization
                     '-s', '1x0',     # smoothing
                     '-f', '2x1',     # shrink factor
                     '-w', '1',   # write out the displacement field (captures affine induced motion at each voxel)
                     '-v', '1',   # verbose output
                     '-o', f'[{registration_prefix},{registration_prefix}.nii.gz,{registration_prefix}_avg.nii.gz]'
                     # output [outputTransformPrefix, outputWarpedImage, outputAverageImage]
                     ]
    # parameters:
    # https://github.com/ANTsX/ANTs/blob/master/Scripts/antsMotionCorrExample
    # https://stnava.github.io/fMRIANTs/
    # https://sourceforge.net/p/advants/discussion/840261/thread/72c36866/

    run_linux_command(commands_list)


def compute_motion_correction_2d(moving_image, fixed_image, registration_prefix):

    print('\nNow computing ANTs Motion Correction in 2D')
    print(datetime.datetime.now())

    commands_list = [f'{ants_bin_path}/antsMotionCorr',
                     '-d', '2',   # dimensionality: 2D
                     '-n', '1',  # number of images used to construct the template image
                     '-m', f'GC[{fixed_image},{moving_image},1,32,Regular,0.25]',
                     # image metric (GC global correlation, CC neighbourhood cross correlation, MI Mutual Information)
                     # fixed image, moving image, metric weight, number of bins, sampling strategy, sampling percentage
                     '-t', 'Affine[0.1]',  # [gradient step]
                     '-i', '33x20',   # number of iterations
                     '-u', '1',       # use a fixed reference image
                     '-e', '1',       # use scales estimator to control optimization
                     '-s', '1x0',     # smoothing
                     '-f', '2x1',     # shrink factor
                     '-w', '1',   # write out the displacement field (captures affine induced motion at each voxel)
                     '-v', '1',   # verbose output
                     '-o', f'[{registration_prefix},{registration_prefix}.nii.gz,{registration_prefix}_avg.nii.gz]'
                     # output [outputTransformPrefix, outputWarpedImage, outputAverageImage]
                     ]

    run_linux_command(commands_list)


if __name__ == '__main__':

    for file in os.listdir(data_path):
        if re.search(r'.hdf5', file):

            filepath = Path(os.sep.join([data_path, file]))
            root_path = filepath.parent
            stack_name = filepath.stem

            if not os.path.exists(f'{root_path}/{aligned_folder}'):
                os.mkdir(f'{root_path}/{aligned_folder}')

            with open(f'{data_path}/{name_text_file}', 'r') as motion_corrected_filelist:
                text = motion_corrected_filelist.read()

            if re.search(stack_name, text):
                print(f'\n{stack_name} is already registered')
            else:
                print(f'\nCurrently working on {stack_name}')

                convert_hdf5_to_nii(filepath)  # automatically saves first 10 frames individually to make a reference

                compute_mean_stack(source_stack_path=root_path / f'{stack_name}_short.nii.gz',
                                   output_stack_path=root_path / f'{stack_name}_short_avg.nii.gz')

                hdf5 = h5py.File(filepath, 'r')['TZYX']

                if hdf5.shape[0] > acceptable_length_per_stack:
                    parts = math.ceil(hdf5.shape[0] / acceptable_length_per_stack)
                    length_parts = math.ceil(hdf5.shape[0] / parts)
                    split_stack(filepath, parts, length_parts)

                    for i in range(parts):
                        convert_hdf5_to_nii(root_path / f'{stack_name}_part{i + 1}.hdf5')

                        compute_motion_correction_3d(
                            moving_image=root_path / f'{stack_name}_part{i + 1}.nii.gz',
                            fixed_image=root_path / f'{stack_name}_short_avg.nii.gz',
                            registration_prefix=root_path / f'{aligned_folder}' / f'{stack_name}_part{i + 1}_aligned')

                        convert_nii_to_nrrd(root_path / f'{aligned_folder}' / f'{stack_name}_part{i + 1}_aligned.nii.gz')
                    combine_parts(root_path / f'{aligned_folder}' / f'{stack_name}', parts)

                else:
                    compute_motion_correction_3d(
                        moving_image=root_path / f'{stack_name}.nii.gz',
                        fixed_image=root_path / f'{stack_name}_short_avg.nii.gz',
                        registration_prefix=root_path / f'{aligned_folder}' / f'{stack_name}_aligned')

                    convert_nii_to_hdf5(root_path / f'{aligned_folder}' / f'{stack_name}_aligned.nii.gz')

                with open(f'{data_path}/{name_text_file}', 'a+') as motion_corrected_filelist:
                    motion_corrected_filelist.write(f'{stack_name}\n')
