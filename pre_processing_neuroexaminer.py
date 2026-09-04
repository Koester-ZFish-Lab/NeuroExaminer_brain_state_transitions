import read_lif
import h5py
import os
import re
from pathlib import Path
import numpy as np
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor

data_path = '/Volumes/raid_126TB/Mikrofluidik/2025'
name_text_file = 'Fishlist.txt'  # for storage of processed fish
Path(f'{data_path}/{name_text_file}').touch()  # create empty text file


def calculate_percentile_max(lif_file, start, end, percentile=99.999):
    pixel_values = []
    for t in range(start, end):
        image = lif_file.getFrame(T=t, channel=0, dtype=np.uint16)
        pixel_values.extend(image.flatten())
    maxvalue = np.percentile(pixel_values, percentile)
    return maxvalue


def process_lif_file(start, end, lif_file, maxvalue):

    # parallel processing of images was an idea - does not work
    # def process_image(t):
        # print(t)
        # image = lif_file.getFrame(T=t, channel=0, dtype=np.uint16)
        # image = ndimage.zoom(image, (1, 0.25, 0.25))
        # image[image > max_value] = max_value
        # image = (image / (max_value / 255)).astype(np.uint8)
        # return image

    # with ThreadPoolExecutor() as executor:
        # volumes_over_time = list(executor.map(process_image, range(start, end)))

    volumes_over_time = []

    for t in range(start, end):
        image = lif_file.getFrame(T=t, channel=0, dtype=np.uint16)
        image = ndimage.zoom(image, (1, 0.5, 0.5))
        image[image > maxvalue] = maxvalue
        image = (image / (maxvalue / 255)).astype(np.uint8)
        volumes_over_time.append(image)
        print(t)

    volumes_over_time = np.array(volumes_over_time, dtype=np.uint8)
    volumes_over_time_flipped = volumes_over_time[:, ::-1, :, :]  # to get stack ventral-to-dorsal

    return volumes_over_time_flipped


for folder, subfolders, files in os.walk(f'{data_path}'):
    for file in files:
        if re.search('.lif', file):

            filepath = Path(os.sep.join([folder, file]))
            fish = filepath.stem

            with open(f'{data_path}/{name_text_file}', 'r') as fishlist:
                text = fishlist.read()

            if re.search(fish, text):
                print(f'\n{fish} has already been processed')
            else:
                print(f'\nCurrently working on {fish}')

                reader = read_lif.Reader(filepath)
                series = reader.getSeries()
                chosen = series[0]

                max_value = calculate_percentile_max(chosen, start=0, end=500,
                                                     percentile=99.999)  # around half the time used for calculation
                print(f'Max value: {max_value}')

                # Get the resolution of the stack
                meta_data = chosen.getMetadata()
                dx = meta_data["voxel_size_x"]
                dy = meta_data["voxel_size_y"]
                dz = meta_data["voxel_size_z"]

                # create hdf5 file
                target_filepath = Path(os.sep.join([folder, f'{fish}.hdf5']))
                last_frame = int(chosen.getTotalDuration()) + 1
                images = process_lif_file(0, last_frame, chosen, max_value)
                hdf5 = h5py.File(target_filepath, 'w')
                dset = hdf5.create_dataset('TZYX', images.shape, data=images, dtype='uint8', compression='gzip')
                dset.attrs["element_size_um"] = [dx * 2, dy * 2, dz]  # *2 because of the downsampling
                hdf5.close()

                with open(f'{data_path}/{name_text_file}', 'a+') as fishlist:
                    fishlist.write(f'{fish} - Max value: {max_value}\n')
