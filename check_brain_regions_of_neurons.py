import numpy as np
import pandas as pd
import deepdish as dd
from pathlib import Path
import os
import re

# change only data_path and reference_brain
data_path = '/Volumes/raid_126TB/Mikrofluidik/2025/aligned_ants_selected/neurons'
# select either zbrain or mapzebrain:
reference_brain = 'mapzebrain'


if reference_brain == 'zbrain':
    all_masks_indexed = dd.io.load('/Volumes/raid_126TB/H2B_GCaMP6s/zbrain_masks_indexed.hdf5')
    name_dataset = 'ind_mask_volume'
    name_text_file = 'Neurons_Brain_Regions.txt'  # for storage of processed fish
    file_endings = 'coordinates_registered'
    new_file_endings = 'zbrain_regions'

    x_res = 0.798
    y_res = 0.798
    z_res = 2

    x_size = 621
    y_size = 1406
    z_size = 138

elif reference_brain == 'mapzebrain':
    all_masks_indexed = dd.io.load(
        '/Volumes/raid_126TB/H2B_GCaMP6s/mapZebrain_regions_new/mapzebrain_masks_new_indexed.hdf5')
    name_dataset = 'mask_volume_indices'
    name_text_file = 'Neurons_Mapzebrain_Regions.txt'  # for storage of processed fish
    file_endings = 'coordinates_registered_mapzebrain'
    new_file_endings = 'mapzebrain_regions'

    x_res = 0.994
    y_res = 0.994
    z_res = 1

    x_size = 597
    y_size = 974
    z_size = 359


Path(f'{data_path}/{name_text_file}').touch()

all_brain_regions = len(all_masks_indexed.keys())

all_masks = np.zeros((all_brain_regions, z_size, y_size, x_size), dtype=bool)  # shape: regions, z, y, x
for i, key in enumerate(all_masks_indexed.keys()):
    all_masks[i][tuple(all_masks_indexed[key][name_dataset])] = 1


for folder, subfolders, files in os.walk(f'{data_path}'):
    for file in files:
        if re.search(f'{file_endings}.csv', file):

            filepath_registered = Path(os.sep.join([folder, file]))
            root_path = filepath_registered.parent
            stack_name = filepath_registered.stem

            with open(f'{data_path}/{name_text_file}', 'r') as brain_regions:
                text = brain_regions.read()

            if re.search(stack_name, text):
                print(f'\n{stack_name} is already processed')
            else:
                print(f'\nCurrently working on {stack_name}')

                cells_registered = pd.read_csv(filepath_registered)

                # physical coordinates have to be converted back to pixel coordinates
                cells_registered['x'] = (cells_registered['x'] / x_res).round(0)
                cells_registered['y'] = (cells_registered['y'] / y_res).round(0)
                cells_registered['z'] = (cells_registered['z'] / z_res).round(0)
                cells_registered = cells_registered.astype({'z': 'int'})

                cell_locations = np.asarray(cells_registered)

                # cells_zbrain_regions = pd.DataFrame()  # Note: very slow with DataFrames!
                cells_zbrain_regions = []

                for x, y, z, t, label in cell_locations:  # iterate over all cells

                    cell_dict = {'x': x, 'y': y, 'z': z, 't': t, 'label': label}
                    # cell_df = pd.DataFrame(data=[[x, y, z, t, label]], columns=['x', 'y', 'z', 't', 'label'])

                    for i, key in enumerate(all_masks_indexed.keys()):  # iterate over all brain regions
                        try:
                            cell_dict[key] = all_masks[i, int(z), int(y), int(x)]
                            # True or False if the cell is in that region
                            # Performance Warning in Pandas, much faster with Dictionary
                        except IndexError:
                            pass

                    # cells_zbrain_regions = pd.concat((cells_zbrain_regions, cell_df), axis=0, ignore_index=True)
                    cells_zbrain_regions.append(cell_dict)

                cells_zbrain_regions_df = pd.DataFrame(cells_zbrain_regions)

                new_file_name = stack_name.replace(file_endings, new_file_endings)
                cells_zbrain_regions_df.to_csv(f'{root_path}/{new_file_name}.csv', index=False)

                with open(f'{data_path}/{name_text_file}', 'a+') as brain_regions:
                    brain_regions.write(f'{stack_name}\n')
