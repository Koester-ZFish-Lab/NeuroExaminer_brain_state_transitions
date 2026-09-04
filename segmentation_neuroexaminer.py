import os
import re
from pathlib import Path
import numpy as np
import nrrd
import tifffile
import h5py
import deepdish as dd
from skimage.feature import peak_local_max, match_template
from skimage.draw import disk
from skimage import filters
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_path = '/Volumes/raid_126TB/Mikrofluidik/2025/aligned_ants_selected'
name_text_file = 'Segmentation.txt'  # for storage of processed fish
Path(f'{data_path}/{name_text_file}').touch()
segmentation_results = []  # empty to save thresholds and number of neurons for each fish
exchange_times = [185, 370, 647, 832]  # frame numbers of medium exchange
# resolution of the images
dx = 0.7188675 * 2
dy = 0.7188675 * 2
dz = 9.9992850

if not os.path.exists(f'{data_path}/neurons/analysis'):
    os.makedirs(f'{data_path}/neurons/analysis')

# cell template for template matching
template = tifffile.imread(r'/Volumes/raid_126TB/cell.tif')
template = ndimage.zoom(template, (0.5, 0.5))

# defining the colors to indicate the changes is dF/F
cmap = sns.diverging_palette(255, 14, s=100, l=60, n=21, center='light', as_cmap=True)


def make_plot(dframe, time_steps, exchange_times, path_to_save):

    minutes = np.round((len(time_steps) * 3.25) / 60)  # one frame every 3.25 seconds
    x_ticklabels = np.arange(0, int(minutes) + 1, 10)
    x_ticks = np.arange(0, len(time_steps) + 2, 185)  # (10 min * 60) / 3.25 approx 185: 185 frames equal 10 min

    fig, ax = plt.subplots(figsize=(12, 10))
    img = ax.imshow(dframe, aspect='auto', cmap=cmap, vmin=-1, vmax=1)

    for exchange_time in exchange_times:
        ax.axvline(x=exchange_time, color='white', linewidth=2)

    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel('time (minutes)', fontsize=22, labelpad=10)
    ax.invert_yaxis()
    ax.set_ylabel('number of neurons', fontsize=22, labelpad=10)

    cbar = fig.colorbar(img, shrink=0.5, aspect=15, pad=0.03)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels([r'$-1.0$', r'$-0.5$', r'$0$', r'$+0.5$', r'$+1.0$'])
    cbar.set_label(r'$\Delta$F/F', fontsize=22)

    fig.savefig(path_to_save, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


for folder, subfolders, files in os.walk(data_path):
    for file in files:
        if re.search(r'aligned.hdf5', file):

            filepath_aligned = Path(os.sep.join([folder, file]))
            root_path = filepath_aligned.parent
            file_name = filepath_aligned.stem
            fish_name = file_name.replace('_aligned', '')

            with open(f'{data_path}/{name_text_file}', 'r') as filelist:
                text = filelist.read()

            if re.search(file_name, text):
                print(f'\n{file_name} is already segmented')
            else:
                print(f'\nCurrently working on {file_name}')

                aligned_stack = dd.io.load(f'{root_path}/{file_name}.hdf5')['TZYX']

                try:
                    anatomy, header = nrrd.read(f'{root_path}/{file_name}_time_averaged.nrrd')

                except FileNotFoundError:

                    f_hdf5 = h5py.File(str(root_path / f'{file_name}.hdf5'), 'r')
                    stack = np.array(f_hdf5['TZYX'], dtype=np.uint8).T
                    short = stack[:, :, :, 0:100]  # better for segmentation when it's not so blurry
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

                    nrrd.write(str(root_path / f'{file_name}_time_averaged.nrrd'), mean_over_time, options)
                    f_hdf5.close()

                    anatomy = mean_over_time.copy()

                anatomy = anatomy.T  # to make it TZYX as hdf5 file

                threshold = filters.threshold_mean(anatomy)  # threshold for segmentation
                threshold = np.round(threshold)
                print('Threshold for segmentation: ', threshold)

                # identify ROIs with template matching
                rois = {}
                roi_id = 0
                radius = 2  # pixel number that defines cell size, 2px = around 5.6 um

                roi_map = np.zeros(anatomy.shape + (3,), dtype=np.uint8)
                # the additional 3 in the shape is for making an RGB image to show a pink dot for each cell

                for plane in range(anatomy.shape[0]):

                    m = match_template(anatomy[plane], template, pad_input=True)  # returns an 'image' with matches

                    plm = peak_local_max(m, min_distance=2, threshold_rel=.2)
                    # returns the coordinates (x,y) of local peaks/maxima

                    # show the fish and add pink dots for each found cell later
                    roi_map[plane, :, :] = (anatomy[plane][..., None]).astype(np.uint8)

                    for y, x in plm:

                        if anatomy[plane, y, x] > threshold:  # minimum intensity of a local peak that might be a cell

                            stencil = disk((y, x), radius, shape=anatomy[plane].shape)
                            rois[roi_id] = dict(plane=plane, y=y, x=x, radius=radius,
                                                trace=aligned_stack[:, plane, stencil[0], stencil[1]].mean(axis=1))

                            # mean along axis = 1: mean over the plane; so mean within the cell roi

                            com = disk((y, x), 1.2, shape=anatomy[plane].shape)  # just a dot for each roi

                            roi_map[plane][com] = (100, 0, 100)  # pink dot for each roi on top of original picture

                        roi_id += 1

                tifffile.imwrite(f'{root_path}/neurons/analysis/{fish_name}_roi_map.tif', roi_map)

                rois_df = pd.DataFrame(rois).T
                number_of_neurons = rois_df.shape[0]
                print('Number of neurons: ', number_of_neurons)

                segmentation_results.append({'fish': fish_name,
                                             'threshold': threshold,
                                             'number of neurons': number_of_neurons})

                # save coordinates of neurons and later transform them with ANTs to z-brain coordinates
                neurons = pd.DataFrame({'x': rois_df['x'], 'y': rois_df['y'], 'z': rois_df['plane'],
                                        't': np.zeros(len(rois_df)), 'label': rois_df.index})

                # pixel coordinates and planes have to be converted into the physical space for ANTs:
                neurons['x'] = neurons['x'] * dx
                neurons['y'] = neurons['y'] * dy
                neurons['z'] = neurons['z'] * dz

                neurons.to_csv(f'{root_path}/neurons/{fish_name}_coordinates.csv', index=False)

                # create and save DataFrame with neuronal time traces
                traces = [v['trace'] for v in rois.values()]
                traces = np.stack(traces, axis=0)
                rois_numbers = list(rois.keys())
                time_steps = np.arange(traces.shape[1])  # also used for plotting
                traces_df = pd.DataFrame(traces, columns=time_steps, index=rois_numbers)
                traces_df.to_csv(f'{root_path}/neurons/{fish_name}_traces.csv', index=False)

                # drop cells that are below the threshold used for segmentation (not imaged properly)
                selected_traces = traces_df.copy()
                for cell in selected_traces.index:
                    if (selected_traces.loc[cell].mean() <= threshold) | (selected_traces.loc[cell].median() <= threshold):
                        selected_traces.drop(cell, axis=0, inplace=True)

                # calculate change in fluorescence deltaF/F
                baseline = traces_df.iloc[:, 90:180].mean(axis=1)  # 90 frames approximately 5 min; half of habituation
                dFF = (traces_df.subtract(baseline, axis=0)).div(baseline, axis=0)

                dFF_selected = dFF[dFF.index.isin(selected_traces.index)]

                make_plot(dFF, time_steps, exchange_times, f'{root_path}/neurons/analysis/{fish_name}_dFF.pdf')
                make_plot(dFF_selected, time_steps, exchange_times,
                          f'{root_path}/neurons/analysis/{fish_name}_dFF_selected.pdf')

                with open(f'{data_path}/{name_text_file}', 'a+') as filelist:
                    filelist.write(f'{file_name} - Threshold: {threshold}\n')

segmentation_results_df = pd.DataFrame(segmentation_results)
output_path = f'{root_path}/neurons/segmentation_results.csv'
# Check if the file already exists
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    combined_df = pd.concat([existing_df, segmentation_results_df], ignore_index=True)
    # Optionally drop duplicates by fish name:
    # combined_df = combined_df.drop_duplicates(subset='fish', keep='last')
else:
    combined_df = segmentation_results_df

combined_df.to_csv(output_path, index=False)
