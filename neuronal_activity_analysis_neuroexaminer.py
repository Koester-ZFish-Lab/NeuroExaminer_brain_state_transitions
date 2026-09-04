import numpy as np
import pandas as pd
import deepdish as dd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path

data_path = '/Volumes/raid_126TB/Mikrofluidik/2025/aligned_ants_selected/neurons'
out_path = f'{data_path}/analysis/neuronal_activity_traces_new'
phases = [(185, 370, 'stimulus 1'), (647, 832, 'stimulus 2')]  # frame numbers of medium exchange
# select reference, either zbrain or mapzebrain:
reference_brain = 'zbrain'

if reference_brain == 'zbrain':
    all_masks_indexed = dd.io.load('/Volumes/raid_126TB/H2B_GCaMP6s/zbrain_masks_indexed.hdf5')
    region_names_modifications = pd.read_excel(f'/Volumes/raid_126TB/H2B_GCaMP6s_new/Region_names_zbrain.xlsx',
                                               sheet_name='Sheet1', header=0, usecols='A,B,C,D,E')
elif reference_brain == 'mapzebrain':
    all_masks_indexed = dd.io.load(
        '/Volumes/raid_126TB/H2B_GCaMP6s/mapZebrain_regions_new/mapzebrain_masks_new_indexed.hdf5')
    region_names_modifications = pd.read_excel(f'/Volumes/raid_126TB/H2B_GCaMP6s_new/Region_names_mapzebrain.xlsx',
                                               sheet_name='Sheet1', header=0, usecols='A,B,C,D,E')

brain_region_names = list(all_masks_indexed.keys())
included_regions = region_names_modifications.loc[region_names_modifications['Include'], 'Original Names'].tolist()

# defining the colors to indicate the changes is dF/F
cmap = sns.diverging_palette(255, 14, s=100, l=60, n=21, center='light', as_cmap=True)

if not os.path.exists(out_path):
    os.makedirs(out_path)


def make_plot(dframe, path_to_save, time_steps, phases, width=6, height=5, minimum=-1, maximum=1):

    minutes = np.round((len(time_steps) * 3.25) / 60)  # one frame every 3.25 seconds
    x_ticklabels = np.arange(0, int(minutes) + 1, 10)
    x_ticks = np.arange(0, len(time_steps) + 2, 185)  # (10 min * 60) / 3.25 approx 185: 185 frames equal 10 min
    y_ticks = np.arange(0, dframe.shape[0], 5000)

    fig, ax = plt.subplots(figsize=(width, height))  # adjust height in case few neurons are plotted
    img = ax.imshow(dframe, aspect='auto', cmap=cmap, vmin=minimum, vmax=maximum)

    for a, b, label in phases:
        ax.axvline(x=a, color='black', linewidth=1.5, linestyle=':')
        ax.axvline(x=b, color='black', linewidth=1.5, linestyle=':')
        ax.text((a + b) / 2, 1.01, label,
                ha='center', va='bottom', fontsize=16, color='0.35',
                transform=ax.get_xaxis_transform(), clip_on=False)

    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel('time (min)', fontsize=20, labelpad=10)
    ax.invert_yaxis()
    ax.set_yticks(y_ticks)
    ax.set_ylabel('number of neurons', fontsize=20, labelpad=10)

    cbar = fig.colorbar(img, shrink=0.5, aspect=10, pad=0.03)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks(ticks=[-1, 0, 1])
    cbar.set_ticklabels([r'$-1$', r'$0$', r'$+1$'])
    cbar.set_label(r'$\Delta$F/F', fontsize=20)

    fig.savefig(path_to_save, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


for file in os.listdir(data_path):
    if re.search(r'coordinates.csv', file):

        filepath = Path(os.sep.join([data_path, file]))
        root_path = filepath.parent
        file_name = filepath.stem
        fish_name = file_name.replace('_coordinates', '')
        print(f'\nCurrently working on {fish_name}')

        cells_brain_regions = pd.read_csv(f'{data_path}/{fish_name}_{reference_brain}_regions.csv')

        traces = pd.read_csv(f'{data_path}/{fish_name}_traces.csv')

        time_steps = np.arange(traces.shape[1])

        # dF/F for all neurons
        baseline = traces.iloc[:, 90:180].mean(axis=1)  # 90 frames approximately 5 min; half of habituation
        dFF = (traces.subtract(baseline, axis=0)).div(baseline, axis=0)
        print(f'Number of neurons: {dFF.shape[0]}')

        # Only keep region columns that are in the included list
        cols_included = [col for col in cells_brain_regions.columns[5:] if col in included_regions]
        # Find neurons that are in at least one included region
        mask_included = cells_brain_regions[cols_included].any(axis=1)
        cells_included = cells_brain_regions[mask_included]
        traces_included = traces[traces.index.isin(cells_included.index)]
        dFF_included = dFF[dFF.index.isin(cells_included.index)]

        # additionally, drop neurons with too many zero intensities (not imaged well, out of field of view)
        # few timepoints can be zero because of motion correction (see below), shouldn't be more than 1% (my decision)
        zero_count = (traces_included == 0).sum(axis=1)  # axis=1 checks along columns for number of zeros
        mask_nonzero = zero_count <= (traces_included.shape[1] * 0.01)
        traces_nonzero = traces_included[mask_nonzero]  # keep only rows with less or equal 1% zero values
        cells_nonzero = cells_brain_regions[cells_brain_regions.index.isin(traces_nonzero.index)]
        dFF_nonzero = dFF[dFF.index.isin(traces_nonzero.index)]

        # sometimes, motion correction does not work for individual time points (only 2-3): drop these time points
        zero_timepoints = (traces_nonzero == 0).sum(axis=0)  # axis=0 checks along rows for number of zeros
        mask_timepoints = zero_timepoints <= (traces_nonzero.shape[0] * 0.01)
        traces_timepoints = traces_nonzero.loc[:, mask_timepoints]  # keep only columns with less than 1% zero values
        cells_timepoints = cells_nonzero.loc[:, cells_nonzero.columns.isin(traces_timepoints.columns)]
        dFF_timepoints = dFF_nonzero.loc[:, dFF_nonzero.columns.isin(traces_timepoints.columns)]

        # save nonzero neurons in included regions separately because needed for further analysis
        cells_nonzero.to_csv(
            f'{data_path}/analysis/{fish_name}_cells_included_regions_{reference_brain}.csv')
        traces_nonzero.to_csv(
            f'{data_path}/analysis/{fish_name}_traces_included_regions_{reference_brain}.csv')
        dFF_nonzero.to_csv(
            f'{data_path}/analysis/{fish_name}_dFF_included_regions_{reference_brain}.csv')

        # make plots of selections
        if dFF_nonzero.shape[0] > 0:
            print(f'Number of neurons nonzero: {dFF_nonzero.shape[0]}')
            make_plot(dFF_nonzero,
                      f'{out_path}/{fish_name}_{reference_brain}_dFF_included_regions_nonzero.pdf',
                      time_steps=time_steps, phases=phases)

        if dFF_timepoints.shape[1] < dFF_nonzero.shape[1]:
            print(f'Number of frames nonzero: {dFF_timepoints.shape[1]}')
            make_plot(dFF_timepoints,
                      f'{out_path}/{fish_name}_{reference_brain}_dFF_included_regions_nonzero_timepoints.pdf',
                      time_steps=time_steps, phases=phases)
