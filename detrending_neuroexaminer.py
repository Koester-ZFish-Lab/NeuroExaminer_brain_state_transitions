import os
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

data_path = "/Volumes/raid_126TB/Mikrofluidik/2025/aligned_ants_selected/neurons/analysis"
output_path = None  # None: write each *_detrended.csv next to its input file
traces_pattern = "*_traces_included_regions_zbrain.csv"   # which fish/exports to process

anchors = [(90, 180), (560, 647), (1020, 1109)]  # near-baseline frames (resting brain) for the fit
tau_bounds = (300.0, 2000.0)  # frames, physical photobleaching tau
baseline_frames = (90, 180)  # F0 window for dF/F (matches the pipeline)
smooth = 7  # frames; light smoothing of the region mean before fitting
min_neurons = 5  # regions with fewer members fall back to the global envelope
float_csv = "%.6g"  # output CSV number format


def exponential(t, B, tau, C):
    return B * np.exp(-t / tau) + C


def bleaching_baseline(mean_dff, n_frames):
    sm = uniform_filter1d(np.asarray(mean_dff, dtype=float), size=smooth, mode="nearest")
    xs, ys = [], []
    for a, b in anchors:
        b = min(b, n_frames)
        if a < b:
            idx = np.arange(a, b)
            xs.append(idx)
            ys.append(sm[idx])
    xs = np.concatenate(xs).astype(float)
    ys = np.concatenate(ys)
    full = np.arange(n_frames, dtype=float)
    try:
        A0, C0 = ys[0] - ys[-1], ys[-1]
        popt, _ = curve_fit(exponential, xs, ys,
                            p0=[A0 if abs(A0) > 1e-3 else 0.1, 600.0, C0], maxfev=20000,
                            bounds=([-10.0, tau_bounds[0], -10.0], [10.0, tau_bounds[1], 10.0]))
        b = exponential(full, *popt)
    except Exception:
        b = np.polyval(np.polyfit(xs, ys, 1), full)  # robust linear fallback, was never used

    return b.astype(np.float32)  # returns raw fitted exponential, should be ~0 by construction


def region_columns(cells):
    cols = [c for c in cells.columns if cells[c].dtype == bool]
    return cols if cols else list(cells.columns[5:])


def detrend_one(traces_path, cells_path):  # per fish detrending
    traces = pd.read_csv(traces_path, index_col=0)
    cells = pd.read_csv(cells_path, index_col=0).reindex(traces.index)   # align to trace neurons
    F = traces.to_numpy(dtype=np.float32)
    N, T = F.shape

    # per-neuron dF/F with the pipeline's baseline convention (F0 = mean of frames 90:180).
    # The bleaching trend is estimated and removed in dF/F space (see module docstring).
    b0, b1 = baseline_frames
    F0 = F[:, b0:min(b1, T)].mean(axis=1)
    F0safe = np.where(F0 == 0, np.nan, F0)
    dff_raw = (F - F0[:, None]) / F0safe[:, None]

    # per-region bleaching baselines B(t) (dF/F space, ~0 at baseline), accumulated per neuron
    base_sum = np.zeros((N, T), dtype=np.float32)
    count = np.zeros(N, dtype=np.int32)
    regions_used = 0
    for r in region_columns(cells):
        mask = cells[r].fillna(False).to_numpy(dtype=bool)
        if mask.sum() < min_neurons:
            continue
        b = bleaching_baseline(np.nanmean(dff_raw[mask], axis=0), T)
        base_sum[mask] += b
        count[mask] += 1
        regions_used += 1

    # global (all-neuron) baseline for neurons with no fittable region
    b_global = bleaching_baseline(np.nanmean(dff_raw, axis=0), T)
    b_neuron = np.repeat(b_global[None, :], N, axis=0).copy()
    has = count > 0
    b_neuron[has] = base_sum[has] / count[has][:, None]

    dff = dff_raw - b_neuron                            # subtract the bleaching trend (additive)

    dff_out = pd.DataFrame(dff, index=traces.index, columns=traces.columns)
    info = dict(n_neurons=N, n_frames=T, regions_used=regions_used,
                neurons_on_global=int((~has).sum()),
                global_drop=float(-np.mean(b_global[-30:])))
    return dff_out, info


def output_filepath(traces_path):
    out_dir = output_path or os.path.dirname(traces_path)
    base = os.path.basename(traces_path)
    return os.path.join(out_dir, base.replace("traces_included_regions", "dFF_included_regions")
                                     .replace(".csv", "_detrended.csv"))


if __name__ == "__main__":
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    traces_files = sorted(glob.glob(os.path.join(data_path, traces_pattern)))
    print(f"found {len(traces_files)} traces files in {data_path}")
    for tp in traces_files:
        name = os.path.basename(tp)
        cp = tp.replace("traces_included_regions", "cells_included_regions")
        if not os.path.exists(cp):
            continue
        try:
            dff_out, info = detrend_one(tp, cp)
        except Exception as e:
            print(f"Error {name}: {e}")
            continue
        dff_out.to_csv(output_filepath(tp), float_format=float_csv)
        print(info)
