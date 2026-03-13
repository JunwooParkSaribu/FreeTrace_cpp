#!/usr/bin/env python3
"""Step-by-step Python tracking test harness.
Dumps intermediate results at each stage so C++ can be compared function-by-function.
Usage: python3 test_tracking_steps.py <loc_csv> <nb_frames> [--gpu_on] [--jump <val>]
"""
import sys
import os
import json
import math
import numpy as np

# Add FreeTrace to path
sys.path.insert(0, os.path.expanduser('~/FreeTrace'))

from FreeTrace.module.data_load import read_localization
from FreeTrace.module.trajectory_object import TrajectoryObj


def main():
    loc_csv = sys.argv[1]
    nb_frames = int(sys.argv[2])
    gpu_on = '--gpu_on' in sys.argv
    jump_threshold = None
    if '--jump' in sys.argv:
        idx = sys.argv.index('--jump')
        jump_threshold = float(sys.argv[idx + 1])

    out_dir = os.path.dirname(loc_csv) or '.'
    prefix = os.path.join(out_dir, 'step_')

    print(f"=== Step-by-step tracking test ===")
    print(f"  Loc CSV: {loc_csv}")
    print(f"  Nb frames: {nb_frames}")
    print(f"  gpu_on: {gpu_on}")
    print(f"  jump_threshold: {jump_threshold}")
    print()

    # ---- Setup globals (same as Tracking.run) ----
    import FreeTrace.Tracking as T

    T.VERBOSE = False
    T.BATCH = False
    T.TIME_FORECAST = int(max(1, min(5, 2)))  # graph_depth=2
    T.CUTOFF = 2
    T.GPU_AVAIL = gpu_on
    T.REG_LEGNTHS = [3, 5, 8]
    T.LOC_PRECISION_ERR = 1.0
    T.ALPHA_MAX_LENGTH = 10
    T.ALPHA_MODULO = 3
    T.DIMENSION = 2
    T.JUMP_THRESHOLD = jump_threshold
    T.VIDEO_PATH = ''
    T.POST_PROCESSING = False
    T.EMP_BINS = np.linspace(0, 20, 40)
    T.NB_TO_OPTIMUM = int(2 ** T.TIME_FORECAST)

    from FreeTrace.module.auxiliary import initialization
    T.CUDA, T.TF = initialization(T.GPU_AVAIL, T.REG_LEGNTHS, ptype=1, verbose=False, batch=False)
    T.TF = T.GPU_AVAIL  # force prediction with models even on CPU

    T.POLY_FIT_DATA = np.load(f'{T.__file__.split("Tracking.py")[0]}/models/theta_hat.npz')
    T.STD_FIT_DATA = np.load(f'{T.__file__.split("Tracking.py")[0]}/models/std_sets.npz')
    T.QT_FIT_DATA = np.load(f'{T.__file__.split("Tracking.py")[0]}/models/qt_99.npz')

    if T.TF:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from FreeTrace.module.load_models import RegModel
        T.REG_MODEL = RegModel(T.REG_LEGNTHS)

    # ---- Step 1: Read localization ----
    print("Step 1: read_localization")
    # Create a fake images array with the right number of frames
    class FakeImages:
        def __init__(self, n):
            self.shape = (n,)
        def __len__(self):
            return self.shape[0]
    images = FakeImages(nb_frames)
    loc, loc_infos = read_localization(loc_csv, images)

    # Dump: frame -> list of [x,y,z]
    loc_dump = {}
    for t in sorted(loc.keys()):
        arr = loc[t]
        if arr.ndim == 2 and arr.shape[1] == 3:
            loc_dump[int(t)] = arr.tolist()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            loc_dump[int(t)] = [[r[0], r[1], 0.0] for r in arr.tolist()]
        else:
            loc_dump[int(t)] = []  # empty frame
    with open(prefix + '1_localizations.json', 'w') as f:
        json.dump(loc_dump, f)
    n_pts = sum(len(v) for v in loc_dump.values() if v)
    print(f"  {len(loc_dump)} frames, {n_pts} total points")

    # ---- Step 2: count_localizations ----
    print("Step 2: count_localizations")
    t_steps, mean_nb, xyz_min, xyz_max = T.count_localizations(loc)
    step2 = {
        't_steps': t_steps.tolist(),
        'mean_nb_per_time': float(mean_nb),
        'xyz_min': xyz_min.tolist(),
        'xyz_max': xyz_max.tolist()
    }
    with open(prefix + '2_count_localizations.json', 'w') as f:
        json.dump(step2, f)
    print(f"  {len(t_steps)} time steps, mean_nb={mean_nb:.4f}")

    # ---- Step 3: segmentation ----
    print("Step 3: segmentation")
    raw_distributions, jump_distribution = T.segmentation(loc, time_steps=t_steps, lag=T.TIME_FORECAST)
    step3 = {
        'dist_x': raw_distributions[0].tolist(),
        'dist_y': raw_distributions[1].tolist(),
        'dist_z': raw_distributions[2].tolist(),
        'seg_distribution_0': jump_distribution[0] if isinstance(jump_distribution[0], list) else jump_distribution[0].tolist(),
    }
    with open(prefix + '3_segmentation.json', 'w') as f:
        json.dump(step3, f)
    print(f"  dist_x: {len(step3['dist_x'])} values")
    print(f"  dist_y: {len(step3['dist_y'])} values")
    print(f"  seg_distribution[0]: {len(step3['seg_distribution_0'])} values")

    # ---- Step 4: approximation ----
    print("Step 4: approximation")
    max_jumps = T.approximation(raw_distributions, time_forecast=T.TIME_FORECAST, jump_threshold=T.JUMP_THRESHOLD)
    step4 = {str(k): float(v) for k, v in max_jumps.items()}
    with open(prefix + '4_approximation.json', 'w') as f:
        json.dump(step4, f)
    print(f"  Jump thresholds: {max_jumps}")

    # ---- Step 5: build_emp_pdf ----
    print("Step 5: build_emp_pdf")
    T.build_emp_pdf(jump_distribution[0], bins=T.EMP_BINS)
    step5 = {
        'emp_pdf': T.EMP_PDF.tolist(),
        'emp_bins': T.EMP_BINS.tolist(),
    }
    with open(prefix + '5_emp_pdf.json', 'w') as f:
        json.dump(step5, f)
    print(f"  EMP_PDF: {len(T.EMP_PDF)} bins")
    print(f"  EMP_BINS: {len(T.EMP_BINS)} edges")

    # ---- Step 6: trajectory_inference (full tracking) ----
    print("Step 6: trajectory_inference (full tracking)")
    t_avail_steps = []
    for time in np.sort(t_steps):
        if len(loc[time][0]) == 3:
            t_avail_steps.append(time)

    trajectory_list = T.forecast(loc, t_avail_steps, max_jumps, nb_frames, realtime_visualization=False)

    # Dump trajectories
    trajs = []
    for traj in trajectory_list:
        tuples = [(int(t), int(i)) for t, i in traj.trajectory_tuples]
        trajs.append({
            'index': traj.index,
            'tuples': tuples,
            'length': len(tuples)
        })
    with open(prefix + '6_trajectories.json', 'w') as f:
        json.dump(trajs, f)
    print(f"  {len(trajs)} trajectories")

    # Also dump as CSV for comparison
    csv_path = prefix + '6_traces_py.csv'
    with open(csv_path, 'w') as f:
        f.write("traj_idx,frame,x,y\n")
        for traj in trajectory_list:
            for (t, idx) in traj.trajectory_tuples:
                x, y = loc[t][idx][0], loc[t][idx][1]
                f.write(f"{traj.index},{t},{x},{y}\n")
    print(f"  Saved to {csv_path}")

    # ---- Summary ----
    total_pts = sum(len(t['tuples']) for t in trajs)
    print(f"\n=== Summary ===")
    print(f"  Trajectories: {len(trajs)}")
    print(f"  Total points in trajectories: {total_pts}")
    print(f"  Jump threshold: {max_jumps}")


if __name__ == '__main__':
    main()
