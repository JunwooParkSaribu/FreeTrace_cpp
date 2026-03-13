"""Run Python tracking with ONNX Runtime NN (instead of TF) and C++ tracking with NN,
then compare results at multiple jump thresholds for all 7 samples.""" # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
import os
import sys
import time
import subprocess
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict

os.chdir('/home/junwoo/claude/FreeTrace_cpp')

samples = [
    ("inputs/sample0.tiff", "outputs_verify_s0", "sample0", 100),
    ("inputs/sample1.tiff", "outputs_verify_s1", "sample1", 350),
    ("inputs/sample2.tif",  "outputs_verify_s2", "sample2", 2001),
    ("inputs/sample3.tif",  "outputs_verify_s3", "sample3", 2001),
    ("inputs/sample4.tif",  "outputs_verify_s4", "sample4", 5000),
    ("inputs/sample5.tif",  "outputs_verify_s5", "sample5", 1001),
    ("inputs/sample6.tif",  "outputs_verify_s6", "sample6", 40),
]

# ============================================================
# ONNX Runtime-based NN inference (replacing TF in Python tracking)
# ============================================================

import onnxruntime as ort

def setup_onnx_models(models_dir="models"):
    """Load ONNX models matching C++ exactly."""
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

    reg_model_nums = [3, 5, 8]
    crits = [3, 5, 8, 8192]

    alpha_sessions = {}
    for n in reg_model_nums:
        path = os.path.join(models_dir, f"reg_model_{n}.onnx")
        alpha_sessions[n] = ort.InferenceSession(path, sess_opts, providers=providers)

    k_session = ort.InferenceSession(
        os.path.join(models_dir, "reg_k_model.onnx"), sess_opts, providers=providers)

    return alpha_sessions, k_session, reg_model_nums, crits


def onnx_displacement(xs, ys):
    disps = []
    for i in range(1, len(xs)):
        disps.append(np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2))
    return disps

def onnx_radius(xs, ys):
    rads = [0.]
    for i in range(1, len(xs)):
        rads.append(np.sqrt((xs[i] - xs[0])**2 + (ys[i] - ys[0])**2))
    return rads

def onnx_abs_subtraction(xs):
    result = [0.]
    for i in range(1, len(xs)):
        result.append(abs(xs[i] - xs[i-1]))
    return result

def onnx_cvt_2_signal(x, y):
    rad_list = np.array(onnx_radius(x, y))
    disp_list = onnx_displacement(x, y)
    mean_disp = np.mean(disp_list)
    rad_norm = rad_list / mean_disp / len(x)
    xs_raw = (x - x[0]) / mean_disp / len(x)
    ys_raw = (y - y[0]) / mean_disp / len(y)

    x_norm = x / np.std(x)
    x_sig = np.cumsum(onnx_abs_subtraction(x_norm)) / len(x)
    y_norm = y / np.std(y)
    y_sig = np.cumsum(onnx_abs_subtraction(y_norm)) / len(y)

    return np.transpose(np.stack((x_sig, rad_norm, xs_raw))), \
           np.transpose(np.stack((y_sig, rad_norm, ys_raw)))

def onnx_model_selection(length, crits, reg_model_nums):
    for i in range(len(crits) - 1):
        if crits[i] <= length < crits[i+1]:
            return reg_model_nums[i]
    return reg_model_nums[-2]


def make_onnx_predict_alphas(alpha_sessions, reg_model_nums, crits):
    """Create predict_alphas function using ONNX Runtime."""
    def predict_alphas(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(x)
        if n < reg_model_nums[0]:
            return 1.0

        mn = onnx_model_selection(n, crits, reg_model_nums)
        # Recoupe trajectory
        windows_x, windows_y = [], []
        for i in range(0, n, 1):
            if i + mn <= n:
                windows_x.append(x[i:i+mn])
                windows_y.append(y[i:i+mn])

        input_signals = []
        for wx, wy in zip(windows_x, windows_y):
            s1, s2 = onnx_cvt_2_signal(wx, wy)
            input_signals.append(s1)
            input_signals.append(s2)
        input_signals = np.reshape(input_signals, [-1, mn, 1, 3]).astype(np.float32)

        sess = alpha_sessions[mn]
        in_name = sess.get_inputs()[0].name
        raw_preds = sess.run(None, {in_name: input_signals})[0]

        if raw_preds.shape[0] > 4:
            return float(np.mean(np.quantile(raw_preds, q=[0.25, 0.75], method='normal_unbiased')))
        else:
            return float(np.mean(raw_preds))
    return predict_alphas


def make_onnx_predict_ks(k_session):
    """Create predict_ks function using ONNX Runtime."""
    def predict_ks(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        disps = onnx_displacement(x, y)
        if len(disps) == 0:
            return 0.5
        if len(x) < 10:
            log_d = np.log10(np.mean(disps))
        else:
            log_d = np.log10(np.mean(np.quantile(disps, q=[0.25, 0.75], method='normal_unbiased')))
        if np.isnan(log_d) or np.isinf(log_d):
            return 0.5

        k_input = np.array([[log_d]], dtype=np.float32)
        in_name = k_session.get_inputs()[0].name
        result = k_session.run(None, {in_name: k_input})[0]
        k = float(result.flat[0])
        if np.isnan(k):
            return 1.0
        return k
    return predict_ks


def run_py_tracking_onnx_nn(input_path, out_dir, name, nframes, jump):
    """Run Python tracking with monkey-patched ONNX NN inference."""
    import FreeTrace.Tracking as T

    loc_file = os.path.join(out_dir, "loc_py.csv")
    if not os.path.exists(loc_file):
        return None, 0

    # Setup ONNX models
    alpha_sessions, k_session, reg_model_nums, crits = setup_onnx_models("models")

    # Set TF=True at module level before run_process (it's only set inside run_process normally)
    T.TF = True

    # Monkey-patch the predict functions
    orig_predict_alphas = T.predict_alphas
    orig_predict_ks = T.predict_ks
    onnx_alpha_fn = make_onnx_predict_alphas(alpha_sessions, reg_model_nums, crits)
    onnx_k_fn = make_onnx_predict_ks(k_session)
    T.predict_alphas = onnx_alpha_fn
    T.predict_ks = onnx_k_fn

    # run_process with gpu_on=True will set TF=True and re-assign predict_alphas/ks,
    # so we use a wrapper that keeps re-patching after initialization
    orig_run = T.run_process
    def patched_run_process(**kwargs):
        result = orig_run(**kwargs)
        return result

    # Instead, patch at a lower level: override the globals dict of the module
    # so that even after run_process sets TF and loads REG_MODEL, our functions stay
    import threading
    _lock = threading.Lock()
    _orig_setattr = T.__class__.__setattr__ if hasattr(T.__class__, '__setattr__') else None

    t0 = time.time()
    try:
        # IMPORTANT: run_process spawns a subprocess (multiprocessing.Process),
        # so monkey-patching module globals has no effect there.
        # We must call T.run() directly (same process) for patches to take effect.
        T.run(
            input_video_path=input_path,
            output_path=out_dir,
            graph_depth=2,
            cutoff=2,
            jump_threshold=float(jump),
            gpu_on=True,
            verbose=False,
            HK_output=False,
            read_loc_file=(loc_file, 1.0),
        )
    finally:
        # Restore originals
        T.predict_alphas = orig_predict_alphas
        T.predict_ks = orig_predict_ks
    elapsed = time.time() - t0

    # Find output file - the tracking names it based on input_video_path
    # When input is "inputs/sample0.tiff", output is "{out_dir}/sample0_traces.csv"
    # When read_loc_file is used, also "{out_dir}/loc_py.csv_traces.csv"
    traces_file = None
    for candidate in [
        os.path.join(out_dir, "loc_py.csv_traces.csv"),
        os.path.join(out_dir, f"{name}_traces.csv"),
    ]:
        if os.path.exists(candidate):
            traces_file = candidate
            break

    if traces_file:
        dest = os.path.join(out_dir, f"py_nn_j{int(jump)}_traces.csv")
        shutil.copy2(traces_file, dest)
        os.remove(traces_file)
        # Also remove the other one if it exists
        for f in [os.path.join(out_dir, "loc_py.csv_traces.csv"),
                   os.path.join(out_dir, f"{name}_traces.csv")]:
            if os.path.exists(f) and f != traces_file:
                os.remove(f)
        return dest, elapsed
    return None, elapsed


def run_cpp_tracking_nn(out_dir, name, nframes, jump):
    """Run C++ tracking with --nn flag."""
    loc_file = os.path.join(out_dir, "loc_py.csv")
    if not os.path.exists(loc_file):
        return None, 0

    dest = os.path.join(out_dir, f"cpp_nn_j{int(jump)}_traces.csv")

    t0 = time.time()
    env = os.environ.copy()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    gpu_ort_lib = os.path.join(os.getcwd(), "onnxruntime-linux-x64-gpu-1.24.3", "lib")
    env["LD_LIBRARY_PATH"] = gpu_ort_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run(
        ["./freetrace_nn_gpu", "track", loc_file, out_dir, str(nframes),
         "--depth", "2", "--cutoff", "2", "--jump", str(jump), "--nn"],
        capture_output=True, text=True, timeout=7200, env=env
    )
    elapsed = time.time() - t0

    cpp_traces = os.path.join(out_dir, "loc_py.csv_traces.csv")
    if os.path.exists(cpp_traces):
        shutil.copy2(cpp_traces, dest)
        os.remove(cpp_traces)
        return dest, elapsed

    if result.returncode != 0:
        print(f"  C++ error: {result.stderr[-300:]}")
    return None, elapsed


def compare_traces(py_file, cpp_file):
    """Compare trajectory CSV files, return stats."""
    df1 = pd.read_csv(py_file)
    df2 = pd.read_csv(cpp_file)
    df1.columns = [c.strip().lower() for c in df1.columns]
    df2.columns = [c.strip().lower() for c in df2.columns]

    frame_pts1 = defaultdict(list)
    frame_pts2 = defaultdict(list)
    for _, row in df1.iterrows():
        frame_pts1[int(row['frame'])].append((float(row['x']), float(row['y'])))
    for _, row in df2.iterrows():
        frame_pts2[int(row['frame'])].append((float(row['x']), float(row['y'])))

    all_frames = sorted(set(frame_pts1.keys()) | set(frame_pts2.keys()))
    matched = only1 = only2 = 0
    max_diff = 0.0

    for frame in all_frames:
        p1 = frame_pts1.get(frame, [])
        p2 = list(frame_pts2.get(frame, []))
        used2 = [False] * len(p2)
        for (x1, y1) in p1:
            best_j, best_d = -1, 1e9
            for j, (x2, y2) in enumerate(p2):
                if used2[j]: continue
                d = ((x1-x2)**2 + (y1-y2)**2)**0.5
                if d < best_d: best_d, best_j = d, j
            if best_j >= 0 and best_d < 0.01:
                used2[best_j] = True
                matched += 1
                max_diff = max(max_diff, best_d)
            else:
                only1 += 1
        only2 += sum(1 for u in used2 if not u)

    n1, n2 = len(df1), len(df2)
    t1 = df1['traj_idx'].nunique() if 'traj_idx' in df1.columns else 0
    t2 = df2['traj_idx'].nunique() if 'traj_idx' in df2.columns else 0

    return {
        'py_pts': n1, 'cpp_pts': n2,
        'py_trajs': t1, 'cpp_trajs': t2,
        'matched': matched, 'py_only': only1, 'cpp_only': only2,
        'max_diff': max_diff,
        'match_pct': 100.0 * matched / max(n1, n2, 1)
    }


def main():
    jump_values = [10]
    if len(sys.argv) > 1:
        jump_values = [float(x) for x in sys.argv[1:]]

    for jump in jump_values:
        print(f"\n{'='*90}")
        print(f"  JUMP = {jump}  (NN enabled, ONNX Runtime on both sides)")
        print(f"{'='*90}")

        results = []
        for input_path, out_dir, name, nframes in samples:
            print(f"\n--- {name} ({nframes} frames) ---")

            # Python tracking with ONNX NN
            print(f"  Python (ONNX NN, jump={jump})...", end="", flush=True)
            py_file, py_time = run_py_tracking_onnx_nn(input_path, out_dir, name, nframes, jump)
            if py_file:
                print(f" {py_time:.1f}s")
            else:
                print(f" FAILED ({py_time:.1f}s)")
                continue

            # C++ tracking with NN
            print(f"  C++ (ONNX NN, jump={jump})...", end="", flush=True)
            cpp_file, cpp_time = run_cpp_tracking_nn(out_dir, name, nframes, jump)
            if cpp_file:
                print(f" {cpp_time:.1f}s")
            else:
                print(f" FAILED ({cpp_time:.1f}s)")
                continue

            # Compare
            stats = compare_traces(py_file, cpp_file)
            results.append((name, nframes, stats, py_time, cpp_time))

            pct = stats['match_pct']
            status = "**100%**" if pct == 100.0 else f"{pct:.2f}%"
            print(f"  -> Py {stats['py_pts']} / C++ {stats['cpp_pts']} pts | "
                  f"Matched {stats['matched']} ({status}) | "
                  f"MaxDiff {stats['max_diff']:.6f} px")

        # Summary table
        if results:
            print(f"\n{'='*90}")
            print(f"  SUMMARY — jump={int(jump)}, NN enabled (ONNX Runtime)")
            print(f"{'='*90}")
            print(f"{'Sample':<10} {'Frames':>6} {'Py Trajs':>9} {'Py Pts':>8} {'C++ Pts':>8} "
                  f"{'Matched':>8} {'Match%':>7} {'Py-only':>8} {'C++-only':>9} {'MaxDiff':>10}")
            print("-" * 105)
            total_py = total_cpp = total_matched = total_py_only = total_cpp_only = 0
            for name, nframes, s, pt, ct in results:
                pct = s['match_pct']
                marker = " ***" if pct < 100.0 else ""
                print(f"{name:<10} {nframes:>6} {s['py_trajs']:>9} {s['py_pts']:>8} {s['cpp_pts']:>8} "
                      f"{s['matched']:>8} {pct:>6.2f}% {s['py_only']:>8} {s['cpp_only']:>9} "
                      f"{s['max_diff']:>10.6f}{marker}")
                total_py += s['py_pts']
                total_cpp += s['cpp_pts']
                total_matched += s['matched']
                total_py_only += s['py_only']
                total_cpp_only += s['cpp_only']
            print("-" * 105)
            total_pct = 100.0 * total_matched / max(total_py, total_cpp, 1)
            print(f"{'TOTAL':<10} {'':>6} {'':>9} {total_py:>8} {total_cpp:>8} "
                  f"{total_matched:>8} {total_pct:>6.2f}% {total_py_only:>8} {total_cpp_only:>9}") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13


if __name__ == "__main__":
    main()
