"""Run Python and C++ tracking with NN models, compare results at multiple jump thresholds.""" # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
import os
import sys
import time
import subprocess
import shutil
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

os.chdir('/home/junwoo/claude/FreeTrace_cpp')
sys.path.insert(0, '/home/junwoo/claude/FreeTrace')

samples = [
    ("inputs/sample0.tiff", "outputs_verify_s0", "sample0", 100),
    ("inputs/sample1.tiff", "outputs_verify_s1", "sample1", 350),
    ("inputs/sample2.tif",  "outputs_verify_s2", "sample2", 2001),
    ("inputs/sample3.tif",  "outputs_verify_s3", "sample3", 2001),
    ("inputs/sample4.tif",  "outputs_verify_s4", "sample4", 5000),
    ("inputs/sample5.tif",  "outputs_verify_s5", "sample5", 1001),
    ("inputs/sample6.tif",  "outputs_verify_s6", "sample6", 40),
]


def run_py_tracking_nn(input_path, out_dir, name, nframes, jump):
    """Run Python FreeTrace tracking with NN (gpu_on=True)."""
    from FreeTrace import Tracking

    loc_file = os.path.join(out_dir, "loc_py.csv")
    if not os.path.exists(loc_file):
        return None, 0

    dest_csv = os.path.join(out_dir, f"{name}_py_traces.csv")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    dest_png = os.path.join(out_dir, f"{name}_py_traces.png")

    t0 = time.time()
    ok = Tracking.run_process(
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
    elapsed = time.time() - t0

    # Find output CSV file
    traces_file = None
    for candidate in [
        os.path.join(out_dir, "loc_py_traces.csv"),
        os.path.join(out_dir, f"{name}_traces.csv"),
    ]:
        if os.path.exists(candidate):
            traces_file = candidate
            break

    if traces_file is None:
        for f in os.listdir(out_dir):
            if f.endswith("_traces.csv") and f != os.path.basename(dest_csv) and "_cpp_" not in f:
                traces_file = os.path.join(out_dir, f)
                break

    if traces_file:
        shutil.copy2(traces_file, dest_csv)
        if traces_file != dest_csv:
            os.remove(traces_file)

        # Find and rename matching PNG
        traces_png = traces_file.replace("_traces.csv", "_traces.png")
        if not os.path.exists(traces_png):
            traces_png = traces_file.replace(".csv", ".png")
        if os.path.exists(traces_png):
            shutil.copy2(traces_png, dest_png)
            if traces_png != dest_png:
                os.remove(traces_png)

        return dest_csv, elapsed
    return None, elapsed


def run_cpp_tracking_nn(input_path, out_dir, name, nframes, jump):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    """Run C++ FreeTrace tracking with --nn flag."""
    loc_file = os.path.join(out_dir, "loc_py.csv")
    if not os.path.exists(loc_file):
        return None, 0

    dest_csv = os.path.join(out_dir, f"{name}_cpp_traces.csv")
    dest_png = os.path.join(out_dir, f"{name}_cpp_traces.png")

    t0 = time.time()
    env = os.environ.copy()
    gpu_ort_lib = os.path.join(os.getcwd(), "onnxruntime-linux-x64-gpu-1.24.3", "lib")
    env["LD_LIBRARY_PATH"] = gpu_ort_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run(
        ["./freetrace_nn_gpu", "track", loc_file, out_dir, str(nframes),
         "--depth", "2", "--cutoff", "2", "--jump", str(jump), "--nn",
         "--tiff", input_path],
        capture_output=True, text=True, timeout=600, env=env
    )
    elapsed = time.time() - t0

    # Find output CSV (C++ may produce either name)
    cpp_traces = os.path.join(out_dir, "loc_py.csv_traces.csv")
    if not os.path.exists(cpp_traces):
        cpp_traces = os.path.join(out_dir, "loc_py_traces.csv")
    if os.path.exists(cpp_traces):
        shutil.copy2(cpp_traces, dest_csv)
        # Find and rename matching PNG
        cpp_png = cpp_traces.replace("_traces.csv", "_traces.png").replace(".csv_traces.csv", ".csv_traces.png")
        if not os.path.exists(cpp_png):
            cpp_png = cpp_traces.replace(".csv", "").replace("_traces", "_traces.png")
        # Try both possible PNG names
        for png_candidate in [
            cpp_traces.replace(".csv", ".png"),
            cpp_traces.replace("_traces.csv", "_traces.png"),
        ]:
            if os.path.exists(png_candidate):
                shutil.copy2(png_candidate, dest_png)
                if png_candidate != dest_png:
                    os.remove(png_candidate)
                break
        if cpp_traces != dest_csv:
            os.remove(cpp_traces)
        return dest_csv, elapsed

    # Also check for other trace files
    for f in os.listdir(out_dir):
        if f.endswith("_traces.csv") and "_py_" not in f and "_cpp_" not in f and f != os.path.basename(dest_csv):
            src = os.path.join(out_dir, f)
            shutil.copy2(src, dest_csv)
            src_png = src.replace("_traces.csv", "_traces.png")
            if os.path.exists(src_png):
                shutil.copy2(src_png, dest_png)
                os.remove(src_png)
            os.remove(src)
            return dest_csv, elapsed

    if result.returncode != 0:
        print(f"  C++ stderr: {result.stderr[-200:]}")
    return None, elapsed


def compare_traces_strict(py_file, cpp_file):
    """Compare two trajectory CSV files with point-level AND trajectory-level matching."""
    df1 = pd.read_csv(py_file)
    df2 = pd.read_csv(cpp_file)
    df1.columns = [c.strip().lower() for c in df1.columns]
    df2.columns = [c.strip().lower() for c in df2.columns]

    frame_pts1 = defaultdict(list)
    frame_pts2 = defaultdict(list)
    for _, row in df1.iterrows():
        frame_pts1[int(row['frame'])].append((float(row['x']), float(row['y']), int(row['traj_idx'])))
    for _, row in df2.iterrows():
        frame_pts2[int(row['frame'])].append((float(row['x']), float(row['y']), int(row['traj_idx'])))

    all_frames = sorted(set(frame_pts1.keys()) | set(frame_pts2.keys()))

    pt_matched = 0
    pt_only1 = 0
    pt_only2 = 0
    max_diff = 0.0
    traj_link_1to2 = defaultdict(list)
    traj_link_2to1 = defaultdict(list)

    for frame in all_frames:
        p1 = frame_pts1.get(frame, [])
        p2 = frame_pts2.get(frame, [])
        used2 = [False] * len(p2)
        for (x1, y1, t1) in p1:
            best_j = -1
            best_d = 1e9
            for j, (x2, y2, t2) in enumerate(p2):
                if used2[j]:
                    continue
                d = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d < 0.01:
                used2[best_j] = True
                pt_matched += 1
                max_diff = max(max_diff, best_d)
                t2 = p2[best_j][2]
                traj_link_1to2[t1].append(t2)
                traj_link_2to1[t2].append(t1)
            else:
                pt_only1 += 1
        pt_only2 += sum(1 for u in used2 if not u)

    n1 = len(df1)
    n2 = len(df2)
    trajs1 = sorted(df1['traj_idx'].unique())
    trajs2 = sorted(df2['traj_idx'].unique())
    traj1_npts = df1.groupby('traj_idx').size().to_dict()
    traj2_npts = df2.groupby('traj_idx').size().to_dict()

    # Trajectory-level: bijective 1:1 matching
    traj1_to_traj2_sets = {t1: set(traj_link_1to2.get(t1, [])) for t1 in trajs1}
    traj2_to_traj1_sets = {t2: set(traj_link_2to1.get(t2, [])) for t2 in trajs2}

    traj_exact = 0
    traj_split = []
    traj_merge = []
    traj_only1 = []
    traj_only2 = []
    matched_t2 = set()

    for t1 in trajs1:
        t2_set = traj1_to_traj2_sets.get(t1, set())
        n_linked = len(traj_link_1to2.get(t1, []))
        if len(t2_set) == 0:
            traj_only1.append(t1)
        elif len(t2_set) == 1:
            t2 = list(t2_set)[0]
            t1_back = traj2_to_traj1_sets.get(t2, set())
            if len(t1_back) == 1 and list(t1_back)[0] == t1:
                if n_linked == traj1_npts[t1] and len(traj_link_2to1.get(t2, [])) == traj2_npts[t2]:
                    traj_exact += 1
                    matched_t2.add(t2)
                else:
                    matched_t2.add(t2)
            else:
                matched_t2.add(t2)
        else:
            traj_split.append((t1, t2_set))
            matched_t2.update(t2_set)

    for t2 in trajs2:
        if t2 not in matched_t2 and len(traj_link_2to1.get(t2, [])) == 0:
            traj_only2.append(t2)

    for t2 in trajs2:
        t1_set = traj2_to_traj1_sets.get(t2, set())
        if len(t1_set) > 1:
            traj_merge.append((t2, t1_set))

    return {
        'py_pts': n1, 'cpp_pts': n2,
        'py_trajs': len(trajs1), 'cpp_trajs': len(trajs2),
        'matched': pt_matched, 'py_only': pt_only1, 'cpp_only': pt_only2,
        'max_diff': max_diff,
        'match_pct': 100.0 * pt_matched / max(n1, n2, 1),
        'traj_exact': traj_exact,
        'traj_split': len(traj_split),
        'traj_merge': len(traj_merge),
        'traj_only_py': len(traj_only1),
        'traj_only_cpp': len(traj_only2),
        'traj_exact_pct': 100.0 * traj_exact / max(len(trajs1), 1),
    }


def main():
    jump_values = [10]
    if len(sys.argv) > 1:
        jump_values = [float(x) for x in sys.argv[1:]]

    for jump in jump_values:
        print(f"\n{'='*80}")
        print(f"  JUMP = {jump}  (NN enabled)")
        print(f"{'='*80}")

        results = []
        for input_path, out_dir, name, nframes in samples:
            print(f"\n--- {name} ({nframes} frames) ---")

            print(f"  Running Python NN tracking (jump={jump})...")
            py_file, py_time = run_py_tracking_nn(input_path, out_dir, name, nframes, jump)
            if py_file:
                print(f"  Python: {py_time:.1f}s -> {py_file}")
            else:
                print(f"  Python: FAILED ({py_time:.1f}s)")
                continue

            print(f"  Running C++ NN tracking (jump={jump})...")
            cpp_file, cpp_time = run_cpp_tracking_nn(input_path, out_dir, name, nframes, jump)
            if cpp_file:
                print(f"  C++: {cpp_time:.1f}s -> {cpp_file}")
            else:
                print(f"  C++: FAILED ({cpp_time:.1f}s)")
                continue

            stats = compare_traces_strict(py_file, cpp_file)
            results.append((name, nframes, stats, py_time, cpp_time))

            pct = stats['match_pct']
            tpct = stats['traj_exact_pct']
            print(f"  Point-level:  Py {stats['py_pts']} / C++ {stats['cpp_pts']} | "
                  f"Matched {stats['matched']} ({pct:.2f}%) | "
                  f"Py-only {stats['py_only']} | C++-only {stats['cpp_only']} | "
                  f"MaxDiff {stats['max_diff']:.6f} px")
            print(f"  Traj-level:   Exact {stats['traj_exact']}/{stats['py_trajs']} ({tpct:.1f}%) | "
                  f"Split {stats['traj_split']} | Merge {stats['traj_merge']} | "
                  f"Py-only {stats['traj_only_py']} | C++-only {stats['traj_only_cpp']}")

        # Summary table
        if results:
            print(f"\n{'='*100}")
            print(f"  SUMMARY — jump={jump}, NN enabled")
            print(f"{'='*100}")
            print(f"{'Sample':<10} {'Frames':>6} {'Py Pts':>8} {'C++ Pts':>8} "
                  f"{'Matched':>8} {'Pt%':>7} {'MaxDiff':>10} "
                  f"{'Trajs':>6} {'Exact':>6} {'Traj%':>7} {'Split':>6} {'Merge':>6}")
            print("-" * 110)
            total_py = total_cpp = total_matched = total_py_only = total_cpp_only = 0
            total_exact = total_trajs = total_split = total_merge = 0
            for name, nframes, s, pt, ct in results:
                print(f"{name:<10} {nframes:>6} {s['py_pts']:>8} {s['cpp_pts']:>8} "
                      f"{s['matched']:>8} {s['match_pct']:>6.2f}% {s['max_diff']:>10.6f} "
                      f"{s['py_trajs']:>6} {s['traj_exact']:>6} {s['traj_exact_pct']:>6.1f}% "
                      f"{s['traj_split']:>6} {s['traj_merge']:>6}")
                total_py += s['py_pts']
                total_cpp += s['cpp_pts']
                total_matched += s['matched']
                total_exact += s['traj_exact']
                total_trajs += s['py_trajs']
                total_split += s['traj_split']
                total_merge += s['traj_merge']
            print("-" * 110)
            total_pct = 100.0 * total_matched / max(total_py, total_cpp, 1)
            total_tpct = 100.0 * total_exact / max(total_trajs, 1)
            print(f"{'TOTAL':<10} {'':>6} {total_py:>8} {total_cpp:>8} "
                  f"{total_matched:>8} {total_pct:>6.2f}% {'':>10} "
                  f"{total_trajs:>6} {total_exact:>6} {total_tpct:>6.1f}% "
                  f"{total_split:>6} {total_merge:>6}")


if __name__ == "__main__":
    main() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
