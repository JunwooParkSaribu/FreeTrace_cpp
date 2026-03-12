"""Compare Python vs C++ FreeTrace localization, excluding border detections."""
import csv
import sys
import numpy as np
sys.path.insert(0, '/Users/junwoo/claude/FreeTrace')
import tifffile

def load_csv(path):
    """Load localization CSV, return list of (frame, x, y, ...) rows."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'frame': int(r['frame']),
                'x': float(r['x']),
                'y': float(r['y']),
                'xvar': float(r['xvar']),
                'yvar': float(r['yvar']),
                'rho': float(r['rho']),
                'intensity': float(r['intensity']),
            })
    return rows

def filter_interior(rows, width, height):
    """Keep only detections within original image bounds [0, width) x [0, height)."""
    return [r for r in rows if 0 <= r['x'] < width and 0 <= r['y'] < height]

def compare(py_rows, cpp_rows, threshold=1.0):
    """Per-frame nearest-neighbor matching within threshold."""
    # Group by frame
    py_by_frame = {}
    for r in py_rows:
        py_by_frame.setdefault(r['frame'], []).append(r)
    cpp_by_frame = {}
    for r in cpp_rows:
        cpp_by_frame.setdefault(r['frame'], []).append(r)

    all_frames = sorted(set(list(py_by_frame.keys()) + list(cpp_by_frame.keys())))

    matched = 0
    py_only = 0
    cpp_only = 0
    pos_diffs = []

    for frame in all_frames:
        py_f = py_by_frame.get(frame, [])
        cpp_f = cpp_by_frame.get(frame, [])

        if not py_f:
            cpp_only += len(cpp_f)
            continue
        if not cpp_f:
            py_only += len(py_f)
            continue

        # Build arrays for vectorized distance
        py_xy = np.array([[r['x'], r['y']] for r in py_f])
        cpp_xy = np.array([[r['x'], r['y']] for r in cpp_f])

        used_cpp = set()
        for i, pxy in enumerate(py_xy):
            dists = np.sqrt(np.sum((cpp_xy - pxy)**2, axis=1))
            order = np.argsort(dists)
            found = False
            for j in order:
                if j not in used_cpp and dists[j] < threshold:
                    used_cpp.add(j)
                    matched += 1
                    pos_diffs.append(dists[j])
                    found = True
                    break
            if not found:
                py_only += 1
        cpp_only += len(cpp_f) - len(used_cpp)

    return matched, py_only, cpp_only, pos_diffs

# Sample info: (tiff_path, extension)
samples = {
    0: ('inputs/sample0.tiff',),
    1: ('inputs/sample1.tiff',),
    2: ('inputs/sample2.tif',),
    3: ('inputs/sample3.tif',),
    4: ('inputs/sample4.tif',),
    5: ('inputs/sample5.tif',),
    6: ('inputs/sample6.tif',),
}

print(f"{'Sample':<10} {'Py(int)':<10} {'C++(int)':<10} {'Matched':<10} {'PyOnly':<8} {'C++Only':<8} {'Rate':<8} {'MaxDiff':<10} {'P99Diff':<10}")
print("-" * 94)

for sid in sorted(samples.keys()):
    tiff_path = samples[sid][0]
    py_csv = f'outputs_verify_s{sid}/py_loc.csv'
    cpp_csv = f'outputs_verify_s{sid}/sample{sid}_loc.csv'

    try:
        img = tifffile.imread(tiff_path)
        h, w = img.shape[1], img.shape[2]
        del img

        py_rows = load_csv(py_csv)
        cpp_rows = load_csv(cpp_csv)

        py_total = len(py_rows)
        cpp_total = len(cpp_rows)

        # Filter to interior only
        py_int = filter_interior(py_rows, w, h)
        cpp_int = filter_interior(cpp_rows, w, h)

        py_border = py_total - len(py_int)
        cpp_border = cpp_total - len(cpp_int)

        matched, py_only, cpp_only, pos_diffs = compare(py_int, cpp_int)

        total_int = matched + py_only
        rate = matched / total_int * 100 if total_int > 0 else 0
        max_diff = max(pos_diffs) if pos_diffs else 0
        p99_diff = np.percentile(pos_diffs, 99) if pos_diffs else 0

        print(f"sample{sid:<3} {len(py_int):<10} {len(cpp_int):<10} {matched:<10} {py_only:<8} {cpp_only:<8} {rate:<8.2f} {max_diff:<10.6f} {p99_diff:<10.6f}")
        print(f"          (border excluded: py={py_border}, cpp={cpp_border}, img={w}x{h})")
    except FileNotFoundError as e:
        print(f"sample{sid:<3} MISSING: {e}")
    except Exception as e:
        print(f"sample{sid:<3} ERROR: {e}")
