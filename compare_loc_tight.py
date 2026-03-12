"""Compare Python vs C++ localization with tightened border exclusion.
Exclude detections within ws//2=3 pixels of any image edge.
"""
import csv, numpy as np, tifffile

def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({'frame': int(r['frame']), 'x': float(r['x']), 'y': float(r['y']),
                         'xvar': float(r['xvar']), 'yvar': float(r['yvar']),
                         'rho': float(r['rho']), 'intensity': float(r['intensity'])})
    return rows

def filter_interior(rows, width, height, margin=3):
    return [r for r in rows if margin <= r['x'] < width - margin and margin <= r['y'] < height - margin]

def compare(py_rows, cpp_rows, threshold=1.0):
    py_by_f, cpp_by_f = {}, {}
    for r in py_rows: py_by_f.setdefault(r['frame'], []).append(r)
    for r in cpp_rows: cpp_by_f.setdefault(r['frame'], []).append(r)
    matched, py_only, cpp_only, diffs = 0, 0, 0, []
    for frame in sorted(set(list(py_by_f)+list(cpp_by_f))):
        pf, cf = py_by_f.get(frame,[]), cpp_by_f.get(frame,[])
        if not pf: cpp_only += len(cf); continue
        if not cf: py_only += len(pf); continue
        pa = np.array([[r['x'],r['y']] for r in pf])
        ca = np.array([[r['x'],r['y']] for r in cf])
        used = set()
        for i,p in enumerate(pa):
            d = np.sqrt(np.sum((ca-p)**2, axis=1))
            for j in np.argsort(d):
                if j not in used and d[j] < threshold:
                    used.add(j); matched += 1; diffs.append(d[j]); break
            else: py_only += 1
        cpp_only += len(cf) - len(used)
    return matched, py_only, cpp_only, diffs

samples = {
    0: ('inputs/sample0.tiff',), 1: ('inputs/sample1.tiff',),
    2: ('inputs/sample2.tif',), 3: ('inputs/sample3.tif',),
    4: ('inputs/sample4.tif',), 5: ('inputs/sample5.tif',),
    6: ('inputs/sample6.tif',),
}

print(f"{'Sample':<10} {'Py(int)':<10} {'C++(int)':<10} {'Matched':<10} {'PyOnly':<8} {'C++Only':<8} {'Rate':<8} {'MaxDiff':<12} {'P99Diff':<12} {'MeanDiff':<12}")
print("-" * 110)

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
        py_int = filter_interior(py_rows, w, h, margin=3)
        cpp_int = filter_interior(cpp_rows, w, h, margin=3)
        py_excl = len(py_rows) - len(py_int)
        cpp_excl = len(cpp_rows) - len(cpp_int)
        matched, py_only, cpp_only, diffs = compare(py_int, cpp_int)
        total = matched + py_only
        rate = matched / total * 100 if total > 0 else 0
        mx = max(diffs) if diffs else 0
        p99 = np.percentile(diffs, 99) if diffs else 0
        mn = np.mean(diffs) if diffs else 0
        print(f"sample{sid:<3} {len(py_int):<10} {len(cpp_int):<10} {matched:<10} {py_only:<8} {cpp_only:<8} {rate:<8.2f} {mx:<12.8f} {p99:<12.8f} {mn:<12.8f}")
        print(f"          (excluded: py={py_excl}, cpp={cpp_excl}, margin=3, img={w}x{h})")
    except FileNotFoundError as e:
        print(f"sample{sid:<3} MISSING: {e}")
