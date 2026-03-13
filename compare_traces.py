#!/usr/bin/env python3
"""Compare two trajectory CSV files point-by-point."""
import sys
import pandas as pd
import numpy as np

def load_traces(path):
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def main():
    f1, f2 = sys.argv[1], sys.argv[2]
    label1 = sys.argv[3] if len(sys.argv) > 3 else 'File1'
    label2 = sys.argv[4] if len(sys.argv) > 4 else 'File2'

    df1 = load_traces(f1)
    df2 = load_traces(f2)

    # Build set of (traj_idx, frame) -> (x, y)
    pts1 = {}
    for _, row in df1.iterrows():
        key = (int(row['traj_idx']), int(row['frame']))
        pts1[key] = (float(row['x']), float(row['y']))

    pts2 = {}
    for _, row in df2.iterrows():
        key = (int(row['traj_idx']), int(row['frame']))
        pts2[key] = (float(row['x']), float(row['y']))

    # Compare by (frame, x, y) — match points regardless of traj_idx
    # Build frame -> list of (x, y) for each
    from collections import defaultdict
    frame_pts1 = defaultdict(list)
    frame_pts2 = defaultdict(list)

    for (tidx, frame), (x, y) in pts1.items():
        frame_pts1[frame].append((x, y, tidx))
    for (tidx, frame), (x, y) in pts2.items():
        frame_pts2[frame].append((x, y, tidx))

    all_frames = sorted(set(frame_pts1.keys()) | set(frame_pts2.keys()))

    matched = 0
    only1 = 0
    only2 = 0
    max_diff = 0.0

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
            if best_j >= 0 and best_d < 0.01:  # within 0.01 px
                used2[best_j] = True
                matched += 1
                max_diff = max(max_diff, best_d)
            else:
                only1 += 1

        only2 += sum(1 for u in used2 if not u)

    n1 = len(pts1)
    n2 = len(pts2)
    trajs1 = len(df1['traj_idx'].unique())
    trajs2 = len(df2['traj_idx'].unique())

    print(f"=== Trajectory Comparison ===")
    print(f"  {label1}: {trajs1} trajectories, {n1} points")
    print(f"  {label2}: {trajs2} trajectories, {n2} points")
    print(f"  Matched: {matched} ({100*matched/max(n1,n2,1):.1f}%)")
    print(f"  {label1}-only: {only1}")
    print(f"  {label2}-only: {only2}")
    print(f"  Max position diff: {max_diff:.6f} px")

if __name__ == '__main__':
    main()
