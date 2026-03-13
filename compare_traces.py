#!/usr/bin/env python3
"""Compare two trajectory CSV files with strict point-level AND trajectory-level matching.""" # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

def load_traces(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def main():
    f1, f2 = sys.argv[1], sys.argv[2]
    label1 = sys.argv[3] if len(sys.argv) > 3 else 'File1'
    label2 = sys.argv[4] if len(sys.argv) > 4 else 'File2'

    df1 = load_traces(f1)
    df2 = load_traces(f2)

    # ---- 1. Point-level matching (per frame, position-based) ----
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
    # Track which traj_idx pairs are linked by matched points
    # point_map: (label1_traj, frame) -> (label2_traj, frame)
    traj_link_1to2 = defaultdict(list)  # label1_traj -> list of label2_traj
    traj_link_2to1 = defaultdict(list)  # label2_traj -> list of label1_traj

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

    n1 = sum(len(v) for v in frame_pts1.values())
    n2 = sum(len(v) for v in frame_pts2.values())
    trajs1 = sorted(df1['traj_idx'].unique())
    trajs2 = sorted(df2['traj_idx'].unique())

    # ---- 2. Trajectory-level matching ----
    # A trajectory in label1 is "fully matched" if ALL its points map to the SAME label2 trajectory
    # and that label2 trajectory also maps entirely back to this label1 trajectory (bijective).

    # Build per-trajectory point counts
    traj1_npts = df1.groupby('traj_idx').size().to_dict()
    traj2_npts = df2.groupby('traj_idx').size().to_dict()

    # For each label1 traj, find which label2 trajs its points mapped to
    traj1_to_traj2_sets = {}
    for t1 in trajs1:
        linked = traj_link_1to2.get(t1, [])
        traj1_to_traj2_sets[t1] = set(linked)

    traj2_to_traj1_sets = {}
    for t2 in trajs2:
        linked = traj_link_2to1.get(t2, [])
        traj2_to_traj1_sets[t2] = set(linked)

    # Classify trajectories
    traj_exact = 0       # bijective 1:1 match, same length, all points matched
    traj_split = []       # label1 traj maps to multiple label2 trajs
    traj_merge = []       # multiple label1 trajs map to same label2 traj
    traj_only1 = []       # label1 traj with no matched points
    traj_only2 = []       # label2 traj with no matched points
    traj_partial = []     # partial overlap (not exact, not split/merge)

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
                # Bijective: check all points matched
                if n_linked == traj1_npts[t1] and len(traj_link_2to1.get(t2, [])) == traj2_npts[t2]:
                    traj_exact += 1
                    matched_t2.add(t2)
                else:
                    traj_partial.append((t1, t2_set))
                    matched_t2.add(t2)
            else:
                # This t2 is shared with other t1s — merge
                traj_partial.append((t1, t2_set))
                matched_t2.add(t2)
        else:
            # Split: label1 traj mapped to multiple label2 trajs
            traj_split.append((t1, t2_set))
            matched_t2.update(t2_set)

    for t2 in trajs2:
        if t2 not in matched_t2:
            if len(traj_link_2to1.get(t2, [])) == 0:
                traj_only2.append(t2)

    # Detect merges: label2 traj receiving points from multiple label1 trajs
    for t2 in trajs2:
        t1_set = traj2_to_traj1_sets.get(t2, set())
        if len(t1_set) > 1:
            traj_merge.append((t2, t1_set))

    # ---- 3. Output ----
    print(f"=== Trajectory Comparison ===")
    print(f"  {label1}: {len(trajs1)} trajectories, {n1} points")
    print(f"  {label2}: {len(trajs2)} trajectories, {n2} points")
    print(f"--- Point-level ---")
    print(f"  Matched: {pt_matched} ({100*pt_matched/max(n1,n2,1):.1f}%)")
    print(f"  {label1}-only: {pt_only1}")
    print(f"  {label2}-only: {pt_only2}")
    print(f"  Max position diff: {max_diff:.6f} px")
    print(f"--- Trajectory-level ---")
    print(f"  Exact match: {traj_exact} / {len(trajs1)} {label1} trajs ({100*traj_exact/max(len(trajs1),1):.1f}%)")
    if traj_split:
        print(f"  Split ({label1}->multiple {label2}): {len(traj_split)}")
        for t1, t2s in traj_split[:5]:
            print(f"    {label1} traj {t1} ({traj1_npts[t1]} pts) -> {label2} trajs {t2s}")
    if traj_merge:
        print(f"  Merge (multiple {label1}->{label2}): {len(traj_merge)}")
        for t2, t1s in traj_merge[:5]:
            print(f"    {label2} traj {t2} ({traj2_npts[t2]} pts) <- {label1} trajs {t1s}")
    if traj_only1:
        print(f"  {label1}-only trajs: {len(traj_only1)}")
    if traj_only2:
        print(f"  {label2}-only trajs: {len(traj_only2)}")
    if traj_partial:
        n_non_exact = len(traj_partial)
        if n_non_exact > 0 and not traj_split and not traj_merge:
            print(f"  Partial match: {n_non_exact}")
            for t1, t2s in traj_partial[:5]:
                print(f"    {label1} traj {t1} ({traj1_npts[t1]} pts) <-> {label2} trajs {t2s}") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

if __name__ == '__main__':
    main()
