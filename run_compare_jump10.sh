#!/bin/bash
# Compare Python vs C++ tracking with fixed jump threshold 10 on all samples
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
set -e

PYTHON=~/claude/claude_venv/bin/python3
CPP=./freetrace

echo "=== Tracking Comparison (jump=10, double precision) ==="
echo ""

for s in 0 1 2 3 4 5 6; do
    dir="outputs_verify_s${s}"
    loc="${dir}/loc_py.csv"

    if [ ! -f "$loc" ]; then
        echo "SKIP sample${s}: no loc_py.csv"
        continue
    fi

    case $s in
        0) frames=100; tiff="inputs/sample0.tiff" ;;
        1) frames=350; tiff="inputs/sample1.tiff" ;;
        2) frames=2001; tiff="inputs/sample2.tif" ;;
        3) frames=2001; tiff="inputs/sample3.tif" ;;
        4) frames=5000; tiff="inputs/sample4.tif" ;;
        5) frames=1001; tiff="inputs/sample5.tif" ;;
        6) frames=40; tiff="inputs/sample6.tif" ;;
    esac

    echo "--- Sample $s ($frames frames) ---"

    # Python (no GPU, jump=10)
    $PYTHON test_tracking_steps.py "$loc" $frames --jump 10 2>/dev/null | grep -E "^(Step 6|  [0-9]+ traj|  Total)" || true

    # C++ (no NN, jump=10)
    $CPP track "$loc" "$dir" $frames --jump 10 --tiff "$tiff" 2>/dev/null | grep -E "^(Found|Jump)" || true

    # Compare
    $PYTHON compare_traces.py "${dir}/step_6_traces_py.csv" "${dir}/loc_py.csv_traces.csv" Python C++
    echo ""
done
