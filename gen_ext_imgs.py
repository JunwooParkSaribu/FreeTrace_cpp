"""Run Python localization while capturing ext_imgs for sharing with C++.
Monkey-patches add_block_noise to save its output.
"""
import sys, os, numpy as np
sys.path.insert(0, '/Users/junwoo/claude/FreeTrace')

sample_id = int(sys.argv[1])
ext = 'tiff' if sample_id <= 1 else 'tif'
tiff = f'inputs/sample{sample_id}.{ext}'
outdir = f'outputs_verify_s{sample_id}'
ws = 7

# Monkey-patch add_block_noise to capture ext_imgs
from FreeTrace.module import image_pad
_orig_add_block_noise = image_pad.add_block_noise
_captured_ext_imgs = []

def _patched_add_block_noise(imgs, extend):
    result = _orig_add_block_noise(imgs, extend)
    result_np = np.array(result, dtype=np.float32)
    _captured_ext_imgs.append(result_np.copy())
    return result

image_pad.add_block_noise = _patched_add_block_noise

# Also patch it in Localization module (it imports image_pad)
import FreeTrace.Localization as Loc
Loc.image_pad.add_block_noise = _patched_add_block_noise

print(f"sample{sample_id}: {tiff}")
from FreeTrace.Localization import run
run(tiff, outdir, window_size=ws, threshold=1.0, gpu_on=False, shift=1)

# Rename output
src = os.path.join(outdir, f'sample{sample_id}_loc.csv')
dst = os.path.join(outdir, 'py_loc.csv')
if os.path.exists(dst):
    os.remove(dst)
os.rename(src, dst)
print(f"  Python loc saved to {dst}")

# Concatenate all captured ext_imgs batches and save
if _captured_ext_imgs:
    all_ext = np.concatenate(_captured_ext_imgs, axis=0)
    bin_path = os.path.join(outdir, f'ext_imgs_s{sample_id}.bin')
    all_ext.astype(np.float32).flatten().tofile(bin_path)
    print(f"  ext_imgs: shape={all_ext.shape}, saved to {bin_path} ({os.path.getsize(bin_path)} bytes)")
else:
    print("  WARNING: no ext_imgs captured!")
