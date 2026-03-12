"""Debug GMM intermediate values for Python vs C++ comparison."""
import sys, os
import numpy as np
import math
sys.path.insert(0, '/Users/junwoo/claude/FreeTrace')
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
import pandas as pd

def gmm_bic_score(estimator, x):
    return -estimator.bic(x)

samples = {
    0: ('inputs/sample0.tiff', 100),
    1: ('inputs/sample1.tiff', 350),
    2: ('inputs/sample2.tif', 2001),
    3: ('inputs/sample3.tif', 2001),
    4: ('inputs/sample4.tif', 5000),
    5: ('inputs/sample5.tif', 1001),
    6: ('inputs/sample6.tif', 40),
}

ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else sorted(samples.keys())

import FreeTrace.Tracking as T

# Monkey-patch approximation to intercept raw_dist
captured = {}

_orig_approx = T.approximation
def _patched_approx(raw_dist, time_forecast=2, jump_threshold=None):
    captured['raw_dist'] = raw_dist
    captured['tf'] = time_forecast
    return _orig_approx(raw_dist, time_forecast=time_forecast, jump_threshold=jump_threshold)
T.approximation = _patched_approx

for sid in ids:
    tiff, nf = samples[sid]
    loc_csv = f'outputs_verify_s{sid}/py_loc.csv'
    if not os.path.exists(loc_csv):
        print(f"sample{sid}: SKIP"); continue

    captured.clear()
    T.run(tiff, f'outputs_verify_s{sid}_gmm_debug', graph_depth=2, cutoff=2,
          jump_threshold=None, gpu_on=False, read_loc_file=(loc_csv, 1.0))

    if 'raw_dist' not in captured:
        print(f"sample{sid}: no raw_dist captured"); continue

    raw_dist = captured['raw_dist']
    print(f"\n{'='*60}")
    print(f"=== sample{sid} ===")
    print(f"{'='*60}")
    print(f"  raw_dist has {len(raw_dist)} dimensions")

    # Replicate approx_gauss step by step
    max_xyz = []
    distributions = raw_dist

    # Quantile filter (same as Python approx_gauss)
    qt_distributions = []
    for i, distribution in enumerate(distributions):
        var_check = np.var(distribution)
        if var_check > 1e-5:
            distribution = np.array(distribution)
            quantile = np.quantile(distribution, [0.025, 0.975])
            filtered = distribution[(distribution > quantile[0]) * (distribution < quantile[1])]
            qt_distributions.append(filtered)
            print(f"  dim{i}: n_raw={len(distribution)}, var={var_check:.6f}, q025={quantile[0]:.6f}, q975={quantile[1]:.6f}, n_filtered={len(filtered)}")
        else:
            print(f"  dim{i}: SKIPPED (var={var_check:.2e})")
    distributions = qt_distributions

    for dim_idx, distribution in enumerate(distributions):
        print(f"\n  --- Dimension {dim_idx} ---")
        var_check = np.var(distribution)
        if var_check <= 1e-5:
            print(f"    SKIPPED after filter (var={var_check:.2e})")
            continue

        # GridSearchCV step (5-fold CV by default)
        param_grid = [
            {"n_components": [1], "means_init": [[[0]]]},
            {"n_components": [2], "means_init": [[[0], [0]]]},
            {"n_components": [3], "means_init": [[[0], [0], [0]]]},
        ]
        grid_search = GridSearchCV(
            GaussianMixture(max_iter=100, n_init=3, covariance_type='full'),
            param_grid=param_grid,
            scoring=gmm_bic_score, verbose=0
        )
        grid_search.fit(distribution.reshape(-1, 1))

        cluster_df = pd.DataFrame(grid_search.cv_results_)[["param_n_components", "mean_test_score"]]
        cluster_df["mean_test_score"] = -cluster_df["mean_test_score"]
        for _, row in cluster_df.iterrows():
            nc = row["param_n_components"]
            bic = row["mean_test_score"]
            print(f"    CV BIC[n_comp={nc}] = {bic:.4f}")

        opt_nb_component = int(np.argmin(cluster_df["mean_test_score"].values) + 1)
        print(f"    Selected n_components (CV) = {opt_nb_component}")

        # Also compute non-CV BIC (like C++ does)
        print(f"    --- Full-data BIC (no CV, like C++) ---")
        best_full_nc = 1
        best_full_bic = 1e30
        for nc in [1, 2, 3]:
            gm = GaussianMixture(n_components=nc, max_iter=100, n_init=3,
                                 covariance_type='full',
                                 means_init=[[0]]*nc).fit(distribution.reshape(-1, 1))
            bic_full = gm.bic(distribution.reshape(-1, 1))
            if bic_full < best_full_bic:
                best_full_bic = bic_full
                best_full_nc = nc
            print(f"    Full BIC[n_comp={nc}] = {bic_full:.4f} (means={gm.means_.flatten()}, vars={gm.covariances_.flatten()}, w={gm.weights_})")
        print(f"    Selected n_components (full data) = {best_full_nc}")

        # BayesianGMM step
        cluster = BayesianGaussianMixture(
            n_components=opt_nb_component, max_iter=100, n_init=3,
            mean_prior=[0], mean_precision_prior=1e7, covariance_type='full'
        ).fit(distribution.reshape(-1, 1))

        print(f"    BayesianGMM (n_comp={opt_nb_component}):")
        print(f"      means    = {cluster.means_.flatten()}")
        print(f"      covars   = {cluster.covariances_.flatten()}")
        print(f"      weights  = {cluster.weights_}")

        # Component selection
        selec_var = []
        for mean_, cov_, weight_ in zip(cluster.means_.flatten(), cluster.covariances_.flatten(), cluster.weights_.flatten()):
            selected = (-1 < mean_ < 1 and weight_ > 0.05)
            print(f"      comp: mean={mean_:.6f}, cov={cov_:.6f}, weight={weight_:.6f} -> {'SEL' if selected else 'rej'}")
            if selected:
                selec_var.append(cov_)

        if selec_var:
            max_var = max(selec_var)
            threshold_dim = math.sqrt(max_var) * 2.5
            max_xyz.append(threshold_dim)
            print(f"    max_var={max_var:.6f}, sqrt*2.5={threshold_dim:.6f}")

    max_euclid = math.sqrt(sum(v**2 for v in max_xyz))
    max_euclid = max(max_euclid, 5.0)
    print(f"\n  FINAL THRESHOLD = {max_euclid:.6f}")
