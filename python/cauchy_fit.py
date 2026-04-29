"""
Unified Cauchy MLE module for Hurst-exponent estimation from displacement
ratios of fBm increments under the blur + noise-aware model.

Model
-----
Under the corrected Cauchy model (equation ch1_rho_blur in thesis):

    mu(H)    = rho_{Delta,R}(H; K, sigma_loc) =
                   (K * J_cov(H,R,Delta) - sigma_loc**2)
                   / (2 * (K * J_var(H,R,Delta) + sigma_loc**2))
    gamma(H) = sqrt(1 - mu(H)**2)

The ratio r = X_{i+1}^Delta / X_i^Delta is exactly Cauchy(mu(H), gamma(H)).

Estimator: CONSTRAINED Cauchy MLE, parametrised by H alone.
This is the estimator achieving the CRLB derived in the thesis.

    H_hat = argmax_H  [ n*log(gamma(H)) - n*log(pi)
                        - sum_i log((r_i - mu(H))**2 + gamma(H)**2) ]

Plug-in parameters (must be supplied externally):
    R          : exposure ratio tau_exp / Delta_t_frame  (0 = instantaneous, 1 = full)
    sigma_loc  : localisation CRLB (same units as positions)
    K          : diffusion-like coefficient  (K * J_var has units of variance)
    Delta      : time-span integer for Delta-frame increments

Fisher information and CRLB
---------------------------
Per-ratio Fisher info on H under the corrected Cauchy model (thesis eq. 52):

    I_1(H; Delta, R, K, sigma_loc) = (drho/dH)**2 / (2 * (1 - rho**2)**2)

Asymptotic CRLB:

    Var(H_hat)  >=  2 * (1 - rho**2)**2 / (n_eff * (drho/dH)**2)

where n_eff is the effective sample size accounting for correlated ratios
(see Tier-3 derivation in ch1_neff_* scripts).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


# ----------------------------------------------------------------------------
# Blur-aware J_var, J_cov (thesis: equations ch1_blur_var, ch1_blur_cov)
# ----------------------------------------------------------------------------

def J_var(H: float, R: float, Delta: float) -> float:
    """Variance integral for blur-aware fBm, normalised by K.

    Var(X_i^Delta) = K * J_var(H, R, Delta)
    """
    if R <= 1e-9:
        return float(Delta ** (2 * H))
    a = 2 * H + 2
    dd = (2 * H + 1) * (2 * H + 2)
    return float(((Delta + R) ** a + (Delta - R) ** a - 2 * Delta ** a - 2 * R ** a)
                 / (R ** 2 * dd))


def J_cov(H: float, R: float, Delta: float) -> float:
    """Covariance integral between consecutive blur-aware fBm increments.

    Cov(X_i^Delta, X_{i+1}^Delta) = K * J_cov(H, R, Delta)
    """
    if R <= 1e-9:
        return float((Delta ** (2 * H)) * (2 ** (2 * H) - 2))
    a = 2 * H + 2
    dd = (2 * H + 1) * (2 * H + 2)
    return float(((2 * Delta + R) ** a + (2 * Delta - R) ** a - 2 * (2 * Delta) ** a
                  - 2 * (Delta + R) ** a - 2 * (Delta - R) ** a + 4 * Delta ** a + 2 * R ** a)
                 / (R ** 2 * dd))


def rho_corrected(H: float, Delta: float, R: float,
                  K: float, sigma_loc: float) -> float:
    """Blur + noise-aware Cauchy location parameter mu(H)."""
    s2 = sigma_loc ** 2
    num = K * J_cov(H, R, Delta) - s2
    den = 2 * (K * J_var(H, R, Delta) + s2)
    return num / den


def gamma_corrected(H: float, Delta: float, R: float,
                    K: float, sigma_loc: float) -> float:
    """Blur + noise-aware Cauchy scale parameter gamma(H) = sqrt(1 - rho**2)."""
    rho = rho_corrected(H, Delta, R, K, sigma_loc)
    return float(np.sqrt(max(1.0 - rho ** 2, 1e-14)))


def drho_dH_numerical(H: float, Delta: float, R: float, K: float,
                      sigma_loc: float, h: float = 1e-5) -> float:
    """Numerical derivative d(rho)/d(H) via centred difference."""
    rp = rho_corrected(H + h, Delta, R, K, sigma_loc)
    rm = rho_corrected(H - h, Delta, R, K, sigma_loc)
    return (rp - rm) / (2 * h)


# ----------------------------------------------------------------------------
# Constrained Cauchy MLE: maximises joint log-likelihood over H
# ----------------------------------------------------------------------------

def _neg_log_likelihood(H: float, ratios: np.ndarray, Delta: float, R: float,
                        K: float, sigma_loc: float) -> float:
    """Negative log-likelihood of ratios under Cauchy(mu(H), gamma(H))."""
    mu = rho_corrected(H, Delta, R, K, sigma_loc)
    g2 = max(1.0 - mu ** 2, 1e-12)
    # log f(r; mu, gamma) = 0.5*log(g2) - log(pi) - log((r-mu)**2 + g2)
    resid2 = (ratios - mu) ** 2
    return -float(np.sum(0.5 * np.log(g2) - np.log(np.pi) - np.log(resid2 + g2)))


def fit_cauchy(ratios: np.ndarray,
               Delta: float,
               R: float = 0.0,
               sigma_loc: float = 0.0,
               K: float = 1.0,
               H_bounds: tuple[float, float] = (0.02, 0.98),
               method: str = "bounded") -> dict:
    """Constrained Cauchy MLE for H on observed displacement ratios.

    Parameters
    ----------
    ratios     : 1D array of displacement ratios r_i = X_{i+1}^Delta / X_i^Delta
    Delta      : time-span (frames) of the underlying increments
    R          : exposure ratio (0 = instantaneous). Default 0.
    sigma_loc  : localisation CRLB (same units as positions). Default 0.
    K          : diffusion-like coefficient. Default 1 (dimensionless / simulations).
    H_bounds   : search bounds for H. Default (0.02, 0.98).
    method     : "bounded" (Brent's, robust) or "powell"/"BFGS".

    Returns
    -------
    dict with keys
      'H'        : MLE estimate
      'rho'      : corresponding rho_{Delta,R}(H_hat)
      'gamma'    : corresponding scale parameter
      'nll'      : negative log-likelihood at MLE
      'drho_dH'  : dRho/dH evaluated at H_hat (for CRLB)
      'converged': bool
    """
    ratios = np.asarray(ratios, dtype=float)
    if ratios.size < 5:
        return dict(H=np.nan, rho=np.nan, gamma=np.nan, nll=np.inf,
                    drho_dH=np.nan, converged=False)

    kwargs = dict(args=(ratios, Delta, R, K, sigma_loc))

    if method == "bounded":
        res = minimize_scalar(_neg_log_likelihood,
                              bounds=H_bounds,
                              method="bounded",
                              options=dict(xatol=1e-8),
                              **kwargs)
    else:
        from scipy.optimize import minimize
        res = minimize(lambda p: _neg_log_likelihood(p[0], *kwargs['args']),
                       x0=[0.5], method=method)

    H_hat = float(res.x) if np.ndim(res.x) == 0 else float(res.x[0])
    H_hat = float(np.clip(H_hat, H_bounds[0] + 1e-6, H_bounds[1] - 1e-6))
    rho = rho_corrected(H_hat, Delta, R, K, sigma_loc)
    gamma = float(np.sqrt(max(1.0 - rho ** 2, 1e-14)))
    drho = drho_dH_numerical(H_hat, Delta, R, K, sigma_loc)
    return dict(H=H_hat, rho=rho, gamma=gamma,
                nll=float(res.fun), drho_dH=float(drho),
                converged=bool(res.success))


# ----------------------------------------------------------------------------
# CRLB helpers (thesis equations ch1_noise_aware_fisher, ch1_noise_aware_crlb)
# ----------------------------------------------------------------------------

def fisher_H_per_ratio(H: float, Delta: float, R: float,
                       K: float, sigma_loc: float) -> float:
    """Per-ratio Fisher information on H under the corrected Cauchy model.

        I_1(H; Delta, R, K, sigma_loc)
            = (drho/dH)**2 / (2 * (1 - rho**2)**2)
    """
    rho = rho_corrected(H, Delta, R, K, sigma_loc)
    drho = drho_dH_numerical(H, Delta, R, K, sigma_loc)
    denom = 2.0 * (1.0 - rho ** 2) ** 2
    return float(drho ** 2 / denom) if denom > 1e-14 else np.inf


def var_H_crlb(H: float, Delta: float, R: float, K: float,
               sigma_loc: float, n_eff: float) -> float:
    """Cramer-Rao lower bound on Var(H_hat) for constrained MLE.

        Var(H_hat) >= 2 * (1 - rho**2)**2 / (n_eff * (drho/dH)**2)
    """
    if n_eff <= 0:
        return np.inf
    fisher = fisher_H_per_ratio(H, Delta, R, K, sigma_loc)
    if not np.isfinite(fisher) or fisher <= 0:  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        # drho/dH ≈ 0 (information singularity, e.g. at boundary H) → CRLB unbounded
        return np.inf
    return 1.0 / (n_eff * fisher)


def sigma_H_crlb(H: float, Delta: float, R: float, K: float,
                 sigma_loc: float, n_eff: float) -> float:
    """Standard deviation bound sqrt(Var(H_hat))."""
    v = var_H_crlb(H, Delta, R, K, sigma_loc, n_eff)
    return float(np.sqrt(v)) if np.isfinite(v) else np.inf


def n_eff_from_var_H(var_H: float, H: float, Delta: float, R: float,
                     K: float, sigma_loc: float) -> float:
    """Invert the CRLB formula to compute empirical n_eff from Var(H_hat).

    n_eff = 2 * (1 - rho**2)**2 / (Var(H_hat) * (drho/dH)**2)
    """
    rho = rho_corrected(H, Delta, R, K, sigma_loc)
    drho = drho_dH_numerical(H, Delta, R, K, sigma_loc)
    if var_H <= 0 or abs(drho) < 1e-14:
        return np.nan
    return 2.0 * (1 - rho ** 2) ** 2 / (var_H * drho ** 2)


# ----------------------------------------------------------------------------
# Data reduction helpers
# ----------------------------------------------------------------------------

def extract_ratios(trajs, Delta: int, min_abs_denom: float = 1e-15) -> np.ndarray:
    """Extract Delta-span displacement ratios from a list of 2D trajectories.

    Each entry in `trajs` is an array of shape (T, >=3) where columns 1,2
    are the x,y coordinates. Uses all Delta offset chains, as required for
    consistent Cauchy fitting.
    """
    out = []
    for tr in trajs:
        for coord in (1, 2):
            p = tr[:, coord]
            if len(p) < 2 * Delta + 1:
                continue
            for off in range(Delta):
                sub = p[off::Delta]
                inc = np.diff(sub)
                if len(inc) < 2:
                    continue
                den = inc[:-1]
                num = inc[1:]
                valid = np.abs(den) > min_abs_denom
                out.append(num[valid] / den[valid])
    return np.concatenate(out) if out else np.array([])


def count_ratios(trajs, Delta: int) -> int:
    """Total number of Delta-span ratios extractable (includes both x and y)."""
    total = 0
    for tr in trajs:
        # Using N_t(T, Delta) = max(T - 2*Delta, 0) per coordinate
        T = len(tr) if hasattr(tr, "__len__") else tr
        T_int = T if isinstance(T, (int, np.integer)) else len(tr)
        total += max(T_int - 2 * Delta, 0)
    return 2 * total  # x and y


def pool_sigma_loc_rms(sigma_per_loc: np.ndarray) -> float:
    """Pool per-localisation sigma_loc values as RMS (not median).

    Reason: plug-in sigma_loc enters the CRLB/Cauchy model via sigma_loc**2.
    Per user feedback_sigma_loc_plugin.md, the correct pool is
        sigma_loc_rms = sqrt( mean(sigma**2) )
    not median(sigma). Using the median under-estimates the noise variance.

    Input
    -----
    sigma_per_loc : 1D array of per-localisation sigma values (same units as positions)

    Output
    ------
    scalar rms value
    """
    s = np.asarray(sigma_per_loc, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(s ** 2)))


# ----------------------------------------------------------------------------
# Tier-3 n_eff THEORY for constrained MLE (exact, closed-form modulo cov tables)
# ----------------------------------------------------------------------------
#
# Pair score cov for the H-score s_H = mu' * s_rho + gam' * s_gam:
#
#     C_H(u; H) = mu'^2 * S_rr(u;H) + gam'^2 * S_gg(u;H) + mu'*gam' * S_rg(u;H)
#
# where S_rr, S_gg, S_rg are tabulated via Gaussian MC once per H (see
# ch1_neff_constrained_cov_table.py). For each trajectory of length T at Delta:
#
#     N_t = max(T - 2*Delta, 0)  ratios per coordinate
#     S_H_t = N_t * I_1_H + 2 * sum_{s=1}^{N_t-1} (N_t - s) * C_H(s/Delta)
#
# Total (both coordinates, all trajs):
#     N_total = 2 * sum_traj N_t
#     S_total_H = 2 * sum_traj S_H_t
#     n_eff_theory = N_total^2 * I_1_H / S_total_H
# ----------------------------------------------------------------------------

_COV_TABLE_CACHE: dict[float, dict] = {}


_AVAILABLE_H_GRID = (0.25, 0.5, 0.75)  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28


def _load_cov_table(H: float) -> dict:
    """Load the tabulated (S_rr, S_gg, S_rg) cov table for the given H.

    Snaps H to the nearest tabulated value in _AVAILABLE_H_GRID = {0.25, 0.5, 0.75}.
    Tables are generated by /tmp/gen_cov_table.py (a parametrised port of
    PhD_thesis/Figures/tmp/ch1_neff_constrained_cov_table.py) and bundled next to
    cauchy_fit.py as cauchy_neff_cov_H<NNN>.npz where NNN = round(H*1000).
    Falls back to the H=0.25 table when an exact-H file is missing (with a
    one-time warning).
    """  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
    import os as _os
    # Snap to nearest available tabulated H.
    H_snap = min(_AVAILABLE_H_GRID, key=lambda h: abs(h - H))
    key = round(H_snap, 4)
    if key in _COV_TABLE_CACHE:
        return _COV_TABLE_CACHE[key]

    h_tag = f"{int(round(H_snap * 1000)):03d}"
    here = _os.path.dirname(_os.path.abspath(__file__))
    candidates = [
        _os.path.join(here, f"cauchy_neff_cov_H{h_tag}.npz"),
    ]
    if H_snap == 0.25:  # legacy path also accepted
        candidates.append(
            "/home/junwoo/claude/PhD_thesis/Figures/tmp/ch1_neff_constrained_cov_table.npz"
        )
    path = None
    for p in candidates:
        if _os.path.exists(p):
            path = p
            break
    if path is None:
        # Fallback: use H=0.25 table with a warning.
        if 0.25 in _COV_TABLE_CACHE:
            _COV_TABLE_CACHE[key] = _COV_TABLE_CACHE[0.25]
            return _COV_TABLE_CACHE[key]
        fallback = _os.path.join(here, "cauchy_neff_cov_H025.npz")
        if _os.path.exists(fallback):
            import warnings as _warnings
            _warnings.warn(
                f"cauchy_neff_cov_H{h_tag}.npz not found; falling back to H=0.25 table. "
                f"n_eff and CRLB bands at fitted H={H:.3g} will be biased.",
                RuntimeWarning, stacklevel=2,
            )
            d = np.load(fallback)
            out = dict(u=d["u"], S_rr=d["S_rr"], S_gg=d["S_gg"], S_rg=d["S_rg"])
            _COV_TABLE_CACHE[key] = out
            _COV_TABLE_CACHE[0.25] = out
            return out
        raise RuntimeError(
            f"cov tables not found near cauchy_fit.py "
            f"(looked for cauchy_neff_cov_H{h_tag}.npz and H025 fallback)"
        )
    d = np.load(path)
    out = dict(u=d["u"], S_rr=d["S_rr"], S_gg=d["S_gg"], S_rg=d["S_rg"])
    _COV_TABLE_CACHE[key] = out
    return out


def n_eff_theory(H: float, Delta: int, trajectory_lengths,
                 R: float = 0.0, K: float = 1.0, sigma_loc: float = 0.0) -> float:
    """Exact theoretical n_eff(Delta) for the constrained Cauchy MLE.

    Parameters
    ----------
    H                    : Hurst exponent
    Delta                : time span (int)
    trajectory_lengths   : iterable of integer trajectory lengths
    R, K, sigma_loc      : blur/noise/diffusion plug-in parameters

    Returns
    -------
    n_eff  (scalar)

    Notes
    -----
    Requires the tabulated (S_rr, S_gg, S_rg) cov table for this H.
    Validated to within ~10% of 300-run constrained-MLE empirical (step7),
    matching 97% of bootstrap 95% CIs across Delta in [1, 69].
    """
    tbl = _load_cov_table(H)
    u_grid = tbl["u"]; S_rr = tbl["S_rr"]; S_gg = tbl["S_gg"]; S_rg = tbl["S_rg"]

    # Cauchy location & scale under the corrected model
    rho = rho_corrected(H, Delta, R, K, sigma_loc)
    gam2 = 1.0 - rho ** 2
    gam = float(np.sqrt(max(gam2, 1e-14)))
    mu_p = drho_dH_numerical(H, Delta, R, K, sigma_loc)
    gam_p = -rho * mu_p / gam
    I1_H = mu_p ** 2 / (2.0 * gam2 ** 2)  # per-sample Fisher info

    # Combined H-score pair cov as a function of u
    def C_H(u_arr: np.ndarray) -> np.ndarray:
        u_arr = np.abs(np.atleast_1d(u_arr))
        Srr = np.interp(u_arr, u_grid, S_rr)
        Sgg = np.interp(u_arr, u_grid, S_gg)
        Srg = np.interp(u_arr, u_grid, S_rg)
        # zero out beyond table range
        out_of_range = u_arr > u_grid[-1]
        Srr[out_of_range] = 0.0
        Sgg[out_of_range] = 0.0
        Srg[out_of_range] = 0.0
        return mu_p ** 2 * Srr + gam_p ** 2 * Sgg + mu_p * gam_p * Srg

    N_total = 0
    S_total_H = 0.0
    for T in trajectory_lengths:
        N_t = int(max(T - 2 * Delta, 0))
        if N_t < 1:
            continue
        S_coord = N_t * I1_H  # diagonal contribution per coord
        if N_t > 1:
            s_arr = np.arange(1, N_t)
            u_arr = s_arr / Delta
            S_coord += 2.0 * float(np.sum((N_t - s_arr) * C_H(u_arr)))
        # x, y independent coordinates
        S_total_H += 2.0 * S_coord
        N_total   += 2 * N_t

    if S_total_H <= 0 or N_total == 0:
        return np.nan
    return float(N_total ** 2 * I1_H / S_total_H)


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Test 1: noise-free, no blur → rho = (2^(2H) - 2)/2 (Delta-independent)
    for H_true in [0.25, 0.5, 0.75]:
        rho_th = (2 ** (2 * H_true) - 2) / 2
        rho_comp = rho_corrected(H_true, Delta=1, R=0.0, K=1.0, sigma_loc=0.0)
        assert abs(rho_th - rho_comp) < 1e-10, (H_true, rho_th, rho_comp)
        print(f"  H={H_true}: rho_th={rho_th:+.4f}, rho_fn={rho_comp:+.4f}  OK")

    # Test 2: fit an iid Cauchy sample with known parameters
    from scipy.stats import cauchy
    H_true = 0.25
    rho_true = rho_corrected(H_true, 1, 0.0, 1.0, 0.0)
    gamma_true = np.sqrt(1 - rho_true ** 2)
    print(f"\nIID Cauchy test at H={H_true}: rho={rho_true:.4f}, gamma={gamma_true:.4f}")
    for n in [1000, 5000, 50000]:
        Hs = []
        for trial in range(200):
            r = cauchy.rvs(loc=rho_true, scale=gamma_true, size=n,
                           random_state=rng.integers(2**31))
            res = fit_cauchy(r, Delta=1, R=0.0, K=1.0, sigma_loc=0.0)
            Hs.append(res["H"])
        Hs = np.array(Hs)
        print(f"  n={n:>6}: mean_H={Hs.mean():.4f} (true {H_true}), std={Hs.std():.4f}")
