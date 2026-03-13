"""Generate BM/fBM trajectories and run Python NN predictions using ONNX Runtime
for fair comparison with C++ ONNX inference.""" # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
import numpy as np
import struct
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_fbm(n_steps, hurst, k=1.0, seed=None):
    """Generate fractional Brownian motion via Cholesky decomposition."""
    if seed is not None:
        np.random.seed(seed)

    n = n_steps - 1
    if n <= 0:
        return np.zeros(1), np.zeros(1)

    H2 = 2.0 * hurst
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            cov[i, j] = 0.5 * ((d + 1) ** H2 - 2 * d ** H2 + max(d - 1, 0) ** H2)

    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += (-eigvals.min() + 1e-10) * np.eye(n)
    L = np.linalg.cholesky(cov)

    zx = np.random.randn(n)
    zy = np.random.randn(n)

    dx = L @ zx * np.sqrt(k)
    dy = L @ zy * np.sqrt(k)

    x = np.concatenate([[0.0], np.cumsum(dx)])
    y = np.concatenate([[0.0], np.cumsum(dy)])
    return x, y


# --- Preprocessing (matching Python FreeTrace load_models.py exactly) ---

def displacement(xs, ys):
    disps = []
    for i in range(1, len(xs)):
        disps.append(np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2))
    return disps

def radius(xs, ys):
    rads = [0.]
    for i in range(1, len(xs)):
        rads.append(np.sqrt((xs[i] - xs[0])**2 + (ys[i] - ys[0])**2))
    return rads

def abs_subtraction(xs):
    result = [0.]
    for i in range(1, len(xs)):
        result.append(abs(xs[i] - xs[i-1]))
    return result

def make_alpha_inputs(xs, ys):
    rad_list = radius(xs, ys)
    disp_list = displacement(xs, ys)
    mean_disp = np.mean(disp_list)
    return (np.array(rad_list) / mean_disp / len(xs),
            (xs - xs[0]) / mean_disp / len(xs),
            (ys - ys[0]) / mean_disp / len(ys))

def cvt_2_signal(x, y):
    rad_list, xs_raw, ys_raw = make_alpha_inputs(x, y)
    x_norm = x / np.std(x)
    x_sig = np.cumsum(abs_subtraction(x_norm)) / len(x)
    y_norm = y / np.std(y)
    y_sig = np.cumsum(abs_subtraction(y_norm)) / len(y)
    return np.transpose(np.stack((x_sig, rad_list, xs_raw))), \
           np.transpose(np.stack((y_sig, rad_list, ys_raw)))

def recoupe_trajectory(x, y, model_num, jump=1):
    couped_x, couped_y = [], []
    for i in range(0, len(x), jump):
        tmp1 = x[i: i + model_num]
        tmp2 = y[i: i + model_num]
        if len(tmp1) == model_num:
            couped_x.append(tmp1)
            couped_y.append(tmp2)
    return couped_x, couped_y

def model_selection(length, crits, reg_model_nums):
    index = 0
    while True:
        if crits[index] <= length < crits[index+1]:
            return reg_model_nums[index]
        index += 1
        if index >= len(crits):
            return reg_model_nums[-2]

def log_displacements(xs, ys):
    disps = displacement(xs, ys)
    if len(xs) < 10:
        return np.log10(np.mean(disps))
    else:
        return np.log10(np.mean(np.quantile(disps, q=[0.25, 0.75], method='normal_unbiased')))


def main():
    import onnxruntime as ort

    models_dir = "models"
    reg_model_nums = [3, 5, 8]
    crits = [3, 5, 8, 8192]

    # Load ONNX models
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    alpha_sessions = {}
    for n in reg_model_nums:
        path = os.path.join(models_dir, f"reg_model_{n}.onnx")
        alpha_sessions[n] = ort.InferenceSession(path, sess_opts, providers=['CPUExecutionProvider'])

    k_session = ort.InferenceSession(
        os.path.join(models_dir, "reg_k_model.onnx"), sess_opts, providers=['CPUExecutionProvider'])

    # Test cases
    test_cases = [
        (4,   0.50, 1.0,  100, "BM_len4"),
        (6,   0.50, 1.0,  101, "BM_len6"),
        (10,  0.50, 1.0,  102, "BM_len10"),
        (20,  0.50, 1.0,  103, "BM_len20"),
        (50,  0.50, 1.0,  104, "BM_len50"),
        (100, 0.50, 1.0,  105, "BM_len100"),
        (4,   0.30, 1.0,  200, "fBM_H03_len4"),
        (10,  0.30, 1.0,  201, "fBM_H03_len10"),
        (20,  0.30, 1.0,  202, "fBM_H03_len20"),
        (50,  0.30, 1.0,  203, "fBM_H03_len50"),
        (10,  0.80, 1.0,  300, "fBM_H08_len10"),
        (20,  0.80, 1.0,  301, "fBM_H08_len20"),
        (50,  0.80, 1.0,  302, "fBM_H08_len50"),
        (20,  0.50, 0.1,  400, "BM_k01_len20"),
        (20,  0.50, 5.0,  401, "BM_k50_len20"),
        (20,  0.50, 0.01, 402, "BM_k001_len20"),
        (3,   0.50, 1.0,  500, "BM_len3_min"),
        (5,   0.50, 1.0,  501, "BM_len5_boundary"),
        (8,   0.50, 1.0,  502, "BM_len8_boundary"),
        (9,   0.50, 1.0,  503, "BM_len9"),
    ]

    out_path = "test_nn_trajectories.bin"
    results = []

    with open(out_path, "wb") as f:
        f.write(struct.pack("i", len(test_cases)))

        for n_steps, hurst, k, seed, label in test_cases:
            x, y = generate_fbm(n_steps, hurst, k, seed)
            n = len(x)

            # Alpha prediction using ONNX Runtime
            if n >= 3:
                mn = model_selection(n, crits, reg_model_nums)
                re_x, re_y = recoupe_trajectory(x, y, mn)
                input_signals = []
                for rx, ry in zip(re_x, re_y):
                    s1, s2 = cvt_2_signal(np.array(rx), np.array(ry))
                    input_signals.append(s1)
                    input_signals.append(s2)
                input_signals = np.reshape(input_signals, [-1, mn, 1, 3]).astype(np.float32)

                sess = alpha_sessions[mn]
                in_name = sess.get_inputs()[0].name
                raw_preds = sess.run(None, {in_name: input_signals})[0]
                n_raw = raw_preds.shape[0]

                if n_raw > 4:
                    pred_alpha = float(np.mean(np.quantile(raw_preds, q=[0.25, 0.75], method='normal_unbiased')))
                else:
                    pred_alpha = float(np.mean(raw_preds))
            else:
                pred_alpha = 1.0
                raw_preds = np.array([1.0])
                mn = 3
                n_raw = 1

            # K prediction using ONNX Runtime
            log_d = log_displacements(x, y)
            k_input = np.array([[log_d]], dtype=np.float32)
            k_in_name = k_session.get_inputs()[0].name
            pred_k = float(k_session.run(None, {k_in_name: k_input})[0][0])
            if np.isnan(pred_k):
                pred_k = 1.0

            results.append((label, n, hurst, k, pred_alpha, pred_k, mn, n_raw))

            # Write binary data
            label_bytes = label.encode('utf-8')
            f.write(struct.pack("i", len(label_bytes)))
            f.write(label_bytes)
            f.write(struct.pack("i", n))
            for v in x:
                f.write(struct.pack("d", float(v)))
            for v in y:
                f.write(struct.pack("d", float(v)))
            f.write(struct.pack("d", pred_alpha))
            f.write(struct.pack("d", pred_k))
            f.write(struct.pack("i", n_raw))
            for v in raw_preds.flatten():
                f.write(struct.pack("d", float(v)))

    print(f"\n{'Label':<22} {'N':>4} {'H':>4} {'k':>5} {'Model':>5} {'#Raw':>5} {'Alpha':>12} {'K':>12}")
    print("-" * 80)
    for label, n, hurst, k, alpha, pred_k, model_num, n_raw in results:
        print(f"{label:<22} {n:>4} {hurst:>4.1f} {k:>5.2f} {model_num:>5} {n_raw:>5} {alpha:>12.8f} {pred_k:>12.8f}")

    print(f"\nSaved {len(test_cases)} trajectories to {out_path}") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13


if __name__ == "__main__":
    main()
