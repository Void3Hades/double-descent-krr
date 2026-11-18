"""
Double Descent in Kernel Ridge Regression (KRR) - 20-Seed Experiment

A reproducible synthetic experiment using NumPy and Matplotlib.
Kernel: K(x_i, x_j) = x_i.T @ x_j
Dual KRR solution: alpha = solve(K + lambda * I, y)
"""
import argparse
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore",category=RuntimeWarning)

def generate_dataset(n_train, n_test, max_features, signal_features, noise_std, seed):
    """Generate Gaussian inputs and targets from one fixed underlying signal."""
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, max_features))
    X_test = rng.normal(size=(n_test, max_features))

    beta = np.zeros(max_features)
    beta[:signal_features] = rng.normal(size=signal_features)
    beta[:signal_features] /= np.linalg.norm(beta[:signal_features])

    y_train = X_train @ beta + noise_std * rng.normal(size=n_train)
    y_test = X_test @ beta + noise_std * rng.normal(size=n_test)
    return X_train, y_train, X_test, y_test


def kernel_ridge_predict(X_train, y_train, X_new, regularization):
    """Fit linear-kernel KRR and predict at X_new."""
    K = X_train @ X_train.T
    alpha = np.linalg.solve(K + regularization * np.eye(len(K)), y_train)
    K_new = X_new @ X_train.T
    return K_new @ alpha, alpha


def run_single_seed(n, n_test, signal_features, noise_std, regularization, seed):
    max_features = 2 * n
    Xtr_all, ytr, Xte_all, yte = generate_dataset(
        n, n_test, max_features, signal_features, noise_std, seed
    )

    below = np.linspace(8, n - 20, 22, dtype=int)
    near = np.arange(n - 18, n + 19, 2)
    above = np.linspace(n + 20, 2 * n, 22, dtype=int)
    ps = np.unique(np.concatenate((below, near, above)))
    ps = ps[(ps >= 1) & (ps <= max_features)]

    train_mse, test_mse = [], []
    for p in ps:
        Xtr, Xte = Xtr_all[:, :p], Xte_all[:, :p]
        pred_train, _ = kernel_ridge_predict(Xtr, ytr, Xtr, regularization)
        pred_test, _ = kernel_ridge_predict(Xtr, ytr, Xte, regularization)
        train_mse.append(np.mean((ytr - pred_train) ** 2))
        test_mse.append(np.mean((yte - pred_test) ** 2))

    return ps, np.array(train_mse), np.array(test_mse)


def run_experiment(n=140, n_test=1200, signal_features=15,
                   noise_std=0.30, regularization=1e-3, seed=42, num_seeds=20):
    if n < 40:
        raise ValueError("Use n >= 40 for a useful sweep.")
    if not 1 <= signal_features < n:
        raise ValueError("signal_features must be between 1 and n-1.")
    if regularization <= 0:
        raise ValueError("regularization must be positive.")

    seeds = [seed + i for i in range(num_seeds)]
    train_mse_list, test_mse_list = [], []
    ps = None

    for s in seeds:
        ps_s, tr_mse, te_mse = run_single_seed(
            n, n_test, signal_features, noise_std, regularization, s
        )
        if ps is None:
            ps = ps_s
        train_mse_list.append(tr_mse)
        test_mse_list.append(te_mse)

    train_mse_arr = np.array(train_mse_list)
    test_mse_arr = np.array(test_mse_list)

    train_mean = np.mean(train_mse_arr, axis=0)
    train_std = np.std(train_mse_arr, axis=0)
    test_mean = np.mean(test_mse_arr, axis=0)
    test_std = np.std(test_mse_arr, axis=0)

    settings = {
        "training_samples": n,
        "test_samples": n_test,
        "signal_features": signal_features,
        "noise_standard_deviation": noise_std,
        "ridge_regularization_lambda": regularization,
        "base_seed": seed,
        "num_seeds": num_seeds,
        "seeds": seeds,
        "kernel": "linear: K(x_i, x_j) = x_i.T @ x_j",
    }
    return ps, train_mean, train_std, test_mean, test_std, settings


def save_outputs(ps, train_mean, train_std, test_mean, test_std, settings, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        out / "results.csv",
        np.column_stack((ps, train_mean, train_std, test_mean, test_std)),
        delimiter=",",
        header="features_p,training_mse_mean,training_mse_std,test_mse_mean,test_mse_std",
        comments="",
        fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g"],
    )
    (out / "experiment_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    n = settings["training_samples"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    # Training error plot with variability band
    axes[0].plot(ps, train_mean, marker="o", markersize=3, label="Mean Training MSE")
    axes[0].fill_between(ps, train_mean - train_std, train_mean + train_std, alpha=0.3, label="±1 Std Dev")
    axes[0].axvline(n, linestyle="--", color="tab:blue", label="p = n")
    axes[0].set(title="Training error (20 Seeds)", xlabel="Feature dimension (p)",
                ylabel="Mean squared error")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Test error plot with variability band
    axes[1].plot(ps, test_mean, marker="o", markersize=3, color="tab:orange", label="Mean Test MSE")
    axes[1].fill_between(ps, test_mean - test_std, test_mean + test_std, color="tab:orange", alpha=0.3, label="±1 Std Dev")
    axes[1].axvline(n, linestyle="--", color="tab:blue", label="p = n")
    axes[1].set(title="Test error: double descent (20 Seeds)",
                xlabel="Feature dimension (p)", ylabel="Mean squared error")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Double Descent with Linear KRR ({settings['num_seeds']} Seeds Average)")
    fig.tight_layout()
    fig.savefig(out / "double_descent_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ps, train_mean, marker="o", markersize=3, label="Training MSE (Mean)")
    ax.fill_between(ps, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(ps, test_mean, marker="o", markersize=3, label="Test MSE (Mean)")
    ax.fill_between(ps, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.axvline(n, linestyle="--", label="Interpolation threshold (p = n)")
    ax.set_yscale("log")
    ax.set(xlabel="Feature dimension (p)", ylabel="Mean squared error (log scale)",
           title=f"Training and test error ({settings['num_seeds']} Seeds Average)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_vs_test_error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run the 20-Seed Double Descent KRR experiment.")
    parser.add_argument("--n", type=int, default=140)
    parser.add_argument("--n-test", type=int, default=1200)
    parser.add_argument("--signal-features", type=int, default=15)
    parser.add_argument("--noise-std", type=float, default=0.30)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=20, help="Number of random seeds to average over.")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    ps, train_mean, train_std, test_mean, test_std, settings = run_experiment(
        n=args.n, n_test=args.n_test, signal_features=args.signal_features,
        noise_std=args.noise_std, regularization=args.regularization, seed=args.seed,
        num_seeds=args.num_seeds
    )
    save_outputs(ps, train_mean, train_std, test_mean, test_std, settings, args.output)
    peak = int(np.argmax(test_mean))
    print("Multi-seed experiment completed successfully.")
    print(f"Training sample size n = {args.n}")
    print(f"Averaged over {args.num_seeds} seeds (base seed = {args.seed})")
    print(f"Maximum mean test MSE = {test_mean[peak]:.6f} (± {test_std[peak]:.6f}) at p = {ps[peak]}")
    print(f"Files saved in: {args.output}")


if __name__ == "__main__":
    main()