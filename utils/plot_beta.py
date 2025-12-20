import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - logB)


def beta_mean(a: float, b: float) -> float:
    return a / (a + b) if (a + b) > 0 else float("nan")


def parse_beta_params_cell(cell: str) -> np.ndarray:
    """
    cell is a string like: "2.0 2.0 2.0 ...", length should be 48 floats:
      alpha_d(12) beta_d(12) alpha_t(12) beta_t(12)
    """
    if not isinstance(cell, str) or not cell.strip():
        raise ValueError("beta_params cell is empty or not a string")

    parts = cell.strip().split()
    arr = np.array([float(p) for p in parts], dtype=np.float64)

    if arr.size != 48:
        raise ValueError(f"Expected 48 beta params, got {arr.size}")

    return arr


def extract_state_ab(arr48: np.ndarray, state: int, which: str) -> tuple[float, float]:
    """
    which: 'duty' or 'duration'
    Returns (alpha, beta) for that state.
    Flatten order assumed:
      [alpha_d 12, beta_d 12, alpha_t 12, beta_t 12]
    """
    if which not in ("duty", "duration"):
        raise ValueError("which must be 'duty' or 'duration'")
    if not (0 <= state <= 11):
        raise ValueError("state must be between 0 and 11")

    if which == "duty":
        a = float(arr48[state])         # alpha_d
        b = float(arr48[12 + state])    # beta_d
    else:
        a = float(arr48[24 + state])    # alpha_t
        b = float(arr48[36 + state])    # beta_t
    return a, b


def plot_beta_evolution(history_csv: str, state: int, which: str, *, snapshots: int = 6) -> None:
    """
    Load history CSV and plot Beta distribution evolution over time for:
      - a chosen state (0..11)
      - 'duty' or 'duration'

    Produces:
      1) PDF snapshots across training (u in 0..1)
      2) alpha/beta/mean vs time (row index)
    """
    history_path = Path(history_csv)
    if not history_path.exists():
        raise FileNotFoundError(f"Could not find: {history_path}")

    df = pd.read_csv(history_path)

    if "beta_params" not in df.columns:
        raise ValueError(
            "CSV missing 'beta_params' column. "
            "Make sure you're pointing at the continuous run history.csv."
        )

    # drop rows with missing beta params
    df = df.dropna(subset=["beta_params"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows with beta_params found")

    # Parse all beta_params into array [N, 48]
    params = np.vstack([parse_beta_params_cell(s) for s in df["beta_params"].astype(str)])

    # Extract alpha/beta per row for this state
    alphas = np.zeros(len(df), dtype=np.float64)
    betas = np.zeros(len(df), dtype=np.float64)
    means = np.zeros(len(df), dtype=np.float64)

    for i in range(len(df)):
        a, b = extract_state_ab(params[i], state, which)
        alphas[i] = a
        betas[i] = b
        means[i] = beta_mean(a, b)

    # Choose snapshot indices evenly spaced
    snapshots = max(2, int(snapshots))
    idxs = np.linspace(0, len(df) - 1, snapshots).astype(int)
    idxs = np.unique(idxs)

    # ---- Plot 1: PDF snapshots (latent u in 0..1) ----
    x = np.linspace(1e-4, 1 - 1e-4, 800)

    plt.figure()
    for i in idxs:
        a = alphas[i]
        b = betas[i]
        y = beta_pdf(x, a, b)
        plt.plot(x, y, label=f"row {i} (a={a:.2f}, b={b:.2f})")

    plt.title(f"Beta PDF evolution: {which} (state={state})")
    plt.xlabel("u (0..1)")
    plt.ylabel("pdf")
    plt.legend()

    # ---- Plot 2: alpha/beta over time ----
    plt.figure()
    plt.plot(alphas, label="alpha")
    plt.plot(betas, label="beta")
    plt.title(f"Beta parameters over time: {which} (state={state})")
    plt.xlabel("row index")
    plt.ylabel("value")
    plt.legend()

    # ---- Plot 3: mean over time ----
    plt.figure()
    plt.plot(means)
    plt.title(f"Beta mean over time: {which} (state={state})")
    plt.xlabel("row index")
    plt.ylabel("mean (alpha/(alpha+beta))")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/runs/continuous/history.csv",
        help="Path to continuous run history.csv",
    )
    parser.add_argument("--state", type=int, default=8, help="State index 0..11")
    parser.add_argument(
        "--which",
        choices=["duty", "duration"],
        default="duty",
        help="Whether to plot duty or duration Beta evolution",
    )
    parser.add_argument("--snapshots", type=int, default=6, help="How many PDF snapshots to plot")
    args = parser.parse_args()

    plot_beta_evolution(args.csv, args.state, args.which, snapshots=args.snapshots)


if __name__ == "__main__":
    main()
