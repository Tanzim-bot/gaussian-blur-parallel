import os
import pandas as pd
import matplotlib.pyplot as plt


def ensure_plots_dir() -> str:
    """
    Ensure that the output/plots directory exists.

    Returns
    -------
    str
        Path to the plots directory.
    """
    plots_dir = os.path.join("output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_execution_time(df: pd.DataFrame, plots_dir: str) -> None:
    """
    Plot execution time vs number of processes for each image size.

    Two lines are shown per graph:
      - Sequential execution time
      - Parallel execution time

    Parameters
    ----------
    df : pandas.DataFrame
        Data loaded from results.csv.
    plots_dir : str
        Directory where plots will be saved.
    """
    sizes = df["size"].unique()

    for size in sizes:
        subset = df[df["size"] == size]

        plt.figure()
        plt.plot(subset["num_procs"], subset["sequential_time"],
                 marker="o", label="Sequential Time")
        plt.plot(subset["num_procs"], subset["parallel_time"],
                 marker="o", label="Parallel Time")

        plt.title(f"Execution Time vs Processes (Image size: {size}x{size})")
        plt.xlabel("Number of Processes")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.legend()

        out_path = os.path.join(plots_dir, f"execution_time_{size}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def plot_speedup(df: pd.DataFrame, plots_dir: str) -> None:
    """
    Plot speedup vs number of processes for each image size.

    Speedup is defined as T_seq / T_par.

    Parameters
    ----------
    df : pandas.DataFrame
        Data loaded from results.csv.
    plots_dir : str
        Directory where plots will be saved.
    """
    sizes = df["size"].unique()

    for size in sizes:
        subset = df[df["size"] == size]

        plt.figure()
        plt.plot(subset["num_procs"], subset["speedup"], marker="o")

        plt.title(f"Speedup vs Processes (Image size: {size}x{size})")
        plt.xlabel("Number of Processes")
        plt.ylabel("Speedup (T_seq / T_par)")
        plt.grid(True)

        out_path = os.path.join(plots_dir, f"speedup_{size}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def plot_efficiency(df: pd.DataFrame, plots_dir: str) -> None:
    """
    Plot efficiency vs number of processes for each image size.

    Efficiency is defined as speedup / number of processes.

    Parameters
    ----------
    df : pandas.DataFrame
        Data loaded from results.csv.
    plots_dir : str
        Directory where plots will be saved.
    """
    sizes = df["size"].unique()

    for size in sizes:
        subset = df[df["size"] == size]

        plt.figure()
        plt.plot(subset["num_procs"], subset["efficiency"], marker="o")

        plt.title(f"Efficiency vs Processes (Image size: {size}x{size})")
        plt.xlabel("Number of Processes")
        plt.ylabel("Efficiency (speedup / processes)")
        plt.grid(True)

        out_path = os.path.join(plots_dir, f"efficiency_{size}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def main() -> None:
    """
    Main entry point.

    - Loads CSV results from output/results/results.csv
    - Generates:
        * Execution time vs processes plots
        * Speedup vs processes plots
        * Efficiency vs processes plots
    - Saves all plots into output/plots/
    """
    # Path to results CSV (created by performance_test.py)
    results_path = os.path.join("output", "results", "results.csv")
    print(f"Loading results from {results_path}...")

    df = pd.read_csv(results_path)

    # Ensure plots directory exists
    plots_dir = ensure_plots_dir()

    print("Generating Execution Time plots...")
    plot_execution_time(df, plots_dir)

    print("Generating Speedup plots...")
    plot_speedup(df, plots_dir)

    print("Generating Efficiency plots...")
    plot_efficiency(df, plots_dir)

    print("All graphs generated successfully!")


if __name__ == "__main__":
    main()