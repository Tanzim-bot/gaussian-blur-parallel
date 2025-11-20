import os
from time import perf_counter
from typing import Callable

import numpy as np
from PIL import Image
from multiprocessing import cpu_count
import csv

from utils import gaussian_kernel
from blur_sequential import gaussian_blur_sequential
from blur_parallel import gaussian_blur_parallel


def time_function(func: Callable, *args, repeats: int = 3) -> float:
    """
    Time a function by running it several times and returning
    the average execution time in seconds.

    Parameters
    ----------
    func : Callable
        The function to benchmark.
    *args :
        Positional arguments passed to func.
    repeats : int, optional
        Number of runs to average over (default = 3).

    Returns
    -------
    float
        Average execution time in seconds.
    """
    times = []
    for _ in range(repeats):
        start = perf_counter()
        func(*args)
        end = perf_counter()
        times.append(end - start)
    return sum(times) / len(times)


def resize_image(img_array: np.ndarray, size: int) -> np.ndarray:
    """
    Resize a grayscale image (numpy array) to size x size pixels.

    Parameters
    ----------
    img_array : numpy.ndarray
        Original image array.
    size : int
        Target width and height.

    Returns
    -------
    numpy.ndarray
        Resized image array.
    """
    img = Image.fromarray(img_array)
    img_resized = img.resize((size, size))
    return np.array(img_resized)


def main():
    """
    Benchmark sequential vs parallel Gaussian blur.

    For each chosen image size and number of processes:
      - Run the sequential blur and record the average time.
      - Run the parallel blur and record the average time.
      - Compute speedup and efficiency.
      - Write results to a CSV file in output/results/results.csv.

    This script is used to generate data for performance graphs
    in the report.
    """

    # ----------------------------
    # Input image setup
    # ----------------------------
    input_path = os.path.join("input", "input_large.jpg")
    base_img = Image.open(input_path).convert("L")
    base_array = np.array(base_img)

    # Precompute Gaussian kernel used for all experiments
    kernel = gaussian_kernel(size=11, sigma=2.0)

    # ----------------------------
    # Experiment parameters
    # ----------------------------

    # Image sizes to test (square images: size x size)
    sizes = [512, 1024, 2048]

    # Candidate process counts (do not exceed logical CPU count)
    max_procs = cpu_count()
    candidate_procs = [1, 2, 4, 8]
    process_counts = [p for p in candidate_procs if p <= max_procs]

    # ----------------------------
    # Results file and directory
    # ----------------------------
    results_dir = os.path.join("output", "results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "results.csv")

    # Open CSV file and write header
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "size",
            "num_procs",
            "sequential_time",
            "parallel_time",
            "speedup",
            "efficiency",
        ])

        # ----------------------------
        # Main benchmarking loop
        # ----------------------------
        for size in sizes:
            print(f"\n=== Testing size {size}x{size} ===")

            # Resize base image to the desired size
            img_resized = resize_image(base_array, size)

            # Time sequential blur once per size
            seq_time = time_function(gaussian_blur_sequential, img_resized, kernel)
            print(f"Sequential time: {seq_time:.4f} s")

            # Time parallel blur for each process count
            for p in process_counts:
                par_time = time_function(gaussian_blur_parallel, img_resized, kernel, p)

                # Calculate speedup and efficiency
                speedup = seq_time / par_time if par_time > 0 else 0.0
                efficiency = speedup / p if p > 0 else 0.0

                print(
                    f"  p={p}: parallel={par_time:.4f} s, "
                    f"speedup={speedup:.2f}, efficiency={efficiency:.2f}"
                )

                # Write one row per experiment
                writer.writerow([size, p, seq_time, par_time, speedup, efficiency])

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()