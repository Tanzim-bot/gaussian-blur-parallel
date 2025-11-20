import os
import numpy as np
from multiprocessing import Pool, cpu_count
from PIL import Image
from time import perf_counter
from utils import gaussian_kernel


def blur_chunk(args):
    """
    Worker function run in each process.

    It applies the Gaussian blur to a *chunk* (a range of rows) of the padded image.

    Parameters
    ----------
    args : tuple
        (padded, kernel, start_row, end_row)
        padded    : 2D numpy array of the padded input image
        kernel    : 2D numpy array (Gaussian kernel)
        start_row : starting row index in the original (unpadded) image
        end_row   : ending row index (non-inclusive) in the original image

    Returns
    -------
    chunk_output : 2D numpy array
        Blurred rows from start_row to end_row (no padding).
    """
    padded, kernel, start_row, end_row = args

    # Half the kernel size (radius)
    k = kernel.shape[0] // 2

    # Number of rows in this chunk and width of the original image
    h = end_row - start_row
    w = padded.shape[1] - 2 * k

    # Allocate an array to store the blurred chunk
    chunk_output = np.zeros((h, w), dtype=float)

    # Convolve kernel over the assigned rows
    for i in range(h):
        for j in range(w):
            # Extract the local neighbourhood in the padded image
            region = padded[start_row + i : start_row + i + 2 * k + 1,
                            j : j + 2 * k + 1]
            # Element-wise multiply with kernel and sum to get blurred pixel
            chunk_output[i, j] = np.sum(region * kernel)

    return chunk_output


def gaussian_blur_parallel(image_array: np.ndarray,
                           kernel: np.ndarray,
                           num_procs: int) -> np.ndarray:
    """
    Apply Gaussian blur to an image using multiprocessing.

    The image is split into horizontal strips, each processed by a separate process.

    Parameters
    ----------
    image_array : 2D numpy array
        Grayscale input image.
    kernel : 2D numpy array
        Gaussian kernel.
    num_procs : int
        Number of worker processes to use.

    Returns
    -------
    output : 2D numpy array (uint8)
        The blurred image.
    """
    # Radius of the kernel
    k = kernel.shape[0] // 2

    # Pad the image so that convolution is valid at the borders
    padded = np.pad(image_array, pad_width=k, mode="edge")
    h, w = image_array.shape

    # --- Split the image rows between processes as evenly as possible ---

    # Base number of rows per chunk
    base_chunk = h // num_procs
    # Some chunks will get one extra row if the division is not exact
    remainder = h % num_procs

    starts = []
    ends = []
    start = 0

    for i in range(num_procs):
        # Distribute the remainder rows one-by-one to the first 'remainder' chunks
        extra = 1 if i < remainder else 0
        end = start + base_chunk + extra
        starts.append(start)
        ends.append(end)
        start = end

    # Build argument list for each worker:
    # every worker receives the padded image, kernel, and its row range
    args_list = [(padded, kernel, s, e) for s, e in zip(starts, ends)]

    # --- Run the blur in parallel using a process pool ---
    with Pool(processes=num_procs) as pool:
        # Each worker returns its blurred chunk
        results = pool.map(blur_chunk, args_list)

    # Stack all chunks vertically to get the full image back
    output = np.vstack(results)

    # Clip values to valid [0, 255] range and convert to uint8 for image saving
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def main():
    """
    Entry point for running the parallel blur as a standalone script.

    - Loads input image from input/input_large.jpg
    - Applies Gaussian blur in parallel
    - Saves blurred image to output/images/
    - Prints execution time
    """
    # ---- Define paths ----
    input_path = os.path.join("input", "input_large.jpg")

    # Put all output images in a dedicated folder to keep the project tidy
    output_images_dir = os.path.join("output", "images")
    os.makedirs(output_images_dir, exist_ok=True)

    # ---- Load image and convert to grayscale ----
    img = Image.open(input_path).convert("L")
    img_array = np.array(img)

    # ---- Create Gaussian kernel ----
    # You can experiment with size and sigma.
    kernel = gaussian_kernel(size=11, sigma=2.0)

    # ---- Choose number of processes ----
    # Uses all available logical CPUs by default.
    # You can change this to a fixed value, e.g. num_procs = 8
    num_procs = cpu_count()

    # ---- Time the parallel blur ----
    start = perf_counter()
    blurred_array = gaussian_blur_parallel(img_array, kernel, num_procs)
    end = perf_counter()

    print(f"Parallel blur time ({num_procs} processes): {end - start:.4f} seconds")

    # ---- Save result image ----
    blurred_img = Image.fromarray(blurred_array)
    output_path = os.path.join(output_images_dir, f"output_parallel_{num_procs}procs.jpg")
    blurred_img.save(output_path)
    print(f"Saved blurred image to: {output_path}")


if __name__ == "__main__":
    # This guard is REQUIRED for multiprocessing on Windows.
    main()