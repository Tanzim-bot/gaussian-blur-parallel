import os
from PIL import Image
import numpy as np
from time import perf_counter
from utils import gaussian_kernel


def gaussian_blur_sequential(image_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a Gaussian blur using a straightforward sequential (single-threaded)
    nested-loop convolution.

    Parameters
    ----------
    image_array : numpy.ndarray
        Grayscale input image.
    kernel : numpy.ndarray
        Precomputed Gaussian kernel.

    Returns
    -------
    output : numpy.ndarray (uint8)
        Blurred image.
    """

    # Kernel radius (e.g., size 11 → k = 5)
    k = kernel.shape[0] // 2

    # Pad the image so convolution is valid at borders
    padded = np.pad(image_array, pad_width=k, mode='edge')

    h, w = image_array.shape
    output = np.zeros_like(image_array, dtype=float)

    # --- Main convolution loop (slow but simple) ---
    for i in range(h):
        for j in range(w):
            # Extract local neighbourhood around pixel (i, j)
            region = padded[i:i + 2*k + 1, j:j + 2*k + 1]

            # Multiply kernel × region and sum for blurred pixel
            output[i, j] = np.sum(region * kernel)

    # Ensure pixel values are valid
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def main():
    """
    Runs the sequential blur as a standalone script.

    - Loads input image from /input/
    - Applies sequential Gaussian blur
    - Saves result into /output/images/
    - Prints execution time
    """

    # ----------------------------
    # Paths & folder management
    # ----------------------------
    input_path = os.path.join("input", "input_large.jpg")

    output_images_dir = os.path.join("output", "images")
    os.makedirs(output_images_dir, exist_ok=True)

    # ----------------------------
    # Load input image
    # ----------------------------
    img = Image.open(input_path).convert("L")     # Convert to grayscale
    img_array = np.array(img)

    # ----------------------------
    # Build Gaussian kernel
    # ----------------------------
    kernel = gaussian_kernel(size=11, sigma=2.0)

    # ----------------------------
    # Time the sequential blur
    # ----------------------------
    start = perf_counter()
    blurred_array = gaussian_blur_sequential(img_array, kernel)
    end = perf_counter()

    print(f"Sequential blur time: {end - start:.4f} seconds")

    # ----------------------------
    # Save output image
    # ----------------------------
    blurred_img = Image.fromarray(blurred_array)
    output_path = os.path.join(output_images_dir, "output_sequential.jpg")
    blurred_img.save(output_path)

    print(f"Saved sequential blurred image to: {output_path}")


if __name__ == "__main__":
    main()