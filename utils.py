import numpy as np

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a 2D Gaussian kernel."""
    k = size // 2
    x = np.arange(-k, k+1)
    y = np.arange(-k, k+1)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel