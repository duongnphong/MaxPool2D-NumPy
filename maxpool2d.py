import numpy as np

from helpers import add_padding, maxpool


def maxpool2d(
    image: np.ndarray,
    kernel_size,
    stride=None,
    padding=0,
) -> np.ndarray:
    """
    Perform a 2D MaxPool operation

    Args:
        image (np.ndarray): Input image
        kernel_size (int or tuple[int, int]): Size of the maxpooling window
        stride (int, optional): Stride value for the maxpooling operation. Default value is kernel_size.
        padding (int, optional): Padding value for the input image. Default is 0.

    Returns:
        np.ndarray: Resulting output of the maxpooling operation.


    Raises:
        TypeError: If `image` is not of type `numpy.ndarray`.
        ValueError: If `kernel_size` is invalid.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image should be of type np.ndarray.")
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, tuple) and len(kernel_size) == 1:
        kernel_size = (kernel_size[0], kernel_size[0])
    elif isinstance(kernel_size, tuple) and len(kernel_size) != 2:
        raise ValueError("Invalid kernel_size.")

    image = add_padding(image=image, padding=padding)
    out = maxpool(image=image, kernel_size=kernel_size)

    return out
