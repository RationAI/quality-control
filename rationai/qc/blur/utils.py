from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import area_opening, erosion

from rationai.qc.typing import BinaryMask, FloatingPointImage


def masked_average_pooling(
    arr: FloatingPointImage, foreground_mask: BinaryMask
) -> FloatingPointImage:
    """Perform masked average pooling, creating mask of 16x16 blocks.

    For images with dimensions not divisible by 16, the function pads the image
    to the nearest multiple of 16. It then reshapes the image into 16x16 blocks,
    computes the average of the foreground pixels in each block, and returns
    the pooled image with the same dimensions as the original image.

    Args:
        arr: Input array to be pooled.
        foreground_mask: Binary mask indicating the foreground pixels.
            1 for foreground, 0 for background.

    Returns:
        output: Pooled array with the same shape as the input.

    """
    h, w = arr.shape

    # Calculate padding needed
    h_pad = (16 - (h % 16)) % 16
    w_pad = (16 - (w % 16)) % 16

    # Pad the input array and the mask
    padded_arr = np.pad(arr, ((0, h_pad), (0, w_pad)), mode="constant")
    padded_mask = np.pad(foreground_mask, ((0, h_pad), (0, w_pad)), mode="constant")

    padded_h, padded_w = padded_arr.shape

    # Reshape into 16x16 blocks
    arr_blocks = padded_arr.reshape(padded_h // 16, 16, padded_w // 16, 16)
    mask_blocks = padded_mask.reshape(padded_h // 16, 16, padded_w // 16, 16)

    # Calculate the sum of foreground values and the number of valid pixels in the block
    sum_values = np.sum(arr_blocks * mask_blocks, axis=(1, 3))
    count_values = np.sum(mask_blocks, axis=(1, 3))

    # Avoid division by zero: where count_values==0, set the average to 0
    with np.errstate(divide="ignore", invalid="ignore"):
        block_means = np.where(count_values > 0, sum_values / count_values, 0)

    # Expand back to the padded size
    padded_output = np.repeat(np.repeat(block_means, 16, axis=0), 16, axis=1)

    # Remove padding to get the output with the original dimensions
    output = padded_output[:h, :w]

    return output


def get_coverage_mask(
    tissue_img: NDArray[Any],
    detection_mask: BinaryMask,
    foreground_mask: BinaryMask,
) -> FloatingPointImage:
    """Creates a blur coverage mask based on the tissue image and binary mask of blur detections.

    Args:
        tissue_img: Image of the tissue.
        detection_mask: Binary image of the blur detection.
        foreground_mask: Binary mask of the tissue, where 1 represents tissue and 0 represents background.

    Returns:
        Blur coverage mask. Coverage ranges from 0.0 to 1.0.

    """
    foreground_area = np.count_nonzero(foreground_mask)

    coverage_mask = np.ones_like(tissue_img).astype(np.float64)
    coverage = (
        (np.sum(detection_mask * foreground_mask) / foreground_area)
        if foreground_area > 0
        else 0
    )
    coverage_mask.fill(coverage)

    return coverage_mask


def simple_foreground_mask(
    grayscale_img: FloatingPointImage,
    threshold: float = 0.92,  # ~235/255
) -> BinaryMask:
    """Creates a simple foreground mask using thresholding and morphological operations.

    Args:
        grayscale_img: Grayscale image of the tissue, values between 0.0 and 1.0.
        threshold: Threshold value for binarization. Default is 0.92.
            Values below this threshold are considered foreground (tissue).

    Returns:
        Binary mask of the tissue, where 1 represents tissue and 0 represents background.

    """
    foreground_mask = grayscale_img < threshold

    foreground_mask = area_opening(foreground_mask)
    foreground_mask = erosion(foreground_mask, footprint=np.ones((5, 5)))

    return foreground_mask
