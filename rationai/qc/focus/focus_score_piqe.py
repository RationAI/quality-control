import numpy as np
from piqe import piqe
from skimage.color import rgb2gray
from skimage.filters import median

from rationai.qc.typing import QcValues, RGBImage


def _get_score_mask(gray, activity_mask):
    # TODO use standard tissue_mask library function instead
    foreground = gray <= 0.96
    foreground_pixels = np.sum(foreground)

    result = np.ones_like(gray)
    rebalanced_score = (
        (np.sum(activity_mask * foreground) / foreground_pixels)
        if foreground_pixels > 0
        else 0
    )
    result.fill(rebalanced_score)

    return result


def focus_score_piqe(img: RGBImage, pixel_size: float = 0.44) -> QcValues:
    """Creates mask of the focus score based on the PIQE algorithm.

    Args:
        img: RGB Image of the tissue, minimum size 10x10 pixels.
        pixel_size: Size of a pixel in micrometers. The score is most accurate
            at pixel sizes around 0.44 micrometers, which is equivalent to
             a level 1 downsample of all slides this function has been tested on.

    Returns:
        Dictionary with the focus score mask.
        Scores range from 0 to 1, where 0 is the worst and 1 is the best.

        **Dictionary values**
        * `focus_score_piqe`: Mask of the focus score.
    """
    print("focus_score_piqe")
    result: QcValues = {}

    kernel_size = 2 * round(1 / pixel_size * 0.44 + 1e-9) + 1
    gray = median(
        rgb2gray(img), footprint=np.ones((kernel_size, kernel_size), dtype=bool)
    )
    (
        _,  # Piqe score (not relevant for our purposes)
        _,  # Noticeable artifacts mask
        _,  # Noise mask
        activity_mask,
    ) = piqe(
        gray * 255
    )  # gray is pixel array with values between 0 and 1, piqe expects values between 0 and 255

    # Activity mask is one column shorter for images
    # with dimensions that are not divisible by 16
    if gray.shape != activity_mask.shape:
        activity_mask = np.pad(activity_mask, ((0, 0), (0, 1)), mode="edge")

    result["focus_score_piqe"] = _get_score_mask(gray, activity_mask)

    return result
