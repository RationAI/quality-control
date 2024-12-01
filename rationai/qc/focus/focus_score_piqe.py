import numpy as np
from piqe import piqe
from skimage.color import rgb2gray
from skimage.filters import median

from rationai.qc.typing import FocusScore, GrayScaleImage, RGBImage


def _get_score_mask(
    grayscale_img: GrayScaleImage, activity_mask: GrayScaleImage
) -> GrayScaleImage:
    # TODO use standard tissue_mask library function instead
    foreground = grayscale_img <= 0.96
    foreground_area = np.sum(foreground)

    score_mask = np.ones_like(grayscale_img)
    rebalanced_score = (
        (np.sum(activity_mask * foreground) / foreground_area)
        if foreground_area > 0
        else 0
    )
    score_mask.fill(rebalanced_score)

    return score_mask


def focus_score_piqe(img: RGBImage, pixel_size: float = 0.44) -> FocusScore:
    """Creates mask of the focus score based on the PIQE algorithm.

    Args:
        img: RGB Image of the tissue, minimum size 10x10 pixels.
        pixel_size: Size of a pixel in micrometers. The score is most accurate
            at pixel sizes around 0.44 micrometers.

    Returns:
        Dictionary with the focus score mask.
            Scores range from 0 to 1, where 0 is the worst and 1 is the best.

    Note:
        The returned dictionary contains the following values:

        | Key                   | Description               |
        |-----------------------|---------------------------|
        | `focus_score_piqe`    | Mask of the focus score.  |
    """
    result: FocusScore = {}

    # Used for calculating the kernel size
    # for the median filter based on the pixel size
    optimal_pixel_size = 0.44

    kernel_size = 2 * round(1 / pixel_size * optimal_pixel_size + 1e-9) + 1
    grayscale_img = median(
        rgb2gray(img), footprint=np.ones((kernel_size, kernel_size), dtype=bool)
    )
    (
        _,  # Piqe score (not relevant for our purposes)
        _,  # Noticeable artifacts mask
        _,  # Noise mask
        activity_mask,
    ) = piqe(
        grayscale_img * 255
    )  # grayscale_img is pixel array with values between 0 and 1, piqe expects values between 0 and 255

    # Activity mask is one column shorter for images
    # with dimensions that are not divisible by 16
    if grayscale_img.shape != activity_mask.shape:
        activity_mask = np.pad(activity_mask, ((0, 0), (0, 1)), mode="edge")

    result["focus_score_piqe"] = _get_score_mask(grayscale_img, activity_mask)

    return result
