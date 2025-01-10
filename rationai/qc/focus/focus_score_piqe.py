import numpy as np
import pyvips
from skimage.color import rgb2gray
from skimage.filters import median

from rationai.masks import tissue_mask
from rationai.qc.focus.piqe import piqe
from rationai.qc.typing import FocusScore, GrayScaleImage, NDArray, RGBImage


def _get_score_mask(
    grayscale_img: GrayScaleImage,
    activity_mask: GrayScaleImage,
    tissue_mask: NDArray[bool],
) -> GrayScaleImage:
    """Creates a focus score mask based on the grayscale image and the piqe activity mask.

    Args:
        grayscale_img: Grayscale image of the tissue.
        activity_mask: Grayscale image of piqe activity_mask.
        tissue_mask: Binary mask of the tissue, where 1 represents tissue and 0 represents background.

    Returns:
        Focus score mask. Scores range from 0 to 1,
        where 0 is the worst and 1 is the best.

    """
    foreground_area = np.count_nonzero(tissue_mask)

    score_mask = np.ones_like(grayscale_img)
    rebalanced_score = (
        (np.sum(activity_mask * tissue_mask) / foreground_area)
        if foreground_area > 0
        else 0
    )
    score_mask.fill(rebalanced_score)

    return score_mask


def focus_score_piqe(
    img: RGBImage,
    pixel_size: float = 0.44,
    foreground_mask: NDArray[bool] | None = None,
) -> FocusScore:
    """Creates mask of the focus score based on the PIQE algorithm.

    Args:
        img: RGB Image of the tissue, minimum size 10x10 pixels.
        pixel_size: Size of a pixel in micrometers. The score is most accurate
            at pixel sizes around 0.44 micrometers.
        foreground_mask: Binary mask of the tissue, where 1 represents tissue and 0 represents background.
            If not provided, the mask is generated using the standard tissue_mask library function.

    Returns:
        Dictionary with the focus score mask.
            Scores range from 0 to 1, where 0 is the worst and 1 is the best.

    Note:
        The returned dictionary contains the following values:

        | Key                   | Description               |
        |-----------------------|---------------------------|
        | `focus_score_piqe`    | Mask of the focus score.  |

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc.focus import focus_score_piqe

    img = immunohistochemistry()
    pixel_size = 0.44  # pixel size of img in micrometers

    result = focus_score_piqe(img, pixel_size)

    score_mask = result["focus_score_piqe"]  # Mask of the focus score range from 0 to 1
    ```

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

    if foreground_mask is None:
        foreground_mask = tissue_mask(
            pyvips.Image.new_from_array(img), mpp=pixel_size
        ).numpy()
        foreground_mask = (foreground_mask > 0).astype(int)

    result["focus_score_piqe"] = _get_score_mask(
        grayscale_img, activity_mask, foreground_mask
    )

    return result
