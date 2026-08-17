import numpy as np
from skimage.color import rgb2gray
from skimage.filters import median

from rationai.qc.blur.piqe import piqe
from rationai.qc.blur.utils import get_coverage_mask, simple_foreground_mask
from rationai.qc.typing import BinaryMask, BlurScore, RGBImage


def blur_score_piqe(
    img: RGBImage,
    pixel_size: float = 0.44,
    foreground_mask: BinaryMask | None = None,
) -> BlurScore:
    """Creates per-pixel and per-image coverage mask of the blur score based on the PIQE algorithm.

    Args:
        img: RGB Image of the tissue, minimum size 10x10 pixels.
        pixel_size: Size of a pixel in micrometers. The score is most accurate
            at pixel sizes around 0.44 micrometers.
        foreground_mask: Binary mask of the tissue, where 1 represents tissue and 0 represents background.
            If not provided, foreground mask is computed by `simple_foreground_mask` function.
            foreground_mask is used to nullify blur detections in the background.
            The standard tissue mask has proven to be too coarse for this use, especially
            for tiles with little or no background.

    Returns:
        Dictionary with the per-pixel and coverage mask
            blur_score_per_pixel - binary mask composed of 16x16 px blocks marking blurred areas.
            blur_score_coverage - coverage mask of the per_pixel detections. Coverage ranges from 0.0 to 1.0.


    Note:
        The returned dictionary contains the following values:

        | Key                         | Description                                           |
        |-----------------------------|-------------------------------------------------------|
        | `blur_score_per_pixel`      | Binary mask of the blur score.                        |
        | `blur_score_coverage`       | Coverage mask of the blur score.                      |
        | `number_of_examined_pixels` | Number of pixels that were evaluated by the function. |
        | `number_of_flagged_pixels`  | Number of pixels labeled as artifacts.                |

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc.blur import blur_score_piqe

    img = immunohistochemistry()
    pixel_size = 0.44  # pixel size of img in micrometers

    result = blur_score_piqe(img, pixel_size)

    # Mask of the blur coverage range from 0.0 to 1.0
    coverage_mask = result["blur_score_coverage"]

    # Binary mask of the blur detections
    blur_score_per_pixel = result["blur_score_per_pixel"]
    ```

    """
    # Used for calculating the kernel size
    # for the median filter based on the pixel size
    optimal_pixel_size = 0.44

    # Ensure kernel is at least 3x3
    kernel_size = max(2 * round(1 / pixel_size * optimal_pixel_size + 1e-9) + 1, 3)

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
        foreground_mask = simple_foreground_mask(grayscale_img)

    # Invert and restrict the mask to foreground
    activity_mask = foreground_mask * ~(activity_mask > 0)

    return {
        "blur_score_per_pixel": activity_mask,
        "blur_score_coverage": get_coverage_mask(
            grayscale_img, activity_mask, foreground_mask
        ),
        "number_of_examined_pixels": int(np.count_nonzero(foreground_mask)),
        "number_of_flagged_pixels": int(np.count_nonzero(activity_mask)),
    }
