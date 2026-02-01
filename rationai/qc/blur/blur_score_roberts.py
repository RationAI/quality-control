import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, roberts

from rationai.qc.blur.utils import (
    get_coverage_mask,
    masked_average_pooling,
    simple_foreground_mask,
)
from rationai.qc.typing import BinaryMask, BlurScore, RGBImage


def blur_score_roberts(
    img: RGBImage,
    threshold: float = 2,
    foreground_mask: BinaryMask | None = None,
) -> BlurScore:
    """Creates per-pixel and per-image coverage mask of the blur score.

    Blur detection is based on the absolute difference between roberts gradient image and roberts gradient image of the gaussian blurred image.

    Args:
        img: RGB Image of the tissue, minimum size 10x10 pixels.
        threshold: Used to classify 16x16 pixel blocks of the output mask.
            Each block consist of mean value of absolute difference between roberts gradient image and roberts gradient image of the gaussian blurred image.
            If the mean value of a block falls below this threshold, the block is classified as blurred.
            threshold can range from 0 to 40+.
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

    from rationai.qc.blur import blur_score_roberts

    img = immunohistochemistry()

    result = blur_score_roberts(img)

    # Mask of the blur coverage range from 0.0 to 1.0
    coverage_mask = result["blur_score_coverage"]

    # Binary mask of the blur detections
    blur_score_per_pixel = result["blur_score_per_pixel"]
    ```
    """
    grayscale_img = rgb2gray(img)

    if foreground_mask is None:
        foreground_mask = simple_foreground_mask(grayscale_img)

    grayscale_img *= 255
    gradient = roberts(grayscale_img)
    gaussian_gradient = roberts(gaussian(grayscale_img, sigma=1))

    blur_score = np.abs(gradient - gaussian_gradient)
    blur_score_pooled = masked_average_pooling(
        arr=blur_score, foreground_mask=foreground_mask
    )

    # Threshold was set to 2 based on empirical testing
    # Can be adjusted based on the desired sensitivity
    # Higher threshold means more pixels are considered blurred
    blur_score_per_pixel = blur_score_pooled < threshold

    # activity_mask is multiplied by the foreground mask to nullify background pixels
    blur_score_per_pixel *= foreground_mask

    return {
        "blur_score_per_pixel": blur_score_per_pixel,
        "blur_score_coverage": get_coverage_mask(
            blur_score_per_pixel,
            detection_mask=blur_score_per_pixel,
            foreground_mask=foreground_mask,
        ),
        "number_of_examined_pixels": int(np.count_nonzero(foreground_mask)),
        "number_of_flagged_pixels": int(np.count_nonzero(blur_score_per_pixel)),
    }
