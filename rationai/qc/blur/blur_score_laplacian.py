import numpy as np
from rationai.staining import StandardConversions, convert_color
from skimage.color import rgb2gray
from skimage.filters import laplace, threshold_otsu
from skimage.morphology.binary import binary_dilation, binary_erosion

from rationai.qc.blur.utils import (
    get_coverage_mask,
    masked_average_pooling,
    simple_foreground_mask,
)
from rationai.qc.typing import BinaryMask, BlurScore, RGBImage


def blur_score_laplacian(
    img: RGBImage,
    threshold: float = 5,
    foreground_mask: BinaryMask | None = None,
) -> BlurScore:
    """Creates per-pixel and per-image coverage mask of the blur score.

    Blur detection is based on the laplacian of the grayscale image.
    Works for H&E stained tissue only, as it uses separation of stains
    to omit gradient values on the edges of hematoxylin nuclei.

    Args:
        img: RGB Image of the H&E stained tissue, minimum size 10x10 pixels.
        threshold: The threshold used to classify individual 16x16 pixel blocks of the output mask.
            If the average absolute Laplacian gradient value within a block falls below
            this threshold, the block is classified as blurred.
            Threshold can range from 0 to 255+.
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

    from rationai.qc.blur import blur_score_laplacian

    img = immunohistochemistry()

    result = blur_score_laplacian(img)

    # Mask of the blur coverage range from 0.0 to 1.0
    coverage_mask = result["blur_score_coverage"]

    # Binary mask of the blur detections
    blur_score_per_pixel = result["blur_score_per_pixel"]
    ```

    """
    grayscale_img = rgb2gray(img)

    if foreground_mask is None:
        foreground_mask = simple_foreground_mask(grayscale_img)

    hematoxylin, _, _ = convert_color(img, StandardConversions.RGB2HER)

    hematoxylin_threshold = threshold_otsu(hematoxylin)
    hematoxylin_mask = hematoxylin > hematoxylin_threshold
    gradient = laplace(255 * grayscale_img, ksize=3)
    gradient = np.abs(gradient)

    footprint = np.ones((3, 3))
    # 2 pixels wide border is removed from the pooling mask for each hematoxylin nucleus
    # to ignore gradient values on the edges of the nuclei
    pooling_mask = foreground_mask * ~(
        binary_dilation(hematoxylin_mask, footprint=footprint)
        ^ binary_erosion(hematoxylin_mask, footprint=footprint)
    )

    blur_score = np.abs(gradient)
    blur_score_pooled = masked_average_pooling(blur_score, pooling_mask)

    # Threshold was set to 5 based on empirical testing
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
