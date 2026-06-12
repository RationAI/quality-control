import numpy as np
from rationai.staining import ColorConversion, convert_color
from skimage.morphology import area_opening

from rationai.qc.typing import (
    BinaryMask,
    ResidualArtifacts,
    ResidualThresholds,
    RGBImage,
)


def _get_foreground_mask(
    img: RGBImage, i0: int = 240, beta: float = 0.15
) -> BinaryMask:
    """Returns binary mask of a foreground for a given tile.

    Args:
        img: Image of the tissue.
        i0: Intensity of the transmitted light (through no stain). Defaults to 240.
        beta: Threshold for a pixel to be considered a foreground. Defaults to 0.15.

    Note:
        Default values for `i0` and `beta` parameters are inspired
        by the <a href="https://github.com/schaugf/HEnorm_python">reference implementation</a>.

    Returns:
        Binary foreground mask of the tile.
    """
    img = img.astype(np.float64)

    # Value of zero corresponds to a pixel that did not capture any light
    # Nearly no stain -> low OD values
    od = np.maximum(0, -np.log((img + 1) / i0))
    od_channel_sum = np.sum(np.where(od >= beta, 1, 0), axis=2)

    # Pixel is labeled as foreground if it is larger than beta in at least one channel
    return od_channel_sum != 0


def _get_debris_coverage(
    img: RGBImage,
    conv: ColorConversion,
    nucleus_area: int,
    thresholds: ResidualThresholds,
) -> tuple[int, int, BinaryMask]:
    """Computes foreground debris coverage by thresholding specifc channels.

    Args:
        img: Image of a tissue.
        conv: Color conversion that should be performed to extract the residual channel.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        thresholds: Thresholds for residual artifact detection.
            See the `StandardResidualThresholds` class for suggested threshold values.

    Returns:
        Number of foreground pixels examined, number of foreground pixels
            marked as artifact, and the thresholded residual channel.
    """
    c1, c2, c3 = np.asarray(
        convert_color(tile=img, conversion=conv, keep_negative_values=True),
        dtype=np.float64,
    )
    th = thresholds

    c1_neg, c2_neg = -np.minimum(c1, 0), -np.minimum(c2, 0)
    c3_pos, c3_neg = np.maximum(c3, 0), -np.minimum(c3, 0)

    # Join results from all thresholded channels
    residual_mask = np.logical_or(
        np.logical_or(c1_neg >= th.c1_negative, c2_neg >= th.c2_negative),
        np.logical_or(c3_pos >= th.c3_positive, c3_neg >= th.c3_negative),
    )

    # Remove artifacts smaller that a single nucleus
    residual_mask = area_opening(residual_mask, area_threshold=nucleus_area)

    foreground_mask = _get_foreground_mask(img)
    foreground_area = np.count_nonzero(foreground_mask)

    # Keep only artifacts in the foreground
    residual_mask &= foreground_mask

    if foreground_area <= 0:
        return 0, 0, residual_mask

    return int(foreground_area), int(np.count_nonzero(residual_mask)), residual_mask


def residual_artifacts_and_coverage(
    img: RGBImage,
    conversion: ColorConversion,
    nucleus_area: int,
    thresholds: ResidualThresholds,
) -> ResidualArtifacts:
    """Creates a binary mask of residual artifacts.

    Args:
        img: Image of a tissue.
        conversion: Conversion that describes the used staining protocol.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        thresholds: Thresholds for residual artifact detection.
            See the `StandardResidualThresholds` class for suggested threshold values.

    Returns:
        Dictionary with a number of examined pixels, number of flagged pixels,
            and a binary mask of artifacts.

    Note:
        The returned dictionary contains the following values:

        | Key                         | Description                                           |
        |-----------------------------|-------------------------------------------------------|
        | `artifacts_per_pixel`       | Binary mask of the detected residual artifacts.       |
        | `number_of_examined_pixels` | Number of pixels that were evaluated by the function. |
        | `number_of_flagged_pixels`  | Number of pixels labeled as artifacts.                |

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc import StandardResidualThresholds, residual_artifacts_and_coverage
    from rationai.staining import StandardConversions


    img = immunohistochemistry()

    result = residual_artifacts_and_coverage(
        img,
        StandardConversions.RGB2HDR,
        nucleus_area=150,
        thresholds=StandardResidualThresholds.HDR,
    )

    mask = result["artifacts_per_pixel"]  # Contains values 0 and 1
    print(result["number_of_flagged_pixels"], result["number_of_examined_pixels"])
    ```
    """
    num_examined, num_flagged, cov_heatmap = _get_debris_coverage(
        img=img,
        conv=conversion,
        nucleus_area=nucleus_area,
        thresholds=thresholds,
    )

    result: ResidualArtifacts = {
        "artifacts_per_pixel": cov_heatmap,
        "number_of_examined_pixels": num_examined,
        "number_of_flagged_pixels": num_flagged,
    }

    return result
