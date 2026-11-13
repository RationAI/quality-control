from typing import get_args

import numpy as np
from rationai.staining import ColorConversion, convert_color

from rationai.qc.image_processing_algorithms import (
    area_opening,
    erosion_with_disk,
    reconstruction,
)
from rationai.qc.typing import (
    BinaryMask,
    NegativeChannel,
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
    # Value of zero corresponds to a pixel that did not capture any light
    # Nearly no stain -> low OD values
    od = np.maximum(0, -np.log((img.astype(np.float64) + 1) / i0))

    # Pixel is labeled as foreground if it is larger than beta in at least one channel
    return np.any(od >= beta, axis=2)


def _get_debris_coverage(
    img: RGBImage,
    conv: ColorConversion,
    nucleus_area: int,
    thresholds: ResidualThresholds,
    suppression_channel: NegativeChannel | None = None,
    erosion_radius: int = 0,
) -> tuple[int, int, BinaryMask]:
    """Computes foreground debris coverage by thresholding specific channels.

    Args:
        img: Image of a tissue.
        conv: Color conversion that should be performed to extract the residual channel.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        thresholds: Thresholds for residual artifact detection.
            See the `StandardResidualThresholds` class for suggested threshold values.
        suppression_channel: Optional negative channel in which small detections are suppressed
            (e.g., clusters of tightly packed nuclei falsely detected in the negative eosin
            channel of H&E images). If specified, only connected components of the detection
            in this channel that survive erosion with a disk of radius `erosion_radius` are kept.
        erosion_radius: Radius of the disk (in pixels) used for erosion. Only relevant
            if `suppression_channel` is specified. Defaults to 0.
            If `suppression_channel` is to be used, it is recommended to set `erosion_radius`
            based on the approximate size of objects that should be suppressed.

    Returns:
        Number of foreground pixels examined, number of foreground pixels
            marked as artifact, and the thresholded residual channel.
    """
    c1, c2, c3 = np.asarray(
        convert_color(tile=img, conversion=conv, keep_negative_values=True),
        dtype=np.float64,
    )

    negative_masks: dict[NegativeChannel, BinaryMask] = {
        "c1_negative": -np.minimum(c1, 0) >= thresholds.c1_negative,
        "c2_negative": -np.minimum(c2, 0) >= thresholds.c2_negative,
        "c3_negative": -np.minimum(c3, 0) >= thresholds.c3_negative,
    }

    mask_c3_pos = +np.maximum(c3, 0) >= thresholds.c3_positive

    if suppression_channel is not None and erosion_radius > 0:
        # Suppress small detections in the selected channel (e.g., tightly packed nuclei)
        channel = negative_masks[suppression_channel]
        marker = erosion_with_disk(channel, radius=erosion_radius)

        negative_masks[suppression_channel] = reconstruction(marker, channel)

    residual_mask = np.logical_or(
        np.logical_or(negative_masks["c1_negative"], negative_masks["c2_negative"]),
        np.logical_or(mask_c3_pos, negative_masks["c3_negative"]),
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
    suppression_channel: NegativeChannel | None = None,
    erosion_radius: int = 0,
) -> ResidualArtifacts:
    """Creates a binary mask of residual artifacts.

    Args:
        img: Image of a tissue.
        conversion: Conversion that describes the used staining protocol.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        thresholds: Thresholds for residual artifact detection.
            See the `StandardResidualThresholds` class for suggested threshold values.
        suppression_channel: Optional negative channel in which small detections are suppressed
            (e.g., clusters of tightly packed nuclei falsely detected in the negative eosin
            channel of H&E images). One of `"c1_negative"`, `"c2_negative"`, or `"c3_negative"`.
            If specified, only connected components of the detection in this channel
            that survive erosion with a disk of radius `erosion_radius` are kept.
        erosion_radius: Radius of the disk (in pixels) used for erosion. Only relevant
            if `suppression_channel` is specified. Defaults to 0.
            If `suppression_channel` is to be used, it is recommended to set `erosion_radius`
            based on the approximate size of objects that should be suppressed.

    Returns:
        Dictionary with a number of examined pixels, number of flagged pixels,
            and a binary mask of artifacts.

    Raises:
        ValueError: If `suppression_channel` is not one of the supported negative channels.

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
    if suppression_channel is not None and suppression_channel not in get_args(
        NegativeChannel
    ):
        raise ValueError(
            f"Unsupported suppression channel '{suppression_channel}'. "
            f"Expected one of: {', '.join(get_args(NegativeChannel))}."
        )

    num_examined, num_flagged, cov_heatmap = _get_debris_coverage(
        img=img,
        conv=conversion,
        nucleus_area=nucleus_area,
        thresholds=thresholds,
        suppression_channel=suppression_channel,
        erosion_radius=erosion_radius,
    )

    result: ResidualArtifacts = {
        "artifacts_per_pixel": cov_heatmap,
        "number_of_examined_pixels": num_examined,
        "number_of_flagged_pixels": num_flagged,
    }

    return result
