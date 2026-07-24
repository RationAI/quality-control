import cv2 as cv
import numpy as np
from rationai.staining import ColorConversion, convert_color

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


def _area_opening(img: BinaryMask, area_threshold: int) -> BinaryMask:
    """Removes connected components smaller than a given area threshold.

    Args:
        img: Binary mask to be processed.
        area_threshold: Minimum area of connected components to be kept.

    Returns:
        Binary mask with small connected components removed.
    """
    output = np.zeros_like(img, dtype=bool)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=4
    )

    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= area_threshold:
            output[labels == i] = True

    return output


def _erosion_with_disk(mask: BinaryMask, radius: int) -> BinaryMask:
    """Erodes a binary mask with a disk of a given radius.

    Args:
        mask: Binary mask to be eroded.
        radius: Radius of the disk used for erosion.

    Returns:
        Eroded binary mask.
    """
    se_size = 2 * radius + 1
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_size, se_size))

    return cv.erode(mask.astype(np.uint8), se).astype(bool)


def _reconstruction(marker: BinaryMask, mask: BinaryMask) -> BinaryMask:
    """Performs morphological reconstruction of a binary mask.

    Args:
        marker: Binary mask that serves as the starting point for reconstruction.
        mask: Binary mask that serves as the constraint for reconstruction.

    Returns:
        Reconstructed binary mask.
    """
    cv_mask = mask.astype(np.uint8)
    cv_marker = marker.astype(np.uint8)

    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    cv_current = cv.bitwise_and(cv_marker, cv_mask)
    cv_reconstructed = np.zeros_like(cv_mask, dtype=np.uint8)

    while True:
        cv.dilate(cv_current, se, dst=cv_reconstructed)
        cv.bitwise_and(cv_reconstructed, cv_mask, dst=cv_reconstructed)

        if np.array_equal(cv_reconstructed, cv_current):
            break

        cv_current = cv_reconstructed.copy()

    return cv_current.astype(bool)


def _get_debris_coverage(
    img: RGBImage,
    conv: ColorConversion,
    nucleus_area: int,
    thresholds: ResidualThresholds,
    nuclei_channel: NegativeChannel | None = None,
    erosion_radius: int = 0,
) -> tuple[int, int, BinaryMask]:
    """Computes foreground debris coverage by thresholding specifc channels.

    Args:
        img: Image of a tissue.
        conv: Color conversion that should be performed to extract the residual channel.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        thresholds: Thresholds for residual artifact detection.
            See the `StandardResidualThresholds` class for suggested threshold values.
        nuclei_channel: Optional negative part of a separated channel that could contain
            incorrectly detected tightly packed nuclei. If specified, only connected
            components of the detection in this channel that survive erosion
            with a disk of radius `erosion_radius` are kept.
        erosion_radius: Radius of the disk used for erosion. Only relevant
            if `nuclei_channel` is specified. Defaults to 0.

    Returns:
        Number of foreground pixels examined, number of foreground pixels
            marked as artifact, and the thresholded residual channel.
    """
    c1, c2, c3 = np.asarray(
        convert_color(tile=img, conversion=conv, keep_negative_values=True),
        dtype=np.float64,
    )

    mask_c1_neg = -np.minimum(c1, 0) >= thresholds.c1_negative
    mask_c2_neg = -np.minimum(c2, 0) >= thresholds.c2_negative

    mask_c3_pos = +np.maximum(c3, 0) >= thresholds.c3_positive
    mask_c3_neg = -np.minimum(c3, 0) >= thresholds.c3_negative

    if nuclei_channel is not None and erosion_radius > 0:
        # Supressing incorrect detections of tightly packed nuclei
        channels = [mask_c1_neg, mask_c2_neg, mask_c3_neg]

        channel = channels[nuclei_channel.value]
        marker = _erosion_with_disk(channel, radius=erosion_radius)

        channels[nuclei_channel.value] = _reconstruction(marker, channel)

        mask_c1_neg, mask_c2_neg, mask_c3_neg = channels

    residual_mask = np.logical_or(
        np.logical_or(mask_c1_neg, mask_c2_neg), np.logical_or(mask_c3_pos, mask_c3_neg)
    )

    # Remove artifacts smaller that a single nucleus
    residual_mask = _area_opening(residual_mask, area_threshold=nucleus_area)

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
    nuclei_channel: NegativeChannel | None = None,
    erosion_radius: int = 0,
) -> ResidualArtifacts:
    """Creates a binary mask of residual artifacts.

    Args:
        img: Image of a tissue.
        conversion: Conversion that describes the used staining protocol.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        thresholds: Thresholds for residual artifact detection.
            See the `StandardResidualThresholds` class for suggested threshold values.
        nuclei_channel: Optional negative part of a separated channel that could contain
            incorrectly detected tightly packed nuclei. If specified, only connected
            components that survive erosion with a disk of radius `erosion_radius` are kept.
        erosion_radius: Radius of the disk used for erosion. Only relevant
            if `nuclei_channel` is specified. Defaults to 0.

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
        nuclei_channel=nuclei_channel,
        erosion_radius=erosion_radius,
    )

    result: ResidualArtifacts = {
        "artifacts_per_pixel": cov_heatmap,
        "number_of_examined_pixels": num_examined,
        "number_of_flagged_pixels": num_flagged,
    }

    return result
