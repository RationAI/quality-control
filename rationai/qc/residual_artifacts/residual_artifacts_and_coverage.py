import numpy as np
import pyvips
from numpy.typing import NDArray
from skimage.morphology import area_opening

from rationai.masks import tissue_mask
from rationai.qc.typing import QcValues, RGBImage
from rationai.staining import ColorConversion, convert_color


def _threshold_residual_channel(
    residual: NDArray[np.float64], threshold: float
) -> NDArray[np.uint8]:
    """Returns binary mask of the residual channel.

    Pixels with value of 1 represent artifacts.

    Args:
        residual: Array representing the residual channel.
        threshold: Value that is enough for a pixel to be marked as artifact.

    Returns:
        Binary mask of the residual channel.
    """
    return np.where(residual >= threshold, 1, 0).astype(np.uint8)


def _get_debris_coverage(
    tile: NDArray[np.uint8],
    conv: ColorConversion,
    nucleus_area: float,
    res_index: int,
    threshold: float,
) -> tuple[float, NDArray[np.uint8]]:
    """Calculates the percentage of the foreground debris coverage.

    Args:
        tile: One tile from the wsi.
        conv: Color conversion that should be performed to extract the residual channel.
        nucleus_area: Area covered by a single nucleus.
        res_index: Index of the residual channel.
        threshold: Value that is enough for a pixel to be marked as artifact.

    Returns:
        Number of pixels marked as artifact compared to number
            of foreground pixels and thresholded residual channel.
    """
    residual = np.asarray(
        convert_color(tile=tile, conversion=conv)[res_index], dtype=np.float64
    )
    residual_mask = _threshold_residual_channel(residual=residual, threshold=threshold)

    # Remove artifacts smaller that a single nucleus
    residual_mask = area_opening(residual_mask, area_threshold=nucleus_area)

    foreground_mask = tissue_mask(pyvips.Image.new_from_array(tile)).numpy()
    foreground_area = np.count_nonzero(foreground_mask)

    # Keep only artifacts in the foreground
    residual_mask = np.multiply(residual_mask, foreground_mask)

    if foreground_area <= 0:
        return 0.0, residual_mask

    return float(round(np.sum(residual_mask) / foreground_area, 4)), residual_mask


def residual_artifacts_and_coverage(
    img: RGBImage,
    conversion: ColorConversion,
    nucleus_area: int,
    res_index: int,
    threshold: float,
) -> QcValues:
    """Creates a binary mask of residual artifacts.

    Args:
        img: Image of the tissue.
        conversion: Conversion that describes the used staining protocol.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        res_index: Index of the residual channel in the used staining conversion.
        threshold: Threshold that determines if a given pixel is an artifact.

    Returns:
        Dictionary with a binary mask.

        **Dictionary values**
        * `coverage_mask`: Binary mask of the detected residual artifacts.
        * `coverage`: A number that states what portion of the image's foreground
            area is covered by the artifacts.

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc.residual_artifacts import residual_artifacts_and_coverage
    from rationai.staining import ColorConversion


    img = immunohistochemistry()

    result = residual_artifacts_and_coverage(
        img, ColorConversion.RGB2HDR, nucleus_area=150, res_index=2, threshold=0.012
    )

    mask = result["coverage_mask"]  # Contains values 0 and 1
    print(result["coverage"])
    ```
    """
    result: QcValues = {}

    coverage, cov_heatmap = _get_debris_coverage(
        tile=img,
        conv=conversion,
        nucleus_area=nucleus_area,
        res_index=res_index,
        threshold=threshold,
    )

    result["coverage_mask"] = cov_heatmap
    result["coverage"] = coverage

    return result
