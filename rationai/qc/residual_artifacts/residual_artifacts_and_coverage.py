import numpy as np
import pyvips
from numpy.typing import NDArray
from PIL.Image import Image
from skimage.morphology import area_opening

from rationai.qc.typing import QcValues
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


def _get_foreground_mask(
    tile: NDArray[np.uint8], i0: int = 240, beta: float = 0.15
) -> NDArray[np.uint8]:
    """Returns binary mask of a foreground for a given tile.

    Args:
        tile: One tile from the wsi.
        i0: Intensity of the transmitted light (through no stain). Defaults to 240.
        beta: Threshold for a pixel to be considered a foreground. Defaults to 0.15.

    Returns:
        Binary foreground mask of the tile.
    """
    if isinstance(tile, Image):
        tile = np.array(tile)

    tile = tile.astype(np.float64)

    # Value of zero corresponds to a pixel that did not capture any light
    # Nearly no stain -> low OD values
    od = np.maximum(0, -np.log((tile + 1) / i0))
    od_channel_sum = np.sum(np.where(od >= beta, 1, 0), axis=2)

    # Pixel is labeled as foreground if it is larger than beta in at least one channel
    foreground_mask = np.where(od_channel_sum != 0, 1, 0).astype(np.uint8)

    return foreground_mask


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

    foreground_mask = _get_foreground_mask(tile=tile)
    foreground_area = np.count_nonzero(foreground_mask)

    # Keep only artifacts in the foreground
    residual_mask = np.multiply(residual_mask, foreground_mask)

    if foreground_area <= 0:
        return 0.0, residual_mask

    return float(round(np.sum(residual_mask) / foreground_area, 4)), residual_mask


def residual_artifacts_and_coverage(
    img: pyvips.Image,
    conversion: ColorConversion,
    nucleus_area: int,
    res_index: int,
    threshold: float,
) -> QcValues:
    """TODO: Precise documentation."""
    result: QcValues = {}

    numpy_img = img.numpy()
    shape = numpy_img.shape[:2]

    coverage, cov_heatmap = _get_debris_coverage(
        tile=numpy_img,
        conv=conversion,
        nucleus_area=nucleus_area,
        res_index=res_index,
        threshold=threshold,
    )

    cov_percent_heatmap = np.full(shape=shape, dtype=np.float64, fill_value=coverage)

    result["cov_heatmap"] = pyvips.Image.new_from_array(cov_heatmap)
    result["cov_percent_heatmap"] = pyvips.new_from_array(cov_percent_heatmap)
    result["coverage"] = coverage

    return result
