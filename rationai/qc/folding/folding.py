import numpy as np
from numpy.ma import MaskedArray
from numpy.typing import NDArray
from skimage.color import rgb2hsv
from skimage.filters import threshold_yen
from skimage.morphology import binary_opening, disk, reconstruction

from rationai.qc.typing import QcValues, RGBImage
from rationai.staining import ColorConversion, convert_color


def _get_threshold(
    img: NDArray[np.float32],
    mask: NDArray[bool],  # type: ignore[PGH003]
    local_tiles: NDArray[np.float32] | None = None,
    local_mask: NDArray[bool] | None = None,  # type: ignore[PGH003]
) -> float:
    """Calculates adequate threshold from given images.

    Args:
        img  : A given channel of an image.
        mask : Background mask of an image.
        local_tiles : Optional n*n tiles in local neighbourhood of tile. Defaults to None.
        local_mask : Optional n*n background mask of local_tiles. Defaults to None.

    Returns:
        Value which can be used to threshold the image.
    """
    if local_tiles is None:
        return threshold_yen(MaskedArray(img, mask).compressed())
    return threshold_yen(MaskedArray(local_tiles, local_mask).compressed())


def folding(
    img: RGBImage,
    level_downsample: float,
    hematoxylin_eosin_stained: bool,
    tissue_mask: NDArray[bool],  # type: ignore[PGH003]
    local_tiles: NDArray[np.uint8] | None = None,
    local_mask: NDArray[bool] | None = None,  # type: ignore[PGH003]
    nucleus_diameter_at_base_level: int = 30,
) -> QcValues:
    """Creates a binary mask of folding artifacts.

    Args:
        img: RGB image of the tissue.
        level_downsample: Downsample at the level at which the image is provided.
        hematoxylin_eosin_stained: True if image is stained using Hematoxylin and Eosin.
        tissue_mask: A mask, where the tissue is labeled 1 and the background 0,
            should be as pixel-precise as possible.
        local_tiles: A local area surrounding the given tile.
        local_mask: Tissue mask of local_tiles.
        nucleus_diameter_at_base_level: Diameter of the nucleus at the highest resolution.

    Returns:
        Dictionary with a binary mask of folds.

    Note:
        The returned dictionary contains the following values:

        | Key                       | Description                           |
        |---------------------------|---------------------------------------|
        | `folding`                 | Binary mask of the detected folds.    |
        | `thresholded_saturation`  |                                       |
        | `thresholded_value`       |                                       |
        | `thresholded_eosin`       |                                       |

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc.folding.folding import folding


    img = immunohistochemistry()
    tissue_mask = function_for_tissue_mask(img)

    result = folding(img, 8, False, tissue_mask)

    mask = result["folding"]  # Contains values 0 and 1
    ```

    """
    result: QcValues = {}

    tile = img
    mask = tissue_mask == 0
    hsv_tile = rgb2hsv(tile)
    saturation_channel, value_channel = hsv_tile[:, :, 1], hsv_tile[:, :, 2]
    local_saturation_channel, local_value_channel, local_eosin_channel = (
        None,
        None,
        None,
    )
    if local_tiles is not None:
        hsv_local = rgb2hsv(local_tiles)
        local_saturation_channel, local_value_channel = (
            hsv_local[:, :, 1],
            hsv_local[:, :, 2],
        )
        local_value_channel = 1 - local_value_channel
    if hematoxylin_eosin_stained:
        _, eosin_channel, _ = convert_color(tile, ColorConversion.RGB2HER)
        if local_tiles is not None:
            _, local_eosin_channel, _ = convert_color(
                local_tiles, ColorConversion.RGB2HER
            )
    else:
        eosin_channel = np.ones_like(mask)
        if local_tiles is not None:
            local_eosin_channel = np.ones_like(local_tiles)

    inverted_value_channel = 1 - value_channel

    value_threshold = _get_threshold(
        inverted_value_channel, mask, local_value_channel, local_mask
    )
    saturation_threshold = _get_threshold(
        saturation_channel, mask, local_saturation_channel, local_mask
    )
    if hematoxylin_eosin_stained:
        eosin_threshold = _get_threshold(
            eosin_channel, mask, local_eosin_channel, local_mask
        )
    else:
        eosin_threshold = 0

    thresholded_saturation = saturation_channel > saturation_threshold
    thresholded_value = inverted_value_channel > value_threshold
    thresholded_eosin = eosin_channel > eosin_threshold

    result["thresholded_saturation"] = thresholded_saturation
    result["thresholded_value"] = thresholded_value
    result["thresholded_eosin"] = thresholded_eosin

    if (
        np.sum(thresholded_value) * 2 > tile.size
        or np.sum(thresholded_saturation) * 2 > tile.size
        or (hematoxylin_eosin_stained and np.sum(thresholded_eosin) * 2 > tile.size)
    ):
        thresholded_value = np.zeros(tile.shape)

    folding_test_markers = binary_opening(
        thresholded_eosin & thresholded_saturation & thresholded_value,
        disk(nucleus_diameter_at_base_level // level_downsample),
    )

    if hematoxylin_eosin_stained:
        folding_test = reconstruction(folding_test_markers, thresholded_eosin)
        result["folding"] = folding_test
        return result
    result["folding"] = folding_test_markers
    return result
