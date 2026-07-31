import numpy as np
from numpy.ma import MaskedArray
from rationai.staining import StandardConversions, convert_color
from skimage.color import rgb2hsv
from skimage.filters import threshold_yen
from skimage.morphology import disk, opening, reconstruction

from rationai.qc.typing import BinaryMask, FloatingPointImage, FoldArtifacts, RGBImage


def _get_threshold(
    img: FloatingPointImage,
    mask: BinaryMask,
    neighborhood_tiles: FloatingPointImage | None = None,
    neighborhood_mask: BinaryMask | None = None,
) -> float:
    """Calculates adequate threshold from given images.

    Args:
        img: A given channel of an image.
        mask: Background mask of an image.
        neighborhood_tiles: Optional n*n tiles in local neighbourhood of tile. Defaults to None.
        neighborhood_mask: Optional n*n background mask of neighborhood_tiles. Defaults to None.

    Returns:
        Value which can be used to threshold the image.
    """
    if neighborhood_tiles is not None and neighborhood_mask is not None:
        local_values = MaskedArray(neighborhood_tiles, ~neighborhood_mask).compressed()

        if local_values.size > 0:
            return threshold_yen(local_values)
        else:
            return 1.0

    values = MaskedArray(img, ~mask).compressed()

    if values.size == 0:
        return 1.0

    return threshold_yen(values)


def folding(
    img: RGBImage,
    mpp: float,
    hematoxylin_eosin_stained: bool,
    tissue_mask: BinaryMask,
    neighborhood_tiles: RGBImage | None = None,
    neighborhood_mask: BinaryMask | None = None,
    cell_nucleus_size: float = 7,
) -> FoldArtifacts:
    """Creates a binary mask of folding artifacts.

    Args:
        img: RGB image of the tissue.
        mpp: Number of microns per pixel of image.
        hematoxylin_eosin_stained: True if image is stained using Hematoxylin and Eosin.
        tissue_mask: A mask, where the tissue is labeled 1 and the background 0,
            should be as pixel-precise as possible.
        neighborhood_tiles: An image of neighborhood surrounding the given tile.
        neighborhood_mask: Tissue mask of neighborhood_tiles.
        cell_nucleus_size: Cell nucleus size in microns. This value is used for morphological operations.
            If estimating the value, it is better to overestimate the value.
            The default value is 7 based on empirical observations.

    Returns:
        Dictionary with a binary mask of folds.

    Note:
        The returned dictionary contains the following values:

        | Key                         | Description                                           |
        |-----------------------------|-------------------------------------------------------|
        | `folding_per_pixel`         | Binary mask of the detected folds.                    |
        | `thresholded_saturation`    |                                                       |
        | `thresholded_value`         |                                                       |
        | `thresholded_eosin`         |                                                       |
        | `number_of_examined_pixels` | Number of pixels that were evaluated by the function. |
        | `number_of_flagged_pixels`  | Number of pixels labeled as artifacts.                |

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc import folding


    img = immunohistochemistry()
    tissue_mask = function_for_tissue_mask(img)

    result = folding(img, 8, False, tissue_mask)

    mask = result["folding_per_pixel"]  # Contains values 0 and 1
    ```

    """
    tile = img
    hsv_tile = rgb2hsv(tile)
    saturation_channel, value_channel = hsv_tile[:, :, 1], hsv_tile[:, :, 2]
    local_saturation_channel, local_value_channel, local_eosin_channel = (
        None,
        None,
        None,
    )
    if neighborhood_tiles is not None:
        hsv_local = rgb2hsv(neighborhood_tiles)
        local_saturation_channel, local_value_channel = (
            hsv_local[:, :, 1],
            hsv_local[:, :, 2],
        )
        local_value_channel = 1 - local_value_channel
    if hematoxylin_eosin_stained:
        _, eosin_channel, _ = convert_color(tile, StandardConversions.RGB2HER)
        if neighborhood_tiles is not None:
            _, local_eosin_channel, _ = convert_color(
                neighborhood_tiles, StandardConversions.RGB2HER
            )
    else:
        eosin_channel = np.ones_like(tissue_mask)
        if neighborhood_tiles is not None:
            local_eosin_channel = np.ones_like(neighborhood_tiles, dtype=np.float64)

    inverted_value_channel = 1 - value_channel

    value_threshold = _get_threshold(
        inverted_value_channel, tissue_mask, local_value_channel, neighborhood_mask
    )
    saturation_threshold = _get_threshold(
        saturation_channel, tissue_mask, local_saturation_channel, neighborhood_mask
    )
    if hematoxylin_eosin_stained and eosin_channel is not None:
        eosin_threshold = _get_threshold(
            eosin_channel, tissue_mask, local_eosin_channel, neighborhood_mask
        )
    else:
        eosin_threshold = 0

    thresholded_saturation = saturation_channel > saturation_threshold
    thresholded_value = inverted_value_channel > value_threshold
    thresholded_eosin = eosin_channel > eosin_threshold

    if (
        np.sum(thresholded_value) * 2 > tile.size
        or np.sum(thresholded_saturation) * 2 > tile.size
        or (hematoxylin_eosin_stained and np.sum(thresholded_eosin) * 2 > tile.size)
    ):
        thresholded_value = np.zeros(tile.shape)

    folding_test_markers = opening(
        thresholded_eosin & thresholded_saturation & thresholded_value,
        disk(cell_nucleus_size // (mpp)),
    )

    if hematoxylin_eosin_stained:
        folding_test = reconstruction(folding_test_markers, thresholded_eosin)
    else:
        folding_test = folding_test_markers

    return {
        "folding_per_pixel": folding_test,
        "thresholded_saturation": thresholded_saturation,
        "thresholded_eosin": thresholded_eosin,
        "thresholded_value": thresholded_value,
        "number_of_examined_pixels": int(np.count_nonzero(tissue_mask)),
        "number_of_flagged_pixels": int(np.count_nonzero(folding_test)),
    }
