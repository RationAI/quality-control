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
        mask: Foreground mask of an image.
        neighborhood_tiles: Optional n*n tiles in local neighbourhood of tile. Defaults to None.
        neighborhood_mask: Optional n*n foreground mask of `neighborhood_tiles`. Defaults to None.

    Returns:
        Value which can be used to threshold the image.
    """
    if (neighborhood_tiles is None) != (neighborhood_mask is None):
        raise ValueError(
            "Both neighborhood_tiles and neighborhood_mask must be provided together."
        )

    if neighborhood_tiles is not None and neighborhood_mask is not None:
        local_values = MaskedArray(neighborhood_tiles, ~neighborhood_mask).compressed()

        if local_values.size > 0:
            return threshold_yen(local_values)

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
        Dictionary with a binary mask of folds, intermediate thresholding results,
            and a number of examined and flagged pixels.

    Note:
        The returned dictionary contains the following values:

        | Key                         | Description                                                                                                                              |
        |-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
        | `folding_per_pixel`         | Binary mask of the detected folds.                                                                                                       |
        | `thresholded_saturation`    | Thresholded saturation channel (intermediate result, useful for debugging).                                                              |
        | `thresholded_value`         | Thresholded inverted value channel (intermediate result, useful for debugging). All zeros if the thresholding is unreliable (see below). |
        | `thresholded_eosin`         | Thresholded eosin channel (intermediate result, useful for debugging). All ones if `hematoxylin_eosin_stained` is `False`.               |
        | `number_of_examined_pixels` | Number of pixels that were evaluated by the function.                                                                                    |
        | `number_of_flagged_pixels`  | Number of pixels labeled as artifacts.                                                                                                   |

        The thresholding is considered unreliable if more than half of the image pixels
        exceed the threshold in the saturation, inverted value, or (for H&E stained images) eosin channel.
        In that case, `thresholded_value` is replaced with zeros and no folds are detected.

    Examples:
    ```python
    import pyvips
    from rationai.masks import tissue_mask
    from skimage.data import immunohistochemistry

    from rationai.qc import folding


    img = immunohistochemistry()
    mpp = 1  # Approximate pixel size of the sample image in micrometers
    img_mask = tissue_mask(pyvips.Image.new_from_array(img), mpp=mpp).numpy() > 0

    result = folding(img, mpp, False, img_mask)

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
            local_eosin_channel = np.ones(
                neighborhood_tiles.shape[:2], dtype=np.float64
            )

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
        np.sum(thresholded_value) * 2 > np.prod(tile.shape[:2])
        or np.sum(thresholded_saturation) * 2 > np.prod(tile.shape[:2])
        or (
            hematoxylin_eosin_stained
            and np.sum(thresholded_eosin) * 2 > np.prod(tile.shape[:2])
        )
    ):
        # If more than half of the pixels for any of the tested images are above the threshold,
        # the thresholding is considered unreliable and an empty mask is computed.
        thresholded_value = np.zeros_like(thresholded_value)

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
