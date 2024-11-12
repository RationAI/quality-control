import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_yen
from skimage.morphology import binary_opening, disk, reconstruction

from rationai.qc.typing import QcValues, RGBImage
from rationai.staining import ColorConversion, convert_color


def _separate_tile_hsv(tile: np.array):
    hsv_image = rgb2hsv(tile)
    return hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]


def _generate_tissue_mask_based_on_intensity(rgb_tile: RGBImage):
    mask = np.logical_or(
        np.logical_or(rgb_tile[:, :, 0] < 230, rgb_tile[:, :, 1] < 230),
        rgb_tile[:, :, 2] < 230,
    )
    # closed_mask = binary_closing(mask, disk(40 // (2**level)))
    # opened_closed_mask = binary_opening(closed_mask, disk(40 // 2**level))
    return mask


def folding(
    img: RGBImage,
) -> QcValues:
    result: QcValues = {}

    tile = img
    mask = _generate_tissue_mask_based_on_intensity(tile)
    mask = mask == 0
    h_channel, eosin_channel, _ = convert_color(tile, ColorConversion.RGB2HER)
    _, saturation_channel, value_channel = _separate_tile_hsv(tile)

    # e_squared = e_channel * saturation_channel

    inverted_value_channel = 1 - value_channel

    value_threshold = threshold_yen(inverted_value_channel)
    saturation_threshold = threshold_yen(saturation_channel)
    eosin_threshold = threshold_yen(eosin_channel)

    thresholded_saturation = saturation_channel > saturation_threshold
    thresholded_value = inverted_value_channel > value_threshold
    thresholded_eosin = eosin_channel > eosin_threshold

    result["thresholded_saturation"] = thresholded_saturation
    result["thresholded_value"] = thresholded_value
    result["thresholded_eosin"] = thresholded_eosin

    if np.sum(thresholded_value) * 2 > tile.size:
        thresholded_value = np.zeros(tile.shape)

    folding_test_markers = binary_opening(
        np.logical_and(thresholded_eosin, thresholded_saturation), disk(30 // 8)
    )

    result["markers"] = folding_test_markers

    folding_test = reconstruction(folding_test_markers, thresholded_eosin)

    # data["fraction_of_fold_A"] = float(np.sum(folding_test)) / folding_test.size
    result["folding"] = folding_test
    return result
