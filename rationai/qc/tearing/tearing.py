import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import area_opening, binary_opening, disk

from rationai.qc.typing import BinaryMask, RGBImage


def _generate_tissue_mask_based_on_intensity(rgb_tile: RGBImage) -> BinaryMask:
    """Generates a tissue mask based on the rgb intensities.

    Args:
        rgb_tile : RGB image of the tile.

    Returns:
        Tissue mask of the given tile.
    """
    mask = np.logical_or(
        np.logical_or(rgb_tile[:, :, 0] < 230, rgb_tile[:, :, 1] < 230),
        rgb_tile[:, :, 2] < 230,
    )
    return mask


def tearing(img: RGBImage) -> dict[str, BinaryMask]:
    """Creates a binary mask of tissue tear artifacts.

    Args:
        img: RGB image of the tile.

    Returns:
        Dictionary with the key tearing test that contains the binary mask of detected tissue artifacts.
    """
    tissue_mask = _generate_tissue_mask_based_on_intensity(img)

    background_mask = ~tissue_mask

    top_hat = background_mask & ~binary_opening(background_mask, disk(15), mode="max")

    labeled_img = label(area_opening(top_hat > 0, 300))

    regions = regionprops(labeled_img)

    for region in regions:
        if region.eccentricity < 0.8:
            labeled_img[labeled_img == region.label] = 0

    tearing_test = labeled_img > 0

    return {"tearing_test": tearing_test}
