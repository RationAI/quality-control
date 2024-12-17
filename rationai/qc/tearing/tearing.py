import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import area_opening, binary_opening, disk

from rationai.qc.typing import BinaryMask, RGBImage


def _generate_tissue_mask_based_on_intensity(img: RGBImage) -> BinaryMask:
    mask = np.logical_or(
        np.logical_or(img[:, :, 0] < 230, img[:, :, 1] < 230),
        img[:, :, 2] < 230,
    )
    return mask


def tearing(img: RGBImage) -> dict[str, BinaryMask]:
    tissue_mask = binary_opening(_generate_tissue_mask_based_on_intensity(img), disk(2))

    background_mask = ~tissue_mask

    top_hat = background_mask & ~binary_opening(background_mask, disk(15), mode="max")

    labeled_img = label(area_opening(top_hat > 0, 300))

    regions = regionprops(labeled_img)

    for region in regions:
        if region.eccentricity < 0.8:
            labeled_img[labeled_img == region.label] = 0

    tearing_test = labeled_img > 0

    return {"tearing_test": tearing_test}
