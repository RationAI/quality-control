import cv2 as cv
import numpy as np

from rationai.qc.typing import BinaryMask


def area_opening(img: BinaryMask, area_threshold: int) -> BinaryMask:
    """Removes connected components smaller than a given area threshold.

    Args:
        img: Binary mask to be processed.
        area_threshold: Minimum area of connected components to be kept.

    Returns:
        Binary mask with small connected components removed.
    """
    _, labels, stats, _ = cv.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=4
    )

    keep = stats[:, cv.CC_STAT_AREA] >= area_threshold
    keep[0] = False  # Background label should not be kept

    return keep[labels]


def erosion_with_disk(mask: BinaryMask, radius: int) -> BinaryMask:
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


def reconstruction(marker: BinaryMask, mask: BinaryMask) -> BinaryMask:
    """Performs morphological reconstruction (by dilation) of a binary mask.

    Args:
        marker: Binary mask that serves as the starting point for reconstruction.
        mask: Binary mask that serves as the constraint for reconstruction.

    Returns:
        Reconstructed binary mask.
    """
    num_labels, labels = cv.connectedComponents(mask.astype(np.uint8), connectivity=8)

    # Keep every mask component that contains at least one marker pixel
    keep = np.zeros(num_labels, dtype=bool)
    keep[labels[marker & mask]] = True
    keep[0] = False  # Background label should not be kept

    return keep[labels]
