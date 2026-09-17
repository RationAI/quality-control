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
    output = np.zeros_like(img, dtype=bool)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=4
    )

    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= area_threshold:
            output[labels == i] = True

    return output


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
