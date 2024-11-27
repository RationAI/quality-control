import numpy as np
from numpy.typing import NDArray

from rationai.qc.typing import QcValues


def is_slide_correctly_stained(
    stain1_diffs: list[float] | NDArray[np.float64],
    stain2_diffs: list[float] | NDArray[np.float64],
    global_threshold: float,
) -> QcValues:
    """Decides if a slide is stained with an expected staining protocol.

    This function decides if the per-tile computed color differences are close
    enough for a slide to be considered stained correctly
    (according to a given threshold). All of the differences
    are aggreagated using median.

    Args:
        stain1_diffs: List of differences for the first stain.
        stain2_diffs: List of differences for the second stain.
        global_threshold: Threshold used to determine if the differences
            are close enough

    Returns:
        Dictionary with answer if the slide is stained correctly.

    Note:
        The returned dictionary contains the following values:

        | Key                   | Description                                                   |
        |-----------------------|---------------------------------------------------------------|
        | `correct_staining`    | True if the differences are considered to be close enough.    |
        | `stain1_diff_median`  | Median of differences for the first stain.                    |
        | `stain2_diff_median`  | Median of difference for the second stain.                    |
    """
    result: QcValues = {}

    median1 = np.nanmedian(stain1_diffs)
    median2 = np.nanmedian(stain2_diffs)

    result["stain1_diff_median"] = median1
    result["stain2_diff_median"] = median2
    result["correct_staining"] = (median1 + median2) < global_threshold

    return result
