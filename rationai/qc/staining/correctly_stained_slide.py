import numpy as np
from numpy.typing import NDArray


def correctly_stained_slide(
    stain1_diffs: list[float] | NDArray[np.float64],
    stain2_diffs: list[float] | NDArray[np.float64],
    global_threshold: float,
) -> bool:
    """Decides if a slide is stained with a correct staining protocol.

    This function decides if the per-tile computed color differences are close
    enough for a slide to be considered stained correctly. All of the differencese
    are aggreagated using median.

    Args:
        stain1_diffs: List of differences for the first stain.
        stain2_diffs: List of differences for the second stain.
        global_threshold: Threshold used to determine if the differences
            are close enough

    Returns:
        True if the differences are considered to be close enough, according
            to the threshold.
    """
    median1 = np.nanmedian(stain1_diffs)
    median2 = np.nanmedian(stain2_diffs)

    return bool((median1 + median2) < global_threshold)
