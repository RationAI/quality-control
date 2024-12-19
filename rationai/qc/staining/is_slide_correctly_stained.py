import numpy as np
from numpy.typing import NDArray

from rationai.qc.typing import CorrectStaining


def is_slide_correctly_stained(
    stain1_diffs: list[float] | NDArray[np.float64],
    stain2_diffs: list[float] | NDArray[np.float64],
    global_threshold: float,
) -> CorrectStaining:
    """Decides if a slide is stained with an expected staining protocol.

    This function decides if the per-tile computed color differences are close
    enough for a slide to be considered stained correctly
    (according to a given threshold). All of the differences
    are aggreagated using median.

    Args:
        stain1_diffs: List of differences for the first stain.
        stain2_diffs: List of differences for the second stain.
        global_threshold: Threshold used to determine if the color differences
            are small enough for a slide to be considered correctly stained.
            Currently recommended thresold values are `28` for H&E stained slides
            and `22` for H&DAB stained slides (assuming the differeces were computed
            by the `ciede_2000` color difference method). Other similar values
            might also provide reasonable results.

    Returns:
        Dictionary with answer if the slide is stained correctly.

    Note:
        The returned dictionary contains the following values:

        | Key                   | Description                                                                                                   |
        |-----------------------|---------------------------------------------------------------------------------------------------------------|
        | `correct_staining`    | True if the differences are considered to be small enough for a slide to be considered as correctly stained.  |
        | `stain1_diff_median`  | Median of differences for the first stain.                                                                    |
        | `stain2_diff_median`  | Median of difference for the second stain.                                                                    |
    """
    median1 = np.nanmedian(stain1_diffs)
    median2 = np.nanmedian(stain2_diffs)

    result: CorrectStaining = {
        "stain1_diff_median": float(median1),
        "stain2_diff_median": float(median2),
        "correct_staining": bool((median1 + median2) < global_threshold),
    }

    return result
