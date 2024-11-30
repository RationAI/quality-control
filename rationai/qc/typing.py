from typing import Any, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


RGBImage: TypeAlias = NDArray[np.uint8]
"""
Three channel RGB image represented by a numpy array.
"""

Stain: TypeAlias = NDArray[np.float64]
"""
Single stain vector (of 3 values) represented by a numpy array.
"""

# TODO: Remove this type when it is no longer used by any function.
QcValues: TypeAlias = dict[str, Any]
"""
Dictionary with values computed by a QC function.
"""


class CorrectStaining(TypedDict):
    """Dictionary containing answer if the slide is stained correctly."""

    correct_staining: bool
    """True if the differences are considered to be small enough
    for a slide to be considered as correctly stained.
    """

    stain1_diff_median: float
    """Median of differences for the first stain."""

    stain2_diff_median: float
    """Median of differences for the second stain."""


class DominantStains(TypedDict):
    """Dictionary containing two detected dominant stains."""

    stain1: Stain
    """First dominant stain vector."""

    stain2: Stain
    """Second dominant stain vector."""


class ResidualArtifacts(TypedDict):
    """Dictionary containing a coverage mask and a coverage number."""

    coverage_mask: NDArray[np.uint8]
    """Binary mask of the detected residual artifacts."""

    coverage: float
    """A number that states what portion of the image's foreground area
    is covered by the artifacts.
    """


class StainingDifference(TypedDict):
    """Dictionary with color differences and a correct staining verdict."""

    stain_diff1: float
    """First stain difference."""

    stain_diff2: float
    """Second stain difference."""

    correct_staining: bool
    """True if the stain values are close to the expected ones
    (i.e., their color difference is small enough).
    """
