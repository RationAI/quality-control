from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


BinaryMask: TypeAlias = NDArray[np.bool_]
"""
Binary mask only containing values 0 and 1.
"""

RGBImage: TypeAlias = NDArray[np.uint8]
"""
Three channel RGB image represented by a numpy array.
"""

GrayScaleImage: TypeAlias = NDArray[np.uint8]
"""
Single channel grayscale image represented by a numpy array.
"""

FloatingPointImage: TypeAlias = NDArray[np.float64]
"""
Floating point image represented by a numpy array.
"""

Stain: TypeAlias = NDArray[np.float64]
"""
Single stain vector (of 3 values) represented by a numpy array.
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
    """Dictionary containing a coverage mask and a number of examined and flagged pixels."""

    coverage_mask: BinaryMask
    """Binary mask of the detected residual artifacts."""

    number_of_examined_pixels: int
    """Number of pixels that were examined by the function and could
    be theoretically marked as artifacts.
    """

    number_of_flagged_pixels: int
    """Number of pixels that were labeled as artifacts by the function."""


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


class BlurScore(TypedDict):
    """Dictionary containing the blur score masks."""

    blur_score_per_pixel: BinaryMask
    """Binary mask composed of 16x16 px blocks marking blurred area"""

    blur_score_coverage: NDArray[np.float64]
    """Coverage mask of the blur score. Coverage ranges from 0.0 to 1.0.
    0.0 - no blur, 1.0 - full blur
    """


class FoldArtifacts(TypedDict):
    """Dictionary containing the fold detection mask and intermediate results."""

    folding: BinaryMask
    """Mask of the fold detection"""

    thresholded_saturation: BinaryMask
    """Thresholded saturation channel. Intermediate result of folding detection. Can be used for debugging."""

    thresholded_value: BinaryMask
    """Thresholded value channel.Intermediate result of folding detection. Can be used for debugging."""

    thresholded_eosin: BinaryMask
    """Thresholded eosin channel.Intermediate result of folding detection. Can be used for debugging."""
