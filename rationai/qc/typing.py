from dataclasses import dataclass
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


@dataclass(frozen=True)
class ResidualThresholds:
    """Class containing thresholds for residual artifact detection.

    The thresholds are applied to negative parts of all three converted channels
    and to the positive part of the third (residual) channel.
    Please note that all thresholds should be specified as **positive values**,
    the negative part of each channel is converted to positive values
    during the detection process.
    """

    c1_negative: float
    """Threshold for the negative part of the first converted channel."""

    c2_negative: float
    """Threshold for the negative part of the second converted channel."""

    c3_negative: float
    """Threshold for the negative part of the third converted channel."""

    c3_positive: float
    """Threshold for the positive part of the third converted channel."""

    def __post_init__(self) -> None:
        if any(
            threshold <= 0
            for threshold in (
                self.c1_negative,
                self.c2_negative,
                self.c3_negative,
                self.c3_positive,
            )
        ):
            raise ValueError("All thresholds should be specified as positive values.")


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
    """Dictionary containing an artifact mask and a number of examined and flagged pixels."""

    artifacts_per_pixel: BinaryMask
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

    blur_score_coverage: FloatingPointImage
    """Coverage mask of the blur score. Coverage ranges from 0.0 to 1.0.
    0.0 - no blur, 1.0 - full blur
    """

    number_of_examined_pixels: int
    """Number of pixels that were examined by the function and could
    be theoretically marked as artifacts.
    """

    number_of_flagged_pixels: int
    """Number of pixels that were labeled as artifacts by the function."""


class FoldArtifacts(TypedDict):
    """Dictionary containing the fold detection mask and intermediate results."""

    folding_per_pixel: BinaryMask
    """Mask of the fold detection"""

    thresholded_saturation: BinaryMask
    """Thresholded saturation channel. Intermediate result of folding detection.
    Can be used for debugging.
    """

    thresholded_value: BinaryMask
    """Thresholded value channel. Intermediate result of folding detection.
    Can be used for debugging.
    """

    thresholded_eosin: BinaryMask
    """Thresholded eosin channel. Intermediate result of folding detection.
    Can be used for debugging.
    """

    number_of_examined_pixels: int
    """Number of pixels that were examined by the function and could
    be theoretically marked as artifacts.
    """

    number_of_flagged_pixels: int
    """Number of pixels that were labeled as artifacts by the function."""
