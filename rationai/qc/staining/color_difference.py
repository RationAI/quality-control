import numpy as np
from numpy.typing import NDArray
from rationai.staining import ColorConversion, ConversionDirection
from skimage.color import deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, rgb2lab

from rationai.qc.typing import Stain


def _stain2rgb(stain: Stain) -> NDArray[np.float64]:
    """Converts a pixel in stain space to rgb space."""
    return np.exp(-np.maximum(stain, 1e-6))


def reference_stain(conversion: ColorConversion, index: int) -> Stain:
    """Return specified stain vector for a given conversion."""
    if conversion.direction == ConversionDirection.RGB2STAIN:
        conversion = conversion.inverse

    return conversion.matrix[index]


def color_difference(
    reference: Stain, comparison: Stain, method: str = "ciede_2000"
) -> float:
    """Computes color difference between two staining vectors.

    Args:
        reference: Reference stain vector of shape (3,).
        comparison: Comparison stain vector of shape (3,).
        method: Method used for computing the color difference.
            Options: `ciede_2000`, `ciede_94`, `cie_76`.

    Note:
        According to our empirical analysis, `ciede_2000` method
        provides the most stable results for the typical use case.
        This use case mainly includes determining the color difference
        between a detected and reference stain vectors.
        On the other hand, `cie_76` proved to be useful for computing
        the difference between two detected stains in order to determine
        if the stain vectors could resemble the same stain.

        All of the color difference functions, and their properties
        can be found on <a href="https://en.wikipedia.org/wiki/Color_difference#">Wikipedia</a>.

    Returns:
        Distance between reference and comparison vectors
        in **Lab colorspace**.
    """
    if np.any(np.isnan(comparison)):
        # Comparison vector is not defined
        return np.nan

    reference = rgb2lab(_stain2rgb(reference))
    comparison = rgb2lab(_stain2rgb(comparison))

    match method:
        case "ciede_2000":
            return deltaE_ciede2000(reference, comparison)
        case "ciede_94":
            return deltaE_ciede94(reference, comparison)
        case "cie_76":
            return deltaE_cie76(reference, comparison)
        case _:
            raise ValueError(f"{method} color difference method is not supported.")


def closest_difference(
    stain1: Stain, stain2: Stain, ref1: Stain, ref2: Stain, method: str
) -> tuple[float, float]:
    """Computes the closest color differences between two sets of stains.

    Args:
        stain1: First comparison stain vector.
        stain2: Second comparison stain vector.
        ref1: First reference stain vector.
        ref2: Second reference stain vector.
        method: Method used for computing the color difference.
            Options: `ciede_2000`, `ciede_94`, `cie_76`.

    Note:
        According to our empirical analysis, `ciede_2000` method
        provides the most stable results for the typical use case.
        This use case mainly includes determining the color difference
        between a detected and reference stain vectors.
        On the other hand, `cie_76` proved to be useful for computing
        the difference between two detected stains in order to determine
        if the stain vectors could resemble the same stain.

        All of the color difference functions, and their properties
        can be found on <a href="https://en.wikipedia.org/wiki/Color_difference#">Wikipedia</a>.

    Returns:
        Color differences between the reference and comparison vectors.
        Comparison vectors are paired with reference vectors in a way that
        the sum of both differences is minimized.
    """
    aligned = (
        color_difference(reference=ref1, comparison=stain1, method=method),
        color_difference(reference=ref2, comparison=stain2, method=method),
    )
    opposite = (
        color_difference(reference=ref2, comparison=stain1, method=method),
        color_difference(reference=ref1, comparison=stain2, method=method),
    )

    sums = np.array([np.sum(aligned), np.sum(opposite)])
    closer_diff = (aligned, opposite)[sums.argmin()]

    return closer_diff[0], closer_diff[1]
