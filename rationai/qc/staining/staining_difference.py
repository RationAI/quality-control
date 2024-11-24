from rationai.qc.staining.color_difference import (
    closest_difference,
    color_difference,
    reference_stain,
)
from rationai.qc.typing import QcValues, Stain
from rationai.staining import ColorConversion


def staining_difference(
    conversion: ColorConversion,
    stain1: Stain,
    stain2: Stain,
    local_threshold: float,
    single_stain_threshold: float,
    color_difference_method: str,
    single_stain_difference_method: str,
) -> QcValues:
    """Decides if the given stains are close engough to the reference stains.

    Args:
        conversion: First two stains of this conversion specify
            the reference stains.

        stain1: First dominant stain.

        stain2: Second dominant stain.

        local_threshold: Threshold that determines if the specified stains
            are close enough to the expected stains.

        single_stain_threshold: Threshold that determines if `stain1` and `stain2`
            are too close to eachother. If so, the region in which the stains were
            detected is assumed to be stained only by a single stain and the second
            color difference is set to 0.

        color_difference_method: Method used for computing the color difference
            between the detected vectors and reference vectors.
            Options: `ciede_2000`, `ciede_94`, `cie_76`.

        single_stain_difference_method: Method used for computing the color difference
            between the two detected stain vectors in order to determine
            if the tissue is only stained by a single stain.
            Options: `ciede_2000`, `ciede_94`, `cie_76`.

    Returns:
        Dictionary with color differences.

        **Dictionary values**
        * `stain_diff1`: First stain difference.
        * `stain_diff2`: Second stain difference.
        * `correct_staining`: True if the stain values are close
            to the expected ones, False otherwise.

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc.staining import dominant_stains, staining_difference
    from rationai.staining import ColorConversion


    img = immunohistochemistry()

    result = dominant_stains(img=img)
    stain1, stain2 = result["stain1"], result["stain2"]

    result = staining_difference(
        ColorConversion.RGB2HDR, stain1, stain2, 33, 20, "ciede_2000", "cie_76"
    )
    print(result["correct_staining"])
    print(result["stain_diff1"], result["stain_diff2"])
    ```
    """
    result: QcValues = {}

    ref1 = reference_stain(conversion=conversion, index=0)
    ref2 = reference_stain(conversion=conversion, index=1)

    diff1, diff2 = closest_difference(
        stain1, stain2, ref1, ref2, color_difference_method
    )

    if (
        color_difference(stain1, stain2, method=single_stain_difference_method)
        < single_stain_threshold
    ):
        # The detected dominant stains are too close, they probably came from a region
        # stained with only one stain. Therefore, no color the difference is computed
        # for the second stain.
        diff1 = min(diff1, diff2)
        diff2 = 0.0

    result["stain_diff1"] = diff1
    result["stain_diff2"] = diff2
    result["correct_staining"] = (diff1 + diff2) < local_threshold

    return result
