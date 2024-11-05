from rationai.qc.staining.color_difference import closest_difference, reference_stain
from rationai.qc.typing import QcValues, Stain
from rationai.staining import ColorConversion


def staining_difference(
    conversion: ColorConversion,
    stain1: Stain,
    stain2: Stain,
    local_threshold: float,
    color_difference: str,
) -> QcValues:
    """Decides if the given stains are close engough to the reference stains.

    Args:
        conversion: First two stains of this conversion specify
            the reference stains.
        stain1: First dominant stain.
        stain2: Second dominant stain.
        local_threshold: Threshold that determines if the specified stains
            are close enough to the expected stains.
        color_difference: Method used for computing the color difference.
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
        ColorConversion.RGB2HDR, stain1, stain2, 33, "ciede_2000"
    )
    print(result["correct_staining"])
    print(result["stain_diff1"], result["stain_diff2"])
    ```
    """
    result: QcValues = {}

    ref1 = reference_stain(conversion=conversion, index=0)
    ref2 = reference_stain(conversion=conversion, index=1)

    diff1, diff2 = closest_difference(stain1, stain2, ref1, ref2, color_difference)

    result["stain_diff1"] = diff1
    result["stain_diff2"] = diff2
    result["correct_staining"] = (diff1 + diff2) < local_threshold

    return result
