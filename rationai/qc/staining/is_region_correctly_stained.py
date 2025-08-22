from rationai.qc.staining.staining_difference import staining_difference
from rationai.qc.typing import RGBImage
from rationai.staining import ColorConversion, ConversionType, estimate_stain_vectors


def is_region_correctly_stained(
    region: RGBImage,
    expected_staining: ColorConversion,
    he_thresholds: tuple[float, float] = (28, 0),
    hdab_thresholds: tuple[float, float] = (22, 25),
) -> bool:
    """Checks if the given region contains the expected staining.

    Args:
        region: Region to check represented as a numpy array.
        expected_staining: Staining that is assumed to be in the provided region.
            The staining is represented by the `ColorConversion` enum.
            Any conversion type representing the staining can be used.
        he_thresholds: Thresholds `(local_threshold, single_stain_threshold)` used by
            the underlying `staining_difference` function if the conversion represents
            H&E staining. For more information about the thresholds, please refer
            to the `staining_difference` function's documentation.
        hdab_thresholds: Thresholds `(local_threshold, single_stain_threshold)` used by
            the underlying `staining_difference` function if the conversion represents
            H&DAB staining. For more information about the thresholds, please refer
            to the `staining_difference` function's documentation.

    Returns:
        `True` if the detected color vectors match the expected staining,
            `False` otherwise.

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc import is_region_correctly_stained
    from rationai.staining import ColorConversion


    img = immunohistochemistry()

    hdab = ColorConversion.RGB2HDR
    he = ColorConversion.RGB2HER

    print(is_region_correctly_stained(region=img, expected_staining=hdab))
    print(is_region_correctly_stained(region=img, expected_staining=he))
    ```
    """
    if expected_staining.conv_type == ConversionType.STAIN2RGB:
        expected_staining = expected_staining.inverse

    if expected_staining == ColorConversion.RGB2HER:
        local_threshold, single_stain_treshold = he_thresholds
    elif expected_staining == ColorConversion.RGB2HDR:
        local_threshold, single_stain_treshold = hdab_thresholds
    else:
        raise ValueError(f"{expected_staining.name} conversion is not supported")

    detected_stain1, detected_stain2 = estimate_stain_vectors(img=region)

    return staining_difference(
        conversion=expected_staining,
        stain1=detected_stain1,
        stain2=detected_stain2,
        local_threshold=local_threshold,
        single_stain_threshold=single_stain_treshold,
    )["correct_staining"]
