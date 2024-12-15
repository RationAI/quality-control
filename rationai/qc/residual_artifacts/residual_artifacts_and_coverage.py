import numpy as np
from skimage.morphology import area_opening

from rationai.qc.typing import BinaryMask, ResidualArtifacts, RGBImage
from rationai.staining import ColorConversion, convert_color


def _get_foreground_mask(
    img: RGBImage, i0: int = 240, beta: float = 0.15
) -> BinaryMask:
    """Returns binary mask of a foreground for a given tile.

    Args:
        img: Image of the tissue.
        i0: Intensity of the transmitted light (through no stain). Defaults to 240.
        beta: Threshold for a pixel to be considered a foreground. Defaults to 0.15.

    Note:
        Default values for `i0` and `beta` parameters are inspired
        by the <a href="https://github.com/schaugf/HEnorm_python">reference implementation</a>.

    Returns:
        Binary foreground mask of the tile.
    """
    img = img.astype(np.float64)

    # Value of zero corresponds to a pixel that did not capture any light
    # Nearly no stain -> low OD values
    od = np.maximum(0, -np.log((img + 1) / i0))
    od_channel_sum = np.sum(np.where(od >= beta, 1, 0), axis=2)

    # Pixel is labeled as foreground if it is larger than beta in at least one channel
    return od_channel_sum != 0


def _get_debris_coverage(
    img: RGBImage,
    conv: ColorConversion,
    nucleus_area: float,
    res_index: int,
    threshold: float,
) -> tuple[float, BinaryMask]:
    """Calculates the percentage of the foreground debris coverage.

    Args:
        img: Image of a tissue.
        conv: Color conversion that should be performed to extract the residual channel.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        res_index: Index of the residual channel.
        threshold: Value that is enough for a pixel to be marked as artifact.
            Currently recommended threshold value for H&E stained slides is `0.005`.
            This value was declared emprirically and should provide a strict
            detection of artifacts.
            Depending on the match between the tissue and the expected staining protocol,
            slightly higher values could also provide reasonable results.

    Returns:
        Number of pixels marked as artifact compared to number
            of foreground pixels and thresholded residual channel.
    """
    residual = np.asarray(
        convert_color(tile=img, conversion=conv)[res_index], dtype=np.float64
    )
    # Experiments showed that if the color of a artifact substantially differs
    # from the expected staining, negative values big in magnitude can be generated
    # in the residual channel.
    residual_mask = np.abs(residual) >= threshold

    # Remove artifacts smaller that a single nucleus
    residual_mask = area_opening(residual_mask, area_threshold=nucleus_area)

    foreground_mask = _get_foreground_mask(img)
    foreground_area = np.count_nonzero(foreground_mask)

    # Keep only artifacts in the foreground
    residual_mask *= foreground_mask

    if foreground_area <= 0:
        return 0.0, residual_mask

    return float(
        round(np.count_nonzero(residual_mask) / foreground_area, 4)
    ), residual_mask


def residual_artifacts_and_coverage(
    img: RGBImage,
    conversion: ColorConversion,
    nucleus_area: int,
    res_index: int,
    threshold: float,
) -> ResidualArtifacts:
    """Creates a binary mask of residual artifacts.

    Args:
        img: Image of a tissue.
        conversion: Conversion that describes the used staining protocol.
        nucleus_area: Approximate area of a single cell nucleus in pixels.
        res_index: Index of the residual channel in the used staining conversion.
        threshold: Threshold that determines if a given pixel is an artifact.
            Currently recommended threshold value for H&E stained slides is `0.005`.
            This value was declared emprirically and should provide a strict
            detection of artifacts.
            Depending on the match between the tissue and the expected staining protocol,
            slightly higher values could also provide reasonable results.

    Returns:
        Dictionary with a binary coverage mask and a coverage number.

    Note:
        The returned dictionary contains the following values:

        | Key               | Description                                                                                   |
        |-------------------|-----------------------------------------------------------------------------------------------|
        | `coverage_mask`   | Binary mask of the detected residual artifacts.                                               |
        | `coverage`        | A number that states what portion of the image's foreground area is covered by the artifacts. |

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc.residual_artifacts import residual_artifacts_and_coverage
    from rationai.staining import ColorConversion


    img = immunohistochemistry()

    result = residual_artifacts_and_coverage(
        img, ColorConversion.RGB2HDR, nucleus_area=150, res_index=2, threshold=0.013
    )

    mask = result["coverage_mask"]  # Contains values 0 and 1
    print(result["coverage"])
    ```
    """
    coverage, cov_heatmap = _get_debris_coverage(
        img=img,
        conv=conversion,
        nucleus_area=nucleus_area,
        res_index=res_index,
        threshold=threshold,
    )

    result: ResidualArtifacts = {
        "coverage_mask": cov_heatmap,
        "coverage": coverage,
    }

    return result
