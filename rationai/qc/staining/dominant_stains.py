from rationai.qc.typing import DominantStains, RGBImage
from rationai.staining import estimate_stain_vectors


def dominant_stains(
    img: RGBImage, i0: int = 240, alpha: int = 1, beta: float = 0.15
) -> DominantStains:
    """Estimates dominant stain vectors for a given image.

    Args:
        img: Tissue stained with two stains.
        i0: Transmitted light intensity (i.e., what is the intensity of light
            that passed through no tissue).
        alpha: Percentile offset for robust stain estimation.
        beta: Threshold for removing transparent pixels in OD-space.

    Note:
        Default values for `i0`, `alpha`, and `beta` parameters are inspired
        by the <a href="https://github.com/schaugf/HEnorm_python">reference implementation</a>.

    Returns:
        Dictionary with two dominant stains.

    Note:
        The returned dictionary contains the following values:

        | Key       | Description                     |
        |-----------|---------------------------------|
        | `stain1`  | First dominant stain vector.    |
        | `stain2`  | Second dominant stain vector.   |

    Examples:
    ```python
    from skimage.data import immunohistochemistry

    from rationai.qc.staining import dominant_stains


    img = immunohistochemistry()
    result = dominant_stains(img=img)

    stain1, stain2 = result["stain1"], result["stain2"]

    # Values should be somewhat close to Hematoxylin and DAB stain vectors.
    print(stain1, stain2)
    ```
    """
    stain1, stain2 = estimate_stain_vectors(img=img, i0=i0, alpha=alpha, beta=beta)

    result: DominantStains = {
        "stain1": stain1,
        "stain2": stain2,
    }

    return result
