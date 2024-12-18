import numpy as np
from numpy.ma import MaskedArray
from numpy.typing import NDArray
from skimage.color import rgb2hsv
from skimage.filters import threshold_yen
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, disk, reconstruction

from rationai.qc.typing import QcValues, RGBImage
from rationai.staining import ColorConversion, convert_color


def _get_mean_std_from_hist(n: NDArray[np.uint32], bins: NDArray[np.int32]):
    """Calculates mean and standard deviation from histogram.

    Args:
        n : Frequency of occurence in the corresponding bin.
        bins : The bins of the histogram

    Returns:
        The mean and the standard deviation of the values.
    """
    mids = 0.5 * (bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    var = np.average((mids - mean) ** 2, weights=n)
    std = np.sqrt(var)
    return mean, std


def folding(
    img: RGBImage,
    mpp: float,
    tissue_mask: BinaryMask,
    local_tiles: RGBImage,
    local_mask: BinaryMask,
    cell_nucleus_size: float = 7,
) -> FoldArtifacts:
    """Creates a binary mask of folding artifacts.

    Args:
        img: RGB image of the tissue.
        mpp: Number of microns per pixel of image.
        tissue_mask: A mask, where the tissue is labeled 1 and the background 0,
            should be as pixel-precise as possible.
        local_tiles: A local area surrounding the given tile.
        local_mask: Tissue mask of local_tiles.
        nucleus_diameter_at_base_level: Diameter of the nucleus at the highest resolution level (typically level 0).

    Returns:
        Dictionary with a binary mask of folds.

    Note:
        The returned dictionary contains the following values:

        | Key                       | Description                           |
        |---------------------------|---------------------------------------|
        | `folding`                 | Binary mask of the detected folds.    |
        | `thresholded_saturation`  |                                       |
        | `thresholded_value`       |                                       |
        | `thresholded_eosin`       |                                       |

    Examples:
    ```python
    import numpy as np
    from PIL import Image

    import pyvips
    from rationai.masks import tissue_mask


    img = np.asarray(Image.open("fold.png").conver("RGB"))  # Investigated image
    local_area_image = img_area = np.asarray(
        Image.open("fold_area.png").convert("RGB")
    )  # Local area of investigated image

    img_mask = tissue_mask(pyvips.Image.new_from_array(img), mpp=pixel_size).numpy() > 0

    img_area_mask = (
        tissue_mask(pyvips.Image.new_from_array(local_area_img), mpp=pixel_size).numpy()
        > 0
    )

    artifacts = folding(
        img=img,
        mpp=1.76,
        tissue_mask=img_mask,
        local_tiles=local_area_img,
        local_mask=img_area_mask,
    )

    mask = artifacts["folding"]  # Contains values 0 and 1
    ```

    """
    cell_nucleus_size_in_img = cell_nucleus_size / mpp

    _, e_channel, _ = convert_color(img, ColorConversion.RGB2HER)
    hsv = rgb2hsv(img)
    saturation_channel, value_channel = hsv[:, :, 1], hsv[:, :, 2]
    inverted_value_channel = 1 - value_channel

    hsv_local_tiles = rgb2hsv(local_tiles)
    local_saturation_channel, local_value_channel = (
        hsv_local_tiles[:, :, 1],
        hsv_local_tiles[:, :, 2],
    )
    _, local_eosin_channel, _ = convert_color(local_tiles, ColorConversion.RGB2HER)

    bins = np.linspace(0, 1, 257)

    local_saturation_histogram = np.histogram(
        MaskedArray(local_saturation_channel, ~local_mask).compressed(), bins
    )
    local_value_histogram = np.histogram(
        MaskedArray(1 - local_value_channel, ~local_mask).compressed(), bins
    )
    local_eosin_histogram = np.histogram(
        MaskedArray(local_eosin_channel, ~local_mask).compressed(), bins
    )

    saturation_threshold = threshold_yen(hist=local_saturation_histogram)
    eosin_threshold = threshold_yen(hist=local_eosin_histogram)
    value_threshold = threshold_yen(hist=local_value_histogram)

    thresholded_saturation = saturation_channel > saturation_threshold
    thresholded_eosin = e_channel > eosin_threshold
    thresholded_value = inverted_value_channel > value_threshold

    number_of_foreground_pixels = np.sum(tissue_mask)

    eosin_values, _ = np.histogram(
        MaskedArray(e_channel, ~tissue_mask).compressed(), bins=bins
    )

    current_mean, _ = _get_mean_std_from_hist(eosin_values, bins)
    local_mean_without_center = 1
    local_std_without_center = 1
    if np.all(local_eosin_histogram[0] >= eosin_values) and np.any(
        local_eosin_histogram[0] > eosin_values
    ):
        local_mean_without_center, local_std_without_center = _get_mean_std_from_hist(
            local_eosin_histogram[0] - eosin_values, bins
        )

    if not (current_mean > local_mean_without_center + local_std_without_center):
        if np.sum(thresholded_saturation) * 2 > number_of_foreground_pixels:
            thresholded_saturation = np.zeros_like(tissue_mask)
        if np.sum(thresholded_value) * 2 > number_of_foreground_pixels:
            thresholded_saturation = np.zeros_like(tissue_mask)
        if np.sum(thresholded_eosin) * 2 > number_of_foreground_pixels:
            thresholded_eosin = np.zeros_like(tissue_mask)

    folding_test_markers = binary_opening(
        thresholded_eosin & thresholded_saturation & thresholded_value,
        disk(cell_nucleus_size_in_img),
    )

    reconstructed_markers = reconstruction(folding_test_markers, thresholded_eosin)

    labeled_img, num_of_labels = label(reconstructed_markers, return_num=True)

    for i in range(1, num_of_labels):
        labeled_region = labeled_img == i
        if np.sum(
            binary_closing(labeled_region, disk(cell_nucleus_size_in_img * 5))
        ) > 1.5 * np.sum(labeled_region):
            labeled_img[labeled_img == i] = 0

    folding_test = labeled_img > 0

    result: FoldArtifacts = {
        "folding": folding_test,
        "thresholded_saturation": thresholded_saturation,
        "thresholded_eosin": thresholded_eosin,
        "thresholded_value": thresholded_value,
    }

    return result
