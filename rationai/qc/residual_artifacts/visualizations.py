import numpy as np
from PIL import Image
from rationai.staining import ColorConversion, convert_color

from rationai.qc.typing import ResidualThresholds, RGBImage


GRAY = np.array([128] * 3, dtype=np.uint8)


def visualize_thresholds(
    img: RGBImage, conversion: ColorConversion, thresholds: ResidualThresholds
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """Visualizes the thresholded channels used for residual artifact detection.

    Args:
        img: Image of a tissue.
        conversion: Color conversion used for stain channel separation.
        thresholds: Thresholds for residual artifact detection.
            See the `StandardResidualThresholds` class for suggested threshold values.

    Returns:
        Tuple of four images showing which pixels are marked as artifacts
            in each of the thresholded channels.
            The channels are visualized in the following order:
                - Negative part of the first converted channel.
                - Negative part of the second converted channel.
                - Negative part of the third converted channel.
                - Positive part of the third converted channel.
    """
    c1, c2, c3 = np.asarray(
        convert_color(tile=img, conversion=conversion, keep_negative_values=True),
        dtype=np.float64,
    )[:, :, :, np.newaxis]

    th = thresholds

    c1_neg, c2_neg = -np.minimum(c1, 0), -np.minimum(c2, 0)
    c3_pos, c3_neg = np.maximum(c3, 0), -np.minimum(c3, 0)

    c1_neg_img = Image.fromarray(np.where(c1_neg >= th.c1_negative, img, GRAY))
    c2_neg_img = Image.fromarray(np.where(c2_neg >= th.c2_negative, img, GRAY))
    c3_neg_img = Image.fromarray(np.where(c3_neg >= th.c3_negative, img, GRAY))
    c3_pos_img = Image.fromarray(np.where(c3_pos >= th.c3_positive, img, GRAY))

    return c1_neg_img, c2_neg_img, c3_neg_img, c3_pos_img
