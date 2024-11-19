import numpy as np

from rationai.qc.typing import QcValues, RGBImage
from rationai.staining import ColorConversion, convert_color


def example_separation_function(img: RGBImage, conversion: ColorConversion) -> QcValues:
    """TODO: precise documentation."""
    result: QcValues = {}

    h_channel, e_channel, r_channel = convert_color(img, conversion)

    result["channel_0"] = h_channel
    result["channel_1"] = e_channel
    result["channel_2"] = r_channel

    result["mean_residual_value"] = np.mean(r_channel)

    return result
