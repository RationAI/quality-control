import numpy as np

from rationai.qc.typing import QcValues, RGBImage
from rationai.staining import ColorConversion, convert_color


def example_separation_function(img: RGBImage, conversion: ColorConversion) -> QcValues:
    result: QcValues = {}

    h_channel, e_channel, r_channel = convert_color(img, conversion)

    result["H_channel"] = h_channel
    result["E_channel"] = e_channel
    result["R_channel"] = r_channel

    result["mean_residual_value"] = np.mean(r_channel)

    return result
