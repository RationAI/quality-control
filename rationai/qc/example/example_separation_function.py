import numpy as np
import pyvips

from rationai.qc.typing import QcValues
from rationai.staining import ColorConversion, convert_color


def example_separation_function(
    img: pyvips.Image, conversion: ColorConversion
) -> QcValues:
    result: QcValues = {}
    numpy_img = img.numpy()

    h_channel, e_channel, r_channel = convert_color(numpy_img, conversion)

    result["H_channel"] = pyvips.Image.new_from_array(h_channel)
    result["E_channel"] = pyvips.Image.new_from_array(e_channel)
    result["R_channel"] = pyvips.Image.new_from_array(r_channel)

    result["mean_residual_value"] = np.mean(r_channel)

    return result
