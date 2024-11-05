from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


RGBImage: TypeAlias = NDArray[np.uint8]
"""
Three channel RGB image represented by a numpy array.
"""

Stain: TypeAlias = NDArray[np.float64]
"""
Single stain vector (of 3 values) represented by a numpy array.
"""

QcValues: TypeAlias = dict[str, Any]
"""
Dictionary with values computed by a QC function.
"""
