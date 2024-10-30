from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


RGBImage: TypeAlias = NDArray[np.uint8]
QcValues: TypeAlias = dict[str, Any]
