from typing import Final

from rationai.qc.typing import ResidualThresholds


class StandardResidualThresholds:
    """Collection of standard threshold values for residual artifact detection.

    This class provides predefined residual artifact threshold values
    for commonly used staining conversions. These thresholds should provide
    a reasonable middle ground between false positives and false negatives,
    but they could require adjustments depending on the specific use case.
    """

    HER: Final[ResidualThresholds] = ResidualThresholds(
        c1_negative=0.02,
        c2_negative=0.02,
        c3_negative=0.04,
        c3_positive=0.005,
    )
    """Suggested threshold values when using the `RGB2HER` conversion
    for residual artifact detection.
    """

    HDR: Final[ResidualThresholds] = ResidualThresholds(
        c1_negative=0.09,
        c2_negative=0.001,
        c3_negative=0.01,
        c3_positive=0.09,
    )
    """Suggested threshold values when using the `RGB2HDR` conversion
    for residual artifact detection.
    """
