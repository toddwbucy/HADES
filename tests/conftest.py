"""Test configuration for filtering noisy third-party warnings."""
import warnings

warnings.filterwarnings(
    "ignore",
    message="Use `ConversionResult` instead.",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPy.* has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"\[Errno 13\] Permission denied.  joblib will operate in serial mode",
    category=UserWarning,
)
