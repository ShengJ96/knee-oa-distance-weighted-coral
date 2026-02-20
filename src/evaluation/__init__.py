"""Model evaluation and comparison utilities."""

try:  # pragma: no cover - optional import guards for partially implemented modules
    from .metrics import comprehensive_evaluation, medical_specific_metrics  # type: ignore
except ImportError:  # pragma: no cover
    comprehensive_evaluation = medical_specific_metrics = None  # type: ignore

try:
    from .comparison import ModelComparator  # type: ignore
except ImportError:  # pragma: no cover
    ModelComparator = None  # type: ignore

try:
    from .visualization import ResultVisualizer  # type: ignore
except ImportError:  # pragma: no cover
    ResultVisualizer = None  # type: ignore

try:
    from .explainability import ModelExplainer  # type: ignore
except ImportError:  # pragma: no cover
    ModelExplainer = None  # type: ignore

__all__ = [
    name
    for name in (
        "comprehensive_evaluation",
        "medical_specific_metrics",
        "ModelComparator",
        "ResultVisualizer",
        "ModelExplainer",
    )
    if globals().get(name) is not None
]
