"""Marker detection (template matching) and alignment-transform fitting."""
from .aligner import (
    AlignmentError,
    AlignmentResult,
    MarkerAligner,
    MarkMeasurement,
    MarkSpec,
    default_cross_factory,
)
from .fitting import MODELS, FitResult, fit_transform
from .matching import (
    MatchResult,
    cross_template,
    match_template,
    ncc_map,
    preprocess,
    square_template,
    template_from_image,
)

__all__ = [
    "AlignmentError",
    "AlignmentResult",
    "MarkerAligner",
    "MarkMeasurement",
    "MarkSpec",
    "default_cross_factory",
    "MODELS",
    "FitResult",
    "fit_transform",
    "MatchResult",
    "cross_template",
    "match_template",
    "ncc_map",
    "preprocess",
    "square_template",
    "template_from_image",
]
