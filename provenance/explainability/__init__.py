"""Explainability utilities for AI text detection."""

from provenance.explainability.heatmaps import (
    compute_sentence_scores,
    format_heatmap_html,
    generate_feature_importance_report,
    generate_token_heatmap,
)

__all__ = [
    "compute_sentence_scores",
    "format_heatmap_html",
    "generate_feature_importance_report",
    "generate_token_heatmap",
]
