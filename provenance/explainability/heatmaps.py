"""Explainability utilities for AI text detection."""

from __future__ import annotations

from provenance.core.base import TokenScore


def generate_token_heatmap(
    text: str,
    scores: list[float] | None = None,
) -> list[TokenScore]:
    tokens = text.split()

    if scores is not None:
        min_len = min(len(scores), len(tokens))
        return [
            TokenScore(token=token, score=score)
            for token, score in zip(tokens[:min_len], scores[:min_len], strict=False)
        ]

    return [TokenScore(token=t, score=0.5) for t in tokens]


def generate_feature_importance_report(
    feature_vector: dict[str, float],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    sorted_features = sorted(
        feature_vector.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return sorted_features[:top_n]


def compute_sentence_scores(
    tokens: list[TokenScore],
    sentence_boundaries: list[tuple[int, int]],
) -> list[float]:
    sentence_scores = []

    for start, end in sentence_boundaries:
        if start >= len(tokens) or end > len(tokens):
            sentence_scores.append(0.5)
            continue

        segment = tokens[start:end]
        if not segment:
            sentence_scores.append(0.5)
            continue

        avg_score = sum(ts.score for ts in segment) / len(segment)
        sentence_scores.append(avg_score)

    return sentence_scores


def format_heatmap_html(heatmap: list[TokenScore]) -> str:
    html_parts = ['<div class="sentinel-heatmap">']

    for token_score in heatmap:
        color = _score_to_color(token_score.score)
        html_parts.append(
            f'<span style="background-color:{color}" title="{token_score.score:.2f}">'
            f"{token_score.token}</span>"
        )

    html_parts.append("</div>")
    return "\n".join(html_parts)


def _score_to_color(score: float) -> str:
    if score < 0.3:
        r = int(200 * (0.3 - score) / 0.3)
        return f"rgb({r}, 255, {r})"
    elif score > 0.7:
        intensity = min(1.0, (score - 0.7) / 0.3)
        r = int(255 * intensity)
        return f"rgb(255, {int(255 * (1 - intensity))}, {int(255 * (1 - intensity))})"
    else:
        gray = 230
        return f"rgb({gray}, {gray}, {gray})"
