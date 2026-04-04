"""Tests for sentinel.explainability.heatmaps module."""

from provenance.core.base import TokenScore
from provenance.explainability.heatmaps import (
    _score_to_color,
    compute_sentence_scores,
    format_heatmap_html,
    generate_feature_importance_report,
    generate_token_heatmap,
)


class TestGenerateTokenHeatmap:
    def test_uniform_heatmap(self):
        text = "Hello world this is a test"
        heatmap = generate_token_heatmap(text)
        assert len(heatmap) == len(text.split())
        assert all(ts.score == 0.5 for ts in heatmap)

    def test_heatmap_with_scores(self):
        text = "Hello world this is"
        scores = [0.2, 0.4, 0.6, 0.8]
        heatmap = generate_token_heatmap(text, scores=scores)
        assert len(heatmap) == 4
        assert [ts.score for ts in heatmap] == scores

    def test_heatmap_mismatched_length(self):
        text = "Hello world this is a test sentence"
        scores = [0.5, 0.5]
        heatmap = generate_token_heatmap(text, scores=scores)
        assert len(heatmap) == 2

    def test_token_score_creation(self):
        text = "one two three"
        heatmap = generate_token_heatmap(text)
        tokens = [ts.token for ts in heatmap]
        assert tokens == ["one", "two", "three"]


class TestComputeSentenceScores:
    def test_basic_sentence_scores(self):
        tokens = [
            TokenScore(token="Hello", score=0.2),
            TokenScore(token="world.", score=0.2),
            TokenScore(token="This", score=0.8),
            TokenScore(token="is", score=0.8),
            TokenScore(token="a", score=0.8),
            TokenScore(token="test.", score=0.8),
        ]
        boundaries = [(0, 2), (2, 6)]
        scores = compute_sentence_scores(tokens, boundaries)
        assert len(scores) == 2
        assert scores[0] == 0.2
        assert scores[1] == 0.8

    def test_empty_tokens(self):
        scores = compute_sentence_scores([], [(0, 0)])
        assert scores == [0.5]

    def test_out_of_bounds_boundary(self):
        tokens = [TokenScore(token="test", score=0.5)]
        scores = compute_sentence_scores(tokens, [(0, 10)])
        assert scores == [0.5]


class TestGenerateFeatureImportanceReport:
    def test_sorted_by_absolute_value(self):
        features = {
            "feature_a": 0.1,
            "feature_b": -0.5,
            "feature_c": 0.3,
        }
        report = generate_feature_importance_report(features, top_n=2)
        assert len(report) == 2
        assert report[0][0] == "feature_b"
        assert report[1][0] == "feature_c"

    def test_top_n_limit(self):
        features = {f"f{i}": i * 0.1 for i in range(20)}
        report = generate_feature_importance_report(features, top_n=5)
        assert len(report) == 5


class TestScoreToColor:
    def test_low_score_green(self):
        color = _score_to_color(0.1)
        assert "rgb" in color

    def test_high_score_red(self):
        color = _score_to_color(0.9)
        assert "rgb" in color

    def test_mid_score_gray(self):
        color = _score_to_color(0.5)
        assert "rgb" in color


class TestFormatHeatmapHtml:
    def test_html_output(self):
        heatmap = [
            TokenScore(token="Hello", score=0.2),
            TokenScore(token="world", score=0.8),
        ]
        html = format_heatmap_html(heatmap)
        assert "<div" in html
        assert "Hello" in html
        assert "world" in html
        assert "</div>" in html
