"""Code domain adapter for AI-generated code detection."""

from __future__ import annotations

import ast
import re
from collections import Counter

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import CalibratedDetectorMixin

try:
    import tree_sitter
    from tree_sitter import Parser
except ImportError:
    tree_sitter = None
    Parser = None


class CodeDetector(CalibratedDetectorMixin, BaseDetector):
    name = "code_detector"
    latency_tier = "medium"
    domains = ["code"]

    LOW_ENTROPY_VARIABLES = {
        "result",
        "data",
        "item",
        "value",
        "temp",
        "tmp",
        "arr",
        "list",
        "dict",
        "obj",
        "info",
        "res",
    }

    def __init__(self):
        self.nlp = None

    def _compute_ast_features(self, code: str, language: str) -> dict[str, float]:
        features: dict[str, float] = {}

        try:
            tree = ast.parse(code)
            functions = [
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ]
            features["function_count"] = float(len(functions))

            depths = []
            for func in functions:
                depth = self._get_function_depth(func)
                depths.append(depth)
            features["avg_function_depth"] = (
                float(sum(depths) / len(depths)) if depths else 0.0
            )
            features["max_function_depth"] = float(max(depths)) if depths else 0.0
        except SyntaxError:
            features["function_count"] = 0.0
            features["avg_function_depth"] = 0.0
            features["max_function_depth"] = 0.0

        variables = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", code)
        variable_counts = Counter(variables)
        low_entropy_count = sum(
            1 for v in variable_counts if v.lower() in self.LOW_ENTROPY_VARIABLES
        )
        features["low_entropy_variable_ratio"] = (
            float(low_entropy_count / len(variable_counts)) if variable_counts else 0.0
        )

        lines = code.split("\n")
        code_lines = [
            line
            for line in lines
            if line.strip()
            and not line.strip().startswith("#")
            and not line.strip().startswith("//")
        ]
        comment_lines = [
            line
            for line in lines
            if line.strip().startswith("#") or line.strip().startswith("//")
        ]
        features["comment_to_code_ratio"] = (
            float(len(comment_lines) / len(code_lines)) if code_lines else 0.0
        )

        return features

    def _get_function_depth(self, node: ast.AST) -> int:
        depth = 0
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While, ast.If, ast.With)):
                depth = max(depth, 1 + self._get_function_depth(child))
            else:
                depth = max(depth, self._get_function_depth(child))
        return depth

    def _compute_logical_complexity(self, code: str) -> float:
        branching_keywords = [
            "if",
            "elif",
            "else",
            "for",
            "while",
            "try",
            "except",
            "finally",
            "and",
            "or",
        ]
        count = sum(len(re.findall(rf"\b{kw}\b", code)) for kw in branching_keywords)
        lines = [line for line in code.split("\n") if line.strip()]
        return float(count / len(lines)) if lines else 0.0

    def _extract_features(self, text: str) -> list[float]:
        ast_features = self._compute_ast_features(text, "python")
        logical_complexity = self._compute_logical_complexity(text)
        return [
            ast_features.get("function_count", 0.0),
            ast_features.get("avg_function_depth", 0.0),
            ast_features.get("max_function_depth", 0.0),
            ast_features.get("low_entropy_variable_ratio", 0.0),
            ast_features.get("comment_to_code_ratio", 0.0),
            logical_complexity,
        ]

    def _extract_feature_names(self) -> list[str]:
        return [
            "function_count",
            "avg_function_depth",
            "max_function_depth",
            "low_entropy_variable_ratio",
            "comment_to_code_ratio",
            "logical_complexity",
        ]

    def detect(self, text: str) -> DetectorResult:
        ast_features = self._compute_ast_features(text, "python")
        logical_complexity = self._compute_logical_complexity(text)

        if ast_features.get("function_count", 0) == 0:
            return DetectorResult(
                score=0.5,
                confidence=0.3,
                metadata={"error": "Could not parse as code", "features": ast_features},
            )

        calibrated = self._get_calibrated_score(text)
        if calibrated is not None:
            score, confidence = calibrated
        else:
            low_entropy_score = ast_features.get("low_entropy_variable_ratio", 0)
            depth_score = min(1.0, ast_features.get("max_function_depth", 0) / 10)
            comment_ratio = ast_features.get("comment_to_code_ratio", 0)

            if low_entropy_score > 0.3:
                score = 0.7
                confidence = 0.7
            elif depth_score < 0.3 and comment_ratio < 0.1:
                score = 0.65
                confidence = 0.6
            elif logical_complexity < 0.2:
                score = 0.6
                confidence = 0.5
            else:
                score = 0.3
                confidence = 0.4

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                **ast_features,
                "logical_complexity": logical_complexity,
                "calibrated": calibrated is not None,
            },
        )


def register(registry) -> None:
    registry.register(CodeDetector)
