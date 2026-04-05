"""FastAPI REST API server for Provenance AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from provenance import Provenance
from provenance.core.registry import get_registry

_provenance_cache: Provenance | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provenance_cache
    registry = get_registry()
    registry.load_entry_points()
    _provenance_cache = Provenance()
    yield


app = FastAPI(
    title="Provenance AI",
    description="Modular AI text detection API",
    version="0.1.0",
    lifespan=lifespan,
)


class DetectRequest(BaseModel):
    text: str
    detectors: list[str] | None = None
    ensemble_strategy: Literal["weighted_average", "stacking", "uncertainty_aware"] = (
        "weighted_average"
    )
    domain: str | None = None


class DetectResponse(BaseModel):
    score: float
    label: Literal["human", "ai", "mixed", "uncertain"]
    confidence: float
    detector_scores: dict | None = None
    heatmap: list | None = None


class BatchDetectRequest(BaseModel):
    texts: list[str]
    detectors: list[str] | None = None
    ensemble_strategy: Literal["weighted_average", "stacking", "uncertainty_aware"] = (
        "weighted_average"
    )


def _get_provenance(
    detectors: list[str] | None = None,
    ensemble_strategy: str = "weighted_average",
    domain: str | None = None,
) -> Provenance:
    global _provenance_cache

    if (
        detectors is None
        and ensemble_strategy == "weighted_average"
        and domain is None
        and _provenance_cache is not None
    ):
        return _provenance_cache

    return Provenance(
        detectors=detectors,
        ensemble_strategy=ensemble_strategy,  # type: ignore[arg-type]
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest) -> DetectResponse:
    if not request.text:
        raise HTTPException(status_code=400, detail="text is required")

    prov = _get_provenance(
        detectors=request.detectors,
        ensemble_strategy=request.ensemble_strategy,
        domain=request.domain,
    )
    result = prov.detect(request.text)

    return DetectResponse(
        score=result.score,
        label=result.label,
        confidence=result.confidence,
        detector_scores={
            name: {
                "score": dr.score,
                "confidence": dr.confidence,
            }
            for name, dr in result.detector_scores.items()
        },
        heatmap=[(ts.token, ts.score) for ts in result.heatmap[:50]],
    )


@app.post("/batch", response_model=list[DetectResponse])
async def batch_detect(request: BatchDetectRequest) -> list[DetectResponse]:
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list is required")

    prov = _get_provenance(
        detectors=request.detectors,
        ensemble_strategy=request.ensemble_strategy,
    )
    results = []
    for text in request.texts:
        result = prov.detect(text)
        results.append(
            DetectResponse(
                score=result.score,
                label=result.label,
                confidence=result.confidence,
                detector_scores={
                    name: {
                        "score": dr.score,
                        "confidence": dr.confidence,
                    }
                    for name, dr in result.detector_scores.items()
                },
                heatmap=[(ts.token, ts.score) for ts in result.heatmap[:50]],
            )
        )

    return results


@app.get("/detectors")
async def list_detectors() -> dict:
    registry = get_registry()
    registry.load_entry_points()
    detectors = registry.list_detectors()

    return {
        "detectors": [
            {
                "name": d.name,
                "latency_tier": d.latency_tier,
                "domains": d.domains,
            }
            for d in detectors
        ]
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "version": "0.1.0"}


def run_server(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
