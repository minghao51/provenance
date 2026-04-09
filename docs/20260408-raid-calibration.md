# RAID Calibration Notes

## Summary

This document records the real-data calibration run completed on 2026-04-08 using the built-in `raid` benchmark dataset through `provenance.calibrate`.

The curated runtime config produced from this run is:

- [`provenance.calibrated.raid.yaml`](../provenance.calibrated.raid.yaml)

The saved model artifacts and summary are:

- [`calibration_models/repetition_raid.pkl`](../calibration_models/repetition_raid.pkl)
- [`calibration_models/burstiness_raid.pkl`](../calibration_models/burstiness_raid.pkl)
- [`calibration_models/surprisal_raid.pkl`](../calibration_models/surprisal_raid.pkl)
- [`calibration_models/curvature_raid.pkl`](../calibration_models/curvature_raid.pkl)
- [`calibration_models/raid_calibration_summary.json`](../calibration_models/raid_calibration_summary.json)

## Selected Models

Only models that improved held-out RAID performance were included in the curated config.

| Detector | Sample Limit | AUROC Before | AUROC After | F1 Before | F1 After | Selected |
|----------|--------------|--------------|-------------|-----------|----------|----------|
| repetition | 200 | 0.6324 | 0.8235 | 0.3000 | 0.9189 | yes |
| burstiness | 200 | 0.6103 | 0.9069 | 0.6667 | 0.8889 | yes |
| surprisal_diveye | 200 | 0.6495 | 0.8775 | 0.6667 | 0.9565 | yes |
| curvature_detectgpt | 150 | 0.3846 | 0.7788 | 0.9286 | 0.9412 | yes |
| entropy | 200 | 0.5000 | 0.3848 | 0.9189 | 0.9189 | no |

`entropy` was excluded because calibration reduced AUROC on the held-out split and did not improve F1.

## How To Use It

CLI:

```bash
uv run provenance detect "Your text here" --config /Users/minghao/Desktop/personal/provenance/provenance.calibrated.raid.yaml
```

Python:

```python
from provenance import Provenance

provenance = Provenance(
    config="/Users/minghao/Desktop/personal/provenance/provenance.calibrated.raid.yaml"
)
result = provenance.detect("Your text here")
```

The config uses explicit `detector_calibration_paths`, so only the selected detectors load saved calibration models.

## Calibration Commands

Representative commands used for the completed run:

```bash
uv run python -m provenance.calibrate train --detector repetition --dataset raid --limit 200 --method isotonic --cv 5 --output-dir calibration_models
uv run python -m provenance.calibrate train --detector burstiness --dataset raid --limit 200 --method isotonic --cv 5 --output-dir calibration_models
uv run python -m provenance.calibrate train --detector surprisal --dataset raid --limit 200 --method isotonic --cv 5 --output-dir calibration_models
uv run python -m provenance.calibrate train --detector curvature --dataset raid --limit 150 --method isotonic --cv 5 --output-dir calibration_models
```

## Notes

- `HC3` was not used for this run because the installed `datasets` client rejects its legacy dataset script.
- The calibration workflow now disables automatic model loading during training runs so saved artifacts do not contaminate baseline measurements.
- `datasets` is now declared in the project dependencies used for calibration workflows.
