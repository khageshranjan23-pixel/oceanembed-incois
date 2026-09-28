# OceanEmbed-NIO implementation kit

Start with **docs/START_HERE_STEP_BY_STEP.md** (36 numbered steps), then **docs/WEBSITE_SCIENTIFIC_WORKFLOW.md**. The original expanded reference is **docs/IMPLEMENTATION_GUIDE.md**. It contains the exact setup sequence, data-access steps, configuration examples, commands, expected outputs, and stop conditions.

## What this is

A runnable research reference implementation aligned with the core of your v4 proposal:

- strict regular-grid source ingestion and unit checks;
- spherical area-overlap remapping, daily aggregation, and depth masking;
- delayed-mode Argo QC and TEOS-10 conversion;
- training-only climatology and vertical EOF fitting;
- CNN encoder, bottleneck temporal attention, optional spatial attention;
- EOF plus nonlinear residual temperature decoder;
- optional masked surface pretraining;
- frozen-backbone Argo observer and learned scale;
- three-member ensemble inference support;
- held-out residual calibration, profile metrics and paired float bootstrap;
- INCOIS monthly-grid comparison with an explicit temperature-definition gate;
- ridge baseline;
- precomputed-result API, linked maps/profiles, date differences, transects, time-depth plots, observational validation, provenance and scientific downloads;
- Docker serving and a synthetic end-to-end smoke test.

**This is not a pretrained model, a demonstrated accuracy result, or a fully validated ocean product.** Provider-specific adapters must be configured from actual downloaded metadata. Real provider downloads, GPU runs and real Argo performance have not been executed in this delivery. The included smoke test uses synthetic fixtures and labels them accordingly.

## First run

Use Python 3.12. From this folder, after activating a virtual environment and installing the correct PyTorch build:

```bash
python -m pip install -e ".[test,data]"
python run_smoke.py
python -m uvicorn oceanembed.api:app --reload
```

Open http://127.0.0.1:8000. The screen must say **SYNTHETIC SMOKE TEST — NOT SCIENTIFIC RESULTS**.

The full real-data process is in the guide. Do not point synthetic metrics at a scientific-results page.

## Files

| File/module | Purpose |
|---|---|
| `docs/IMPLEMENTATION_GUIDE.md` | Setup, acquisition, preprocessing, training, evaluation, deployment |
| `docs/EXTENSIONS.md` | Required work for higher-end experiments and production |
| `docs/SCIENTIFIC_CONTRACT.md` | Scientific assumptions, safeguards, limits |
| `configs/` | Reviewed download and ingestion templates |
| `oceanembed/science.py` | Regridding, interpolation, collocation, metrics |
| `oceanembed/prepare.py`, `pack.py` | Raw reviewed NetCDF to training cube |
| `oceanembed/argo.py` | Raw core GDAC profile QC and temperature conversion |
| `oceanembed/fit.py` | Training-only normalization, climatology, EOF |
| `oceanembed/model.py`, `train.py` | Core architecture, SSL and teacher training |
| `oceanembed/observer.py` | Sparse observation correction |
| `oceanembed/predict.py` | Base/corrected/ensemble predictions |
| `oceanembed/evaluate.py`, `compare.py` | Evaluation, calibration, paired comparisons |
| `oceanembed/incois.py` | INCOIS analysed-grid comparison |
| `oceanembed/baseline.py` | Ridge baseline |
| `oceanembed/advanced.py` | Optional adapter/GradNorm building blocks, disabled by default |
| `oceanembed/api.py`, `web/index.html` | Small dependency-light public explorer |
| `tests/` | Scientific transformation and code checks |

The frontend is plain HTML/CSS/JavaScript served by FastAPI, deliberately making the first installation one service. Next.js/TypeScript is an optional UI replacement; it is not secretly required by this kit.

## Website update

The research website now includes calibrated profile intervals, Argo overlays, date comparisons, transects, time-depth plots and a validation dashboard. Read docs/WEBSITE_SCIENTIFIC_WORKFLOW.md for supported views, exact clicks and interpretation limits.
