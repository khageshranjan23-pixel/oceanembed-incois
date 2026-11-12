<div align="center">

# 🌊 OceanEmbed-NIO

### Satellite-to-Subsurface Ocean Temperature Reconstruction
#### Smart India Hackathon 2026 · Problem Statement **SIH26066**
#### Ministry of Earth Sciences (MoES) / INCOIS · Team **VarunNetra** · RGIPT

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API_Server-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://docker.com/)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

**OceanEmbed-NIO** is a deep-learning research implementation that reconstructs 3D subsurface ocean temperature profiles (0–1000 m) from satellite sea-surface observations — eliminating the need for expensive ship-based measurements across the Indian Ocean.

[Problem Statement SIH26066](#-problem-statement) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Team](#-team)

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [What OceanEmbed Does](#-what-oceanembed-does)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Scientific Limitations](#%EF%B8%8F-scientific-limitations)
- [Documentation](#-documentation)
- [Team](#-team)

---

## 🎯 Problem Statement

> **SIH26066 — Ministry of Earth Sciences (MoES) / INCOIS**
>
> Ocean subsurface temperature data is critical for weather forecasting, climate modelling, monsoon prediction, and maritime safety. Traditional profiling via Argo floats and ship-based probes is expensive, spatially sparse, and temporally lagged. Satellite observations cover the ocean surface continuously but cannot directly sense below it.
>
> **Challenge:** Build a deep learning framework that accurately reconstructs 3D subsurface ocean temperature profiles (0–1000 m) from readily available satellite surface variables — bridging the gap between low-cost observation and high-value ocean insight across the North Indian Ocean.

---

## 🔭 What OceanEmbed Does

OceanEmbed trains a **neural encoder–decoder** on historical oceanographic data to learn the statistical relationship between sea-surface variables and the full vertical temperature structure below. After training on paired satellite + Argo/GLORYS data, the model predicts a complete 15-level temperature profile at any Indian Ocean grid point from satellite data alone.

```
Satellite Surface Observations
  (SST · SSH · SSS)
         │
         ▼
  ┌─────────────────────────────┐
  │   CNN Encoder               │
  │   + Bottleneck Attention    │   ◄── learns surface–subsurface mapping
  │   + Regime Gating           │       (Arabian Sea / Bay of Bengal)
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │   EOF + Nonlinear Residual  │
  │   Temperature Decoder       │   ◄── reconstructs 15 depth levels
  └─────────────┬───────────────┘
                │
                ▼
  3D Subsurface Temperature Profile
  (0 m → 1000 m · 15 depth levels)
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Data Acquisition Layer                        │
│  CMEMS satellite products · Argo GDAC float profiles · GLORYS        │
│  INCOIS analysed monthly grid                                        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                      Pre-processing Pipeline                         │
│  Spherical area-overlap remapping · daily aggregation · depth mask   │
│  Delayed-mode Argo QC · TEOS-10 Conservative Temperature conversion  │
│  Training-only climatology · vertical EOF fitting                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                     Neural Network Core (PyTorch)                    │
│                                                                      │
│  ┌──────────────┐   ┌────────────────┐   ┌───────────────────────┐  │
│  │ CNN Encoder  │──►│ Bottleneck +   │──►│ EOF + Nonlinear       │  │
│  │              │   │ Temporal Attn  │   │ Residual Decoder      │  │
│  └──────────────┘   └────────────────┘   └───────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ Regime Gating: Arabian Sea (lon < 78°E) | Bay of Bengal      │    │
│  │ Argo Observer: sparse observation correction (frozen backbone)│    │
│  │ Ensemble: 3-member inference + learned uncertainty scale     │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                   Evaluation & Calibration                           │
│  Held-out residual calibration · Profile RMSE · Bias · Correlation  │
│  Paired Argo float bootstrap · INCOIS grid comparison               │
│  Ridge baseline comparison                                           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                     Web Explorer (FastAPI + Vanilla JS)              │
│  Interactive map · Depth profile charts · Date comparisons           │
│  Transect viewer · Time-depth plots · Validation dashboard          │
│  Provenance & scientific data downloads · Docker deployment          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Core Language** | Python 3.12 |
| **Deep Learning** | PyTorch (CNN encoder, attention, EOF decoder) |
| **Data / Science** | NumPy, xarray, h5netcdf, TEOS-10 |
| **Argo Processing** | GDAC profile QC, delayed-mode correction |
| **API Server** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML / CSS / JavaScript (zero extra build step) |
| **Containerization** | Docker |
| **Testing** | pytest-based scientific transformation checks + synthetic smoke test |

---

## 📁 Project Structure

```
oceanembed-incois/
├── oceanembed/                   # Core Python package
│   ├── acquire.py                # Satellite & reanalysis data download
│   ├── prepare.py                # Raw NetCDF → training cube
│   ├── pack.py                   # Cube packaging & masking
│   ├── argo.py                   # Argo GDAC QC & TEOS-10 conversion
│   ├── fit.py                    # Climatology, normalization, EOF fitting
│   ├── model.py                  # CNN encoder + attention + decoder architecture
│   ├── observer.py               # Sparse Argo observation correction module
│   ├── predict.py                # Base / corrected / ensemble prediction
│   ├── evaluate.py               # Metrics, calibration, profile scoring
│   ├── compare.py                # Paired float bootstrap comparisons
│   ├── incois.py                 # INCOIS analysed-grid comparison
│   ├── baseline.py               # Ridge regression baseline
│   ├── advanced.py               # Optional adapter/GradNorm building blocks
│   ├── api.py                    # FastAPI application entry point
│   ├── data.py                   # Data loading utilities
│   └── inspect.py                # Diagnostic / inspection tools
├── web/                          # Frontend (served by FastAPI)
│   ├── index.html                # Explorer SPA shell
│   ├── app.js                    # Interactive map, charts, downloads
│   ├── style.css / topbar.css    # Styling
│   ├── incois.css                # INCOIS branding tokens
│   ├── guidance.js / events      # UI guidance system
│   └── vendor/                   # Vendored JS dependencies
├── docs/
│   ├── START_HERE_STEP_BY_STEP.md   # 36-step setup walkthrough ← start here
│   ├── IMPLEMENTATION_GUIDE.md      # Full setup, acquisition, training, deploy
│   ├── WEBSITE_SCIENTIFIC_WORKFLOW.md  # Web explorer usage & interpretation
│   ├── SCIENTIFIC_CONTRACT.md       # Assumptions, safeguards, hard limits
│   └── EXTENSIONS.md               # Higher-end experiments & production path
├── configs/                      # Reviewed download & ingestion templates
├── tests/                        # Scientific and unit tests
├── run_smoke.py                  # Synthetic end-to-end smoke test
├── Dockerfile                    # Container image definition
├── render.yaml                   # Render.com deployment config
├── pyproject.toml                # Package metadata & dependencies
└── requirements-serve.txt        # Minimal serving requirements
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12** (`python --version`)
- A compatible **PyTorch** build for your hardware (CPU / CUDA / ROCm)
- **Docker** (optional, for containerized serving)

### 1. Install

```bash
# Clone the repository
git clone https://github.com/khageshranjan23-pixel/oceanembed-incois.git
cd oceanembed-incois

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install PyTorch first (pick the right build for your hardware)
# See https://pytorch.org/get-started/locally/

# Install the package with test and data extras
python -m pip install -e ".[test,data]"
```

### 2. Run the Smoke Test

```bash
python run_smoke.py
```

> **Expected output:** A page labelled **`SYNTHETIC SMOKE TEST — NOT SCIENTIFIC RESULTS`**
> This validates the full pipeline end-to-end using synthetic fixtures — no real data or GPU required.

### 3. Start the API Server

```bash
python -m uvicorn oceanembed.api:app --reload
```

Open **http://127.0.0.1:8000** in your browser to access the interactive ocean explorer.

### 4. Full Real-Data Pipeline

> [!IMPORTANT]
> Real-data acquisition requires CMEMS credentials and GPU compute. Follow the **36-step guide** in [`docs/START_HERE_STEP_BY_STEP.md`](docs/START_HERE_STEP_BY_STEP.md) for the complete workflow.

```bash
# Download satellite & Argo data
python -m oceanembed.acquire --config configs/download.yaml

# Preprocess & pack training cube
python -m oceanembed.prepare && python -m oceanembed.pack

# Train the model
python -m oceanembed.model --train

# Evaluate
python -m oceanembed.evaluate

# Serve results
python -m uvicorn oceanembed.api:app
```

### Docker

```bash
docker build -t oceanembed-nio .
docker run -p 8000:8000 oceanembed-nio
```

---

## 🔌 API Reference

The FastAPI server exposes these endpoints at `http://localhost:8000`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Web explorer frontend |
| `/docs` | `GET` | Interactive Swagger UI |
| `/api/profile` | `GET` | Retrieve a temperature profile (`date`, `lat`, `lon`) |
| `/api/coverage` | `GET` | Supported dates, grid points, depth levels |
| `/api/metrics` | `GET` | Per-depth RMSE, bias, correlation by region/target |
| `/api/compare` | `GET` | GLORYS & Argo paired comparison data |
| `/api/incois` | `GET` | INCOIS monthly-grid comparison results |
| `/api/provenance` | `GET` | Data source hashes and processing lineage |

Full interactive docs are available at **`/docs`** (Swagger UI) and **`/redoc`** (ReDoc) when the server is running.

---

## ⚠️ Scientific Limitations

> [!IMPORTANT]
> OceanEmbed-NIO is a **research prototype** built for SIH 2026. It is **not** an operational forecasting tool and must not be used for safety-critical decisions.

> [!WARNING]
> The included smoke test uses **synthetic fixtures only**. Real provider downloads, GPU training runs, and real Argo validation have not been executed in this delivery. Do not present synthetic metrics as scientific results.

| Limitation | Detail |
|-----------|--------|
| **No pretrained weights** | No pretrained model checkpoint is included; weights require real-data training |
| **Provider adapters** | CMEMS download adapters must be configured from actual provider metadata |
| **Uncertainty** | Model uncertainty is a learned scale parameter — not a calibrated confidence interval |
| **Region labels** | Arabian Sea / Bay of Bengal split at 78°E longitude — a reporting partition, not a physical boundary |
| **BLT proxy** | Temperature-threshold crossings are used as mixed-layer/barrier-layer proxies; density-based diagnostics are unavailable |
| **Leakage risk** | Full-series climatology in training targets may inflate in-sample performance; held-out evaluation is required |
| **GPU requirement** | Real training requires CUDA-compatible GPU (prototype tested on RTX 2050 4 GB) |

See [`docs/SCIENTIFIC_CONTRACT.md`](docs/SCIENTIFIC_CONTRACT.md) for the full set of assumptions and safeguards.

---

## 📚 Documentation

| Document | Description |
|---------|-------------|
| [`docs/START_HERE_STEP_BY_STEP.md`](docs/START_HERE_STEP_BY_STEP.md) | **Start here** — 36-step numbered walkthrough from clone to deployed results |
| [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md) | Full setup, data acquisition, preprocessing, training, evaluation, deployment |
| [`docs/WEBSITE_SCIENTIFIC_WORKFLOW.md`](docs/WEBSITE_SCIENTIFIC_WORKFLOW.md) | Web explorer views, exact navigation, interpretation limits |
| [`docs/SCIENTIFIC_CONTRACT.md`](docs/SCIENTIFIC_CONTRACT.md) | Scientific assumptions, hard limits, and safeguards |
| [`docs/EXTENSIONS.md`](docs/EXTENSIONS.md) | Higher-end experiment extensions and production path |

---

## 👥 Team

**Team VarunNetra** · Rajiv Gandhi Institute of Petroleum Technology (RGIPT)

| Member | Role |
|--------|------|
| Khagesh Ranjan | Team Leader · Model Architecture |
| Sudipto Ghosh | Frontend & Integration |
| Ayush | Data Pipeline |
| Muskan | Evaluation & Metrics |
| Satyam | Backend & Deployment |
| Shivanshu | Scientific Validation |

*Smart India Hackathon 2026 · Problem Statement SIH26066*
*Ministry of Earth Sciences (MoES) / Indian National Centre for Ocean Information Services (INCOIS)*

---

<div align="center">

**OceanEmbed-NIO · Team VarunNetra · RGIPT · SIH 2026**

*Not an official Government of India product.*
*Synthetic smoke-test results are not scientific validation.*

</div>
