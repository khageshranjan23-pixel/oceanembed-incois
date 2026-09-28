# OceanEmbed-NIO: expanded implementation reference

For the latest linear 36-step sequence, read START_HERE_STEP_BY_STEP.md first. The updated six-view research website is documented in WEBSITE_SCIENTIFIC_WORKFLOW.md.

Read this in order. Commands run from the extracted `oceanembed-kit` directory unless stated otherwise. Copy commands one at a time and stop when a command fails. File names beginning with `REPLACE_` are explicit provider-specific configuration fields, not valid downloadable IDs.

## 0. Scope and what has actually been tested

This kit implements a working core research pipeline and a small result explorer. Its synthetic workflow exercises statistics fitting, the network, training, observation correction, prediction, calibration, evaluation, and API responses. It does not contain a trained ocean model or real accuracy results. Check `SMOKE_RESULT.json` and `TESTED_ENVIRONMENT.json` for this delivery's executed checks.

No one can supply a scientifically defensible universal "highest accuracy" checkpoint without training and evaluating your chosen product versions. The code prevents several common mistakes, but actual product metadata, QC, grid geometry and sampling must be reviewed. Read the scientific contract before changing those checks.

The reference model is intentionally smaller than the final 15–25M-parameter ambition. Start here, measure, and add capacity only when validation supports it. Optional research modules and production work are explained in `EXTENSIONS.md`.

## 1. Full technology stack

| Layer | Exact stack in this kit | Why |
|---|---|---|
| Editor | VS Code + Microsoft Python extension | One place to edit/run/debug |
| Language | Python 3.12 | Supported scientific ecosystem |
| Environments | `venv` + pip | Isolated and reproducible |
| Tensor framework | PyTorch | CNN, attention, autograd, GPU |
| Numeric processing | NumPy + SciPy | Arrays, remapping, statistics |
| Scientific files | xarray + netCDF4; Dask for multifile reads | CF-style NetCDF handling |
| Observation tables | pandas CSV initially | Inspectable profile tables |
| Thermodynamics | GSW | TEOS-10 salinity/temperature/depth conversions |
| Baseline | scikit-learn Ridge | Reproducible statistical baseline |
| Copernicus access | Official `copernicusmarine` | Regional subsetting |
| NASA access | Official `earthaccess` | Authentication/search/download |
| Raw Argo | Official GDAC index + HTTPS files | Original profile flags and provenance |
| Training storage | Memory-mapped `.npy` arrays | Avoid repeatedly loading the entire cube |
| Output storage | Compressed `.npz` per day | Small portable demo exports |
| Backend | FastAPI + Uvicorn | CPU serving of saved outputs |
| Frontend | HTML/CSS/JavaScript + locally bundled Plotly | No Node/npm dependency for first working demo |
| Tests | pytest + FastAPI TestClient | Scientific invariants and endpoint checks |
| Deployment | Docker or a Python web service | Separate serving from training |
| Optional scale-up | Zarr/object storage/CDN/Next.js | See extension plan |

No SQL database is required for the first version. Arrays do not belong as millions of rows in PostgreSQL. Add a small database later for run/catalogue metadata if needed.

## 2. Install and open the project: click by click

### Windows

1. Download Python 3.12 from https://www.python.org/downloads/ . Choose the installer matching your machine. If the installer offers **Add python.exe to PATH**, enable it.
2. Install VS Code from https://code.visualstudio.com/ .
3. Download this ZIP. Right-click it → **Extract All** → choose a short path, for example `C:\Projects`.
4. Open VS Code → **File → Open Folder** → select `C:\Projects\oceanembed-kit`.
5. Click **Extensions** in the left bar. Search **Python** and install the Microsoft extension. Pylance is useful too.
6. Choose **Terminal → New Terminal**. For easiest activation, use the terminal dropdown → **Command Prompt**.
7. Run:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
```

8. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the interpreter inside `.venv`.
9. Keep the terminal in the folder containing `pyproject.toml` and `run_smoke.py`.

If you prefer PowerShell, use `.venv\Scripts\Activate.ps1`. If script execution is blocked, use Command Prompt or invoke `.venv\Scripts\python.exe` directly; you do not need to weaken machine-wide execution policy.

### macOS/Linux

1. Install Python 3.12 and VS Code.
2. Extract the ZIP in a project directory.
3. In VS Code choose **File → Open Folder** and open `oceanembed-kit`.
4. Install the Microsoft Python extension.
5. Open **Terminal → New Terminal** and run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
```

6. Open the Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Linux), select **Python: Select Interpreter**, and choose `.venv`.

### Choose the PyTorch build correctly

Open https://docs.pytorch.org/get-started/locally/ . Select your operating system, Pip, Python, and the compute platform supported by your hardware. Copy the generated command into the activated terminal. CPU is enough for the smoke test. For real training, use a suitable NVIDIA GPU machine and its compatible CUDA wheel. This kit defaults to CUDA when available, otherwise CPU. Apple MPS is not part of the tested path; use CPU for the smoke test and a Linux GPU for long runs.

Then:

```bash
python -m pip install -e ".[test,data]"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pytest -q tests
```

The first command installs the local package and dependencies. CPU PyTorch can be installed first with the official selector to avoid inadvertently downloading an unnecessary CUDA build. Initial installation can be large.

After the installation works, record your exact environment:

```bash
python -m pip freeze > requirements-lock-local.txt
```

Do not assume the included tested-version report is a portable CUDA lockfile for every operating system.

## 3. Run the complete synthetic smoke test

```bash
python run_smoke.py
```

It runs the tests, creates synthetic data, fits climatology/EOF, trains a tiny model, fits the observer, exports maps, calibrates intervals, evaluates synthetic holdouts and checks the API. It creates `data/smoke`, `runs/smoke`, and `outputs/smoke`.

Expected final message:

```text
SMOKE TEST PASSED. Start: python -m uvicorn oceanembed.api:app --reload
```

Start the explorer:

```bash
python -m uvicorn oceanembed.api:app --reload
```

1. Open http://127.0.0.1:8000 in your browser.
2. Confirm the orange banner says **SYNTHETIC SMOKE TEST — NOT SCIENTIFIC RESULTS**.
3. Select a date.
4. Select 100 m in the depth dropdown.
5. Click **Load field**.
6. Click the map: the profile updates on the right.
7. Scroll down to see synthetic metrics.
8. Open http://127.0.0.1:8000/docs to inspect API endpoints.
9. Stop the server with `Ctrl+C` in the terminal.

This confirms plumbing only. Low synthetic error does not establish real ocean skill. Do not use these screenshots as observational validation.

## 4. Create the real-data workspace

Create these folders using VS Code's Explorer → New Folder, or your file manager:

```text
data/raw/
data/harmonized/
data/cube/
data/argo/
runs/core/
outputs/
```

Start with **June 2020**, plus preceding days when needed for seven-day history. A 30-day pilot is for debugging, not a representative final training set. The final training data should span 2016–2021; selection/calibration use 2022 and final testing uses 2023–2025.

The supplied `train` command deliberately requires both training and 2022 validation windows. To do a real-data pilot training run, also prepare a small 2022 pilot period; do not relabel 2020 as 2022 to bypass the check.

## 5. Create accounts and retrieve data

### 5A. Copernicus Marine: SST, SSS, SLA, GLORYS

1. Open https://data.marine.copernicus.eu/ .
2. Register/sign in through the service's account controls.
3. Search the catalogue using the product names below.
4. Open the product page → read **Description**, **Documentation**, and **Data access/Services**. Labels can change; use the documented function if the wording differs.
5. Open the download/subset interface. Select the DAILY dataset, not a monthly product with a similar title.
6. Set the regional selection to roughly 44.5°E–105.5°E and 4.5°N–30.5°N. This halo supplies edge cells for the 0.25° target grid.
7. Set the pilot dates. Download only necessary variables.
8. Save the generated Python command or dataset ID. Record the product version and temperature/units definition.
9. Download a small NetCDF sample and inspect it before continuing.

| Quantity | Product to locate | Critical check |
|---|---|---|
| SST | OSTIA daily foundation SST, appropriate reprocessed record | Often Kelvin; verify attributes |
| SSS | `MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013` or documented approved alternative | Blended satellite/in-situ lineage; uncertainty/QC |
| SLA | DUACS daily sea-level anomaly product | Choose SLA consistently; do not mix SLA and absolute dynamic topography |
| Teacher | `GLOBAL_MULTIYEAR_PHY_001_030` | Potential temperature; native levels bracketing 1000 m |

A documented GLORYS daily dataset ID is present in `configs/download-glorys.json`. Verify the current catalogue entry before using it.

Authenticate using the official tool:

```bash
copernicusmarine login
```

Enter your account details when prompted. Do not paste credentials into source code or commit credential files.

Open `configs/download-glorys.json`. Check the dataset and variables against the catalogue, then change `"reviewed": false` to `true`. Run:

```bash
python -m oceanembed.acquire --config configs/download-glorys.json
```

For SST/SSS/SLA, duplicate that JSON file and replace `dataset_id`, `variables`, output folder and depth arguments with values from the relevant catalogue page. Remove `minimum_depth` and `maximum_depth` for products with no depth axis. Copy the exact dataset variable names; a catalogue product ID is not a dataset ID.

Download training GLORYS through 2021 and teacher-validation data for 2022. Do not download/pack 2023–2025 GLORYS as training targets. You need surface inputs, but not teacher targets, for final test inference.

### 5B. NASA Earthdata: CCMP and OSCAR

1. Register/sign in at https://urs.earthdata.nasa.gov/ .
2. Open https://search.earthdata.nasa.gov/ or the PO.DAAC product page.
3. Search `CCMP_WINDS_10M6HR_L4_V3.1`.
4. Confirm six-hourly U/V, the version, date coverage and documentation.
5. Apply the North Indian Ocean bounding box and June 2020 time range.
6. Follow the download/authentication instructions. Accept required provider app access if the service asks.
7. For code access, review `configs/download-ccmp.json`, set `reviewed=true`, then run:

```bash
python -m oceanembed.acquire --config configs/download-ccmp.json
```

8. For OSCAR, duplicate the Earthdata config and replace `short_name` with the chosen documented OSCAR v2.0 collection. Use final historical data for the historical experiment.
9. Keep U and V components. Do not replace them with speed alone.

Earthdata's bounding-box search selects intersecting granules; it does not guarantee the downloaded files are spatially subset. Some daily global files can be large. Inspect one day's file size and use provider subsetting/streaming where supported before downloading years.

The downloader has a granule-count cap to prevent accidental large requests. Increase it deliberately only after confirming the search results.

### 5C. Bathymetry

Download a documented bathymetry grid such as GEBCO from its official provider. Extract a rectilinear regional subset with the same halo. Preserve source citation/version. The ingest code expects **positive-down ocean depth, land missing**. If the source is elevation positive upwards, configure `scale=-1` and verify the resulting ocean values. Source geometry must be rectilinear for this code; do not feed a curvilinear grid into it.

### 5D. Raw Argo

The downloader uses the official core-profile GDAC index, filters time/region, and downloads original files.

Pilot:

```bash
python -m oceanembed.download_argo --start 2020-06-01 --end 2020-06-30 --out data/raw/argo-pilot --limit 100
```

The `--limit` is deliberately small. For a complete benchmark, use a limit larger than the reported match count, or download in month/year batches. A sorted first-100 subset is not representative and must not be used for the headline scientific score.

For the full period:

```bash
python -m oceanembed.download_argo --start 2016-01-01 --end 2025-12-31 --out data/raw/argo --limit 1000000
```

Run this only after inspecting profile counts and transfer sizes. The script preserves the filtered index, request, and source directory structure. Network/provider errors stop the run after bounded retries; reruns skip already downloaded files.

### 5E. INCOIS gridded Argo

1. Open https://erddap.incois.gov.in/erddap/griddap/incois_argo_mnt_VAM.html .
2. Select `TEMP` and `SAL`.
3. Select the intended dates and geographic overlap with the project.
4. Retain native depths between 5 and 1000 m.
5. Choose NetCDF output and download.
6. Save as `data/raw/incois.nc`.
7. Read the provider's analysis/temperature documentation. Confirm whether `TEMP` is potential or in-situ temperature and preserve the reference. The current generic `Temperature` attribute alone does not resolve this. The comparison command intentionally requires your explicit definition.

## 6. Inspect each product before configuring ingestion

```bash
python -m oceanembed.inspect data/raw/glorys-pilot/glorys-pilot.nc
```

Repeat for one SST, SSS, SLA, current, wind and bathymetry file. Record:

- Actual variable names and dimensions.
- Exact unit strings.
- Latitude/longitude ordering and coordinate names.
- Time meaning and number of samples per day.
- Fill values (xarray normally decodes CF fill values to NaN).
- QC variables and flag meanings.
- Depth values.
- Product version, processing history, and source lineage.

Open `configs/ingest-template.json`. Save a copy as `configs/ingest.json`. Configure each source entry.

Example only, IF inspection confirms these attributes:

```json
{
  "reviewed": true,
  "files": "data/raw/sst/*.nc",
  "multi_file": true,
  "variable": "analysed_sst",
  "expected_units": "kelvin",
  "rename": {"lat": "latitude", "lon": "longitude"},
  "scale": 1,
  "offset": -273.15,
  "min_samples_per_day": 1,
  "min_support": 0.8,
  "qc": []
}
```

The exact Kelvin string may be `K`, `kelvin`, or another metadata convention. Set `expected_units` to the actual string and apply the correct conversion. If SST is already Celsius, use offset 0. Do not keep -273.15 merely because it appears in the template.

`qc: []` is only acceptable after verifying the product's existing valid-data masks and deciding no additional flag filter is needed. To accept particular QC values:

```json
"qc": [{"variable": "quality_level", "allowed": [4, 5]}]
```

That example is not a universal product rule. Read that product's flag meanings. Argo flag numbers are not satellite flag numbers. Bit-field QC requires decoding before ingestion; equality comparisons are not a correct bit-mask decoder.

For dimensions such as a single surface depth, use:

```json
"select_index": {"depth": 0}
```

Do not remove the teacher depth dimension. Rename it to `depth` if necessary.

The `files` field accepts one filename by default. Set `multi_file=true` for a glob covering multiple compatible files. It requires Dask. All files must share reviewed variables, units, version semantics and coordinate geometry. Split version changes into separate reviewed ingestions.

For CCMP set `min_samples_per_day=4` for complete daily means. This sacrifices incomplete days rather than silently treating three observations as a complete four-time average. Model masks can handle the resulting missing input.

For teacher targets, verify Celsius potential temperature; leave `offset=0` if already Celsius. For sea level convert centimetres to metres only if metadata says centimetres. Vectors must already be eastward/northward; rotate grid-relative vectors with a product-specific routine before this ingest step.

## 7. Harmonize and inspect the pilot

```bash
python -m oceanembed.prepare --spec configs/ingest.json
```

Expected outputs:

```text
data/harmonized/bathymetry.nc
data/harmonized/sst/2020-06-01.nc
data/harmonized/sss/2020-06-01.nc
...
data/harmonized/teacher/2020-06-01.nc
```

The code calculates cell-edge overlaps on a sphere for regular rectilinear cell-centred fields. It is not a generic mesh remapper. Curvilinear grids, unusual cell boundaries and swaths need their own reviewed adapter.

`support` means valid fraction of total target-cell area under this representation. It is not a count of independent satellite observations, not confidence, and not full physical ocean-area fraction. This conservative support policy can exclude coastal cells; document it.

For the 0 m teacher label, the code uses the shallowest model level. Other levels use linear vertical interpolation, requiring native coverage beyond 1000 m.

Inspect results:

```bash
python -m oceanembed.inspect data/harmonized/sst/2020-06-01.nc
python -m oceanembed.inspect data/harmonized/teacher/2020-06-01.nc
```

For plots, in a notebook or Python script:

```python
import xarray as xr
import matplotlib.pyplot as plt
with xr.open_dataset('data/harmonized/teacher/2020-06-01.nc') as ds:
    ds.value.sel(depth=100).plot()
    plt.show()
```

Install matplotlib if not already available. Confirm coastlines, realistic field patterns, unit magnitudes and correct north/south orientation. Validate numerical ranges against provider metadata, not a guessed hard clipping interval.

## 8. Build the cube and process Argo

After harmonizing all required dates:

```bash
python -m oceanembed.pack --input data/harmonized --output data/cube --start 2016-01-01 --end 2025-12-31
```

The packer writes memory-mapped arrays and permits missing source days. It will reject blind-period teacher files. Missing source days remain NaN with support zero. It does not fabricate a source or perform unrecorded interpolation.

The initial cube is retrospective. Its age channel is zero because no stale carry-forward is applied; zero here means the field's represented day matches the target day, NOT that the product was published instantly. A real issue-time-aware age pipeline is a separate extension.

Process Argo:

```bash
python -m oceanembed.argo --directory data/raw/argo --out data/argo/profiles.csv
```

Defaults accept delayed-mode core profiles, good position/time flags, adjusted T/S/P flag 1, and configured finite adjusted-error bounds. The defaults are engineering filters, not an official universal Argo prescription. Preserve the audit JSON. Review sample counts and manually inspect profiles before freezing filters.

Pressure is converted to depth with GSW. Adjusted practical salinity and in-situ temperature are converted to potential temperature at 0 dbar. No extrapolation to 0 m is permitted. Interpolation gaps default to 25 m at target depths <=200 m and 100 m deeper; evaluate sensitivity on training profiles and freeze your final thresholds.

The script creates `train`, `selection`, `calibration`, and `test` labels. 2022 float IDs are deterministically assigned to selection/calibration groups. Future test floats may have appeared before 2023; report that as future-time generalization. To claim unseen-float generalization, implement the stricter split in the scientific contract.

## 9. Fit training-only normalization, climatology and EOF

```bash
python -m oceanembed.fit --cube data/cube --out runs/core --modes 8
```

Outputs:

```text
runs/core/stats.npz
runs/core/fit_manifest.json
```

The fitted climatology uses annual/semiannual harmonics, per grid cell and depth, with a small numerical regularizer. Input normalization uses only 2016–2021. EOF modes use sampled complete deep-water anomaly columns, also training-only. Missing below-bottom profiles are never filled with zero for PCA.

For production, audit cells with insufficient training target support. The current fit can return near-zero coefficients where no targets exist; those cells must remain outside the final valid mask. This should normally follow the conservative bathymetry and teacher-support contract, but must be checked for the actual products.

## 10. Train and evaluate the ridge baseline

```bash
python -m oceanembed.baseline --cube data/cube --stats runs/core/stats.npz --out outputs/ridge-selection --start 2022-01-01 --end 2022-12-31 --alpha 10
python -m oceanembed.evaluate --predictions outputs/ridge-selection --argo data/argo/profiles.csv --split selection --out outputs/ridge-selection-metrics
```

Try a small predeclared alpha set, for example 1, 10, 100, and select using 2022 only. Compare identical profile/depth pairs. Sampled training rows keep the baseline feasible; increase the per-day sample budget if learning curves support it.

The climatology score is computed on the same collocated pairs in evaluation. The kit includes ridge, but not a tuned XGBoost/MLP/standard-U-Net benchmark suite. Those experiments remain required before claiming superiority over all strong baselines; implementation steps are in `EXTENSIONS.md`.

## 11. Train the first core model

CPU smoke uses width 8. On a GPU start with width 16 or 32 and a small batch:

```bash
python -m oceanembed.train --cube data/cube --run runs/core --epochs 30 --width 16 --history 7 --batch 2 --patch 64 --device cuda
```

If you do not have a CUDA GPU, use `--device cpu`; long real-data training may be impractical. Reduce `--batch` first when GPU memory is exhausted. Reduce width second. Do not change depth or geography to disguise a memory failure.

Expected outputs:

```text
runs/core/best.pt
runs/core/model.json
runs/core/train_config.json
runs/core/history.json
```

The default objective is masked Huber profile error plus fixed-weight vertical-gradient and 0–300m mean-anomaly terms. Defaults are stable starting choices, not tuned physics constants. The default uncertainty head is untrained: do not display its scale. The observer trains a scale against real observation residuals later.

Teacher validation chooses checkpoints in this starter. To make the strongest final claim, compare candidate/checkpoint exports against the predeclared 2022 Argo selection score and keep that decision separate from calibration. Never use calibration or test profiles for architecture selection.

Do not over-interpret 30 epochs: examine validation curves. Add patience-based early stopping or compare a fixed small set of epoch budgets; preserve the budget and decision before opening the test.

For spatial attention add `--spatial`. To learn full-domain context, fine-tune with `--patch 0 --batch 1`, in a NEW run folder with copied statistics and matching architecture:

```bash
python -c "import pathlib,shutil; pathlib.Path('runs/full-domain').mkdir(parents=True,exist_ok=True); shutil.copyfile('runs/core/stats.npz','runs/full-domain/stats.npz')"
python -m oceanembed.train --cube data/cube --run runs/full-domain --epochs 10 --width 16 --history 7 --batch 1 --patch 0 --pretrained runs/core/best.pt --device cuda
```

If the original model used `--spatial`, include it again. A different architecture cannot strictly load the old checkpoint. Full-domain attention requires profiling; do not claim patch-local attention provides basin-wide context.

## 12. Optional masked pretraining experiment

Use a separate run so SSL does not overwrite your selected checkpoint/config:

```bash
python -c "import pathlib,shutil; pathlib.Path('runs/ssl').mkdir(parents=True,exist_ok=True); shutil.copyfile('runs/core/stats.npz','runs/ssl/stats.npz')"
python -m oceanembed.train --cube data/cube --run runs/ssl --ssl --epochs 15 --width 16 --history 7 --batch 2 --patch 64 --device cuda
python -m oceanembed.train --cube data/cube --run runs/ssl --pretrained runs/ssl/ssl.pt --epochs 30 --width 16 --history 7 --batch 2 --patch 64 --device cuda
```

The implementation masks blocks over all history days and hides associated masks/support. It reconstructs only originally valid latest-day surface values. The broader v4 swath, channel, stale, coastal corruption and quality-aware masking experiments are extensions. Missing-channel augmentation is included in supervised training.

Compare SSL/non-SSL under the same downstream training budget, matched selection profiles and at least multiple final seeds. SSL is retained only when it improves the task or robustness.

## 13. Train Argo observer correction

```bash
python -m oceanembed.observer --cube data/cube --run runs/core --argo data/argo/profiles.csv --epochs 100 --device cuda
```

The teacher backbone is loaded in evaluation mode and never optimized here. The small adapter predicts residual temperature and a positive Laplace scale. It learns from training Argo and selects its checkpoint on the 2022 selection subset. Calibration/test labels are filtered out.

Training applies the same four-cell spatial observation operator to adapter outputs as deployed collocation; it does not assume a nonlinear network commutes with interpolation. The learned scale describes collocated residual spread and is subsequently calibrated; it is not a resolved physical error budget.

Export base and corrected selection predictions:

```bash
python -m oceanembed.predict --cube data/cube --runs runs/core --base-only --out outputs/base-selection --start 2022-01-01 --end 2022-12-31 --device cuda
python -m oceanembed.predict --cube data/cube --runs runs/core --out outputs/core-selection --start 2022-01-01 --end 2022-12-31 --device cuda
python -m oceanembed.evaluate --predictions outputs/base-selection --argo data/argo/profiles.csv --split selection --out outputs/base-selection-metrics
python -m oceanembed.evaluate --predictions outputs/core-selection --argo data/argo/profiles.csv --split selection --out outputs/core-selection-metrics
python -m oceanembed.compare --a outputs/core-selection-metrics/pairs.csv --b outputs/base-selection-metrics/pairs.csv --out outputs/observer-comparison.json
```

Negative `rmse_a_minus_b` favours the corrected model. The paired WMO bootstrap is approximate under remaining spatial/time dependence. Check depth/basin behaviour, not just a single average. Remove the adapter if it does not improve validation. A base-only release needs its own trained/calibrated uncertainty path; the kit deliberately exports NaN scales for base-only inference.

## 14. Ensemble and uncertainty calibration

For the full ensemble repeat statistics copying/training/observer training into `runs/seed17`, `runs/seed29`, `runs/seed43`, using `--seed 17`, `--seed 29`, and `--seed 43` for teacher training. Keep the chosen architecture identical. The observer script currently uses a fixed initialization seed; change it to a configurable distinct seed if testing independent observer initializations. Base-model diversity still differs across teacher seeds.

Example ensemble export:

```bash
python -m oceanembed.predict --cube data/cube --runs runs/seed17 runs/seed29 runs/seed43 --out outputs/ensemble-2022 --start 2022-01-01 --end 2022-12-31 --device cuda
python -m oceanembed.evaluate --predictions outputs/ensemble-2022 --argo data/argo/profiles.csv --split calibration --out outputs/calibration
```

For a one-model prototype use `--runs runs/core` instead. Do not describe one member as an ensemble.

Calibration requires a minimum per-depth sample/float count. The engineering floor of 30 observations/3 floats is not a statistical guarantee. If a depth fails, acquire more valid data or design/freeze a pooled calibration scheme before the test. Do not lower the floor until a desired test score appears.

The code uses scaled absolute-residual calibration, not conformalized quantile regression. It targets pointwise marginal intervals under relevant assumptions. Floats are correlated and 2023–2025 can shift from 2022, so coverage must be measured. It does not guarantee a whole 15-depth profile is jointly covered at 90%.

## 15. Freeze and run the final test

Record the chosen model, weights, stats, product versions, QC, interpolation rules, sources, configuration and calibration. If using Git, commit code/configs before opening test labels; never commit credentials or all raw arrays.

```bash
python -m oceanembed.predict --cube data/cube --runs runs/seed17 runs/seed29 runs/seed43 --out outputs/final-test --start 2023-01-01 --end 2025-12-31 --device cuda
python -m oceanembed.evaluate --predictions outputs/final-test --argo data/argo/profiles.csv --split test --unlock-test --calibration outputs/calibration/calibration.json --out outputs/final-test-metrics
```

Use the SAME model list/configuration as calibration. A mismatched calibration signature is rejected. If testing only one model, use that one consistently in both steps.

Outputs:

- `pairs.csv`: exact collocations behind the results.
- `metrics.json`: counts, float/profile counts, RMSE, MAE, bias, correlation, anomaly correlation, climatology-relative skill, coverage and width.

The report is depth-wise. For basin/season stratification, define basin polygons and seasonal bins BEFORE looking at test performance, attach labels to `pairs.csv`, and call the same `metrics` function per group. Avoid a crude longitude threshold that incorrectly partitions geography. Add the reporting script described in `EXTENSIONS.md`.

No regression "accuracy percentage" is created. Literature scores are not your results. If a real error is found after test access, document the exposure and use a new untouched evaluation for the revised system where feasible.

## 16. INCOIS benchmark

First confirm the exact temperature definition. Then run one of these, choosing only the documented one:

```bash
python -m oceanembed.incois --predictions outputs/final-test --reference data/raw/incois.nc --out outputs/incois --temperature-kind potential --definition-source "Provider document title/version/section confirming potential temperature"
```

or:

```bash
python -m oceanembed.incois --predictions outputs/final-test --reference data/raw/incois.nc --out outputs/incois --temperature-kind in_situ --definition-source "Provider document title/version/section confirming in-situ temperature"
```

The latter uses INCOIS salinity/depth to harmonize the reference. It does not feed subsurface INCOIS salinity into model inference. The code aggregates spatially and monthly, requires at least 25 supported days per cell by default, and exports monthly/depth metrics. Document sensitivity to this completeness threshold.

The monthly analysed product is not raw truth and may share underlying Argo observations with the profile benchmark. Do not call those two tracks independent of each other. No INCOIS 0 m score is fabricated.

## 17. Missing-sensor and wind experiments

Missing SSS test:

```bash
python -m oceanembed.predict --cube data/cube --runs runs/core --drop sss --out outputs/no-sss-selection --start 2022-01-01 --end 2022-12-31 --device cuda
python -m oceanembed.evaluate --predictions outputs/no-sss-selection --argo data/argo/profiles.csv --split selection --out outputs/no-sss-selection-metrics
```

`--drop` hides a channel and its support/mask at inference. This is a sensor-failure experiment, not a retrained feature ablation. For a retrained ablation, hide that channel consistently in both training and inference and train a fresh run. Do not use full-input calibration intervals with a missing-input configuration; the signature check rejects that mismatch.

ASCAT-C swath data cannot be passed through `prepare.py`, which accepts only rectilinear grids. Follow the explicit swath-processing steps in `EXTENSIONS.md`, then write compatible daily files and compare common periods. Do not publish an ASCAT result before that adapter and ablation are actually run.

## 18. Publish a public read-only demo

The updated UI is a six-view research explorer. Read `WEBSITE_SCIENTIFIC_WORKFLOW.md` and use the checked `oceanembed.bundle` command in the numbered guide. The deployment commands below still apply.

1. Create `outputs/public`.
2. Copy a small, representative set of real `.npz` dates from the frozen test export.
3. Copy the corresponding `manifest.json`.
4. Copy `outputs/final-test-metrics/metrics.json` to `outputs/public/metrics.json`.
5. Preserve `synthetic:false` only for genuinely real data. Never edit a synthetic flag to create an apparent real result.
6. If the page presents full-test metrics with selected map dates, label that distinction. The maps are examples; the score describes the stated evaluation period.

Local Linux/macOS:

```bash
export OCEANEMBED_RESULTS=outputs/public
python -m uvicorn oceanembed.api:app --host 0.0.0.0 --port 8000
```

Windows Command Prompt:

```bat
set OCEANEMBED_RESULTS=outputs/public
python -m uvicorn oceanembed.api:app --host 0.0.0.0 --port 8000
```

This is local serving, not yet a public URL.

### Render Python web-service route

Consult https://render.com/docs/deploy-fastapi for current settings and https://render.com/docs/web-services for service behaviour. Provider labels and plan limits can change.

1. Create a GitHub repository for the code. Keep it private initially if desired.
2. Upload the project files. Exclude `.venv`, raw data, credentials, training runs, and massive predictions.
3. The supplied `.gitignore` excludes `outputs/`. To include a SMALL approved public sample, intentionally add only `outputs/public` with `git add -f outputs/public`, or alter the ignore rule specifically for that folder. Do not force-add all outputs.
4. Sign into Render → **New → Web Service**.
5. Connect/select that repository.
6. Set runtime to Python.
7. Set the Python version using the provider's current documented mechanism; choose a supported 3.12 release.
8. Build command:

```bash
pip install -r requirements-serve.txt
```

9. Start command:

```bash
python -m uvicorn oceanembed.api:app --host 0.0.0.0 --port $PORT
```

10. Add environment variable `OCEANEMBED_RESULTS` with value `outputs/public`.
11. Set health-check path to `/health` if the service offers it.
12. Choose an appropriate service plan after checking current pricing and sleep/storage behaviour.
13. Deploy. Read build/runtime logs. Open the generated HTTPS URL.
14. Test `/health`, `/api/dates`, `/api/meta`, the map and a profile.
15. Open the public URL in a private/incognito browser to verify it does not depend on your signed-in session.

No provider credentials or GPU are needed in the serving service. It reads approved saved results. Do not run downloading or training during web requests.

For Docker, after preparing `outputs/public`:

```bash
docker build -t oceanembed-demo .
docker run --rm -p 8000:8000 oceanembed-demo
```

The Dockerfile is a serving image. It is not a CUDA training image. The included Python API has no expensive public inference endpoint, reducing compute abuse and simplifying scaling.

## 19. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No module named oceanembed` | Wrong folder/interpreter | Open project root, activate `.venv`, `pip install -e .` |
| CUDA unavailable | CPU wheel, driver or unsupported environment | Reinstall using official PyTorch selector; use CPU smoke first |
| Dataset ID not found | Product ID used, dataset retired/version changed | Copy active DAILY dataset ID from official catalogue |
| Units mismatch | Unreviewed product attributes | Inspect source, set exact expected units and correct conversion |
| No valid profiles | Real-time data, restrictive QC, wrong region/date, incomplete download | Read audit/index and inspect original files; do not silently remove QC |
| Calibration insufficient | Too few valid profiles/floats at depth | Acquire more data or predesign pooled calibration |
| Out-of-memory | Full-domain batch/attention or too-wide network | Reduce batch, use patches, reduce width; profile before scale-up |
| All-NaN coastal outputs | Strict support/depth/stencil rules | Inspect masks; this can be correct, do not fill across land |
| No teacher validation windows | Pilot contains only 2020 | Prepare a separate real 2022 pilot period |
| Date absent from API | Missing inputs/history or not exported | Check predict logs, available dates, history continuity |
| Synthetic badge | You are serving `outputs/smoke` | Keep it; switch only after real-data export exists |
| Bad skill despite low teacher loss | Teacher bias, climatology dominance, bad collocation or weak relationship | Inspect Argo residuals, units and baselines; do not tune on test |
| Calibration mismatch | Weights, members or sensor setup differs | Recalibrate the frozen final configuration on reserved calibration data |
| Web returns 503 | Results manifest missing/wrong path | Set result path and deploy selected outputs |

## 20. What to do first, today

1. Install the environment.
2. Run `python run_smoke.py`.
3. Open the local explorer and confirm the synthetic banner.
4. Create provider accounts.
5. Download one day from each product.
6. Inspect names/units/QC and fill the ingestion config.
7. Expand to the 30-day training pilot plus a 2022 pilot.
8. Harmonize and inspect maps.
9. Only then scale downloads and training.

Do not spend the first week polishing a website whose scientific data pipeline has not been checked.

## Official references checked for this implementation

- Copernicus subset API: https://help.marine.copernicus.eu/en/articles/8283072-copernicus-marine-toolbox-api-subset
- Copernicus GUI subsetting: https://help.marine.copernicus.eu/en/articles/8078281-how-to-download-copernicus-marine-data-with-the-graphical-user-interface-subset-form
- Earthaccess quick start: https://earthaccess.readthedocs.io/en/stable/user/quick-start/
- Argo GDAC access: https://argo.ucsd.edu/data/data-from-gdacs/
- Argo profile handling: https://argo.ucsd.edu/data/how-to-use-argo-files/
- TEOS-10 potential temperature: https://www.teos-10.org/pubs/gsw/html/gsw_pt0_from_t.html
- INCOIS metadata: https://erddap.incois.gov.in/erddap/griddap/incois_argo_mnt_VAM.html
- PyTorch installation: https://docs.pytorch.org/get-started/locally/
- VS Code environments: https://code.visualstudio.com/docs/python/environments
- Render deployment: https://render.com/docs/deploy-fastapi

These establish provider/library operations, not validation of your future model's accuracy.
