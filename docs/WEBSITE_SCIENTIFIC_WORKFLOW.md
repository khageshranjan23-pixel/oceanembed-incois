# Website implementation and scientific visualization guide

The website source is implemented in `web/index.html`, `web/style.css`, `web/app.js`, and `oceanembed/research.py`. It is a read-only scientific viewer backed by precomputed model outputs. It does not invent observations, rerun training on clicks, or require a browser-side GPU.

## 1. The researcher workflow

The information architecture follows questions a researcher can answer:

| Question | Implemented view | Scientific meaning |
|---|---|---|
| Where is the warm/cold structure at a depth/date? | Horizontal temperature map | Native model grid, potential temperature in °C |
| What differs from the seasonal expectation? | Anomaly map | Prediction minus training-only seasonal climatology |
| How broad is the supported interval? | Interval-width map and profile shading | Calibrated pointwise width where matching calibration exists |
| What was supplied to the model? | Valid-input-fraction map | Fraction of seven current-day channels valid; not confidence |
| What is the vertical structure here? | Clickable vertical profile | Native depth predictions and climatology, positive-down depth |
| Does it agree with an actual profile? | Argo profile selector/reveal | Separate observation points and matched predictions |
| What changed between two dates? | Side-by-side fields and B−A difference | Matched grids and common valid cells, not model error |
| How does structure vary along a path? | Great-circle transect and section | Horizontal interpolation, native depth representation |
| How does structure evolve at one point? | Time–depth plot and time series | Missing calendar days preserved as gaps |
| How good is it on observed samples? | Validation dashboard | Depth metrics, climatology skill, residuals, counts and coverage |
| Does the PS benchmark agree at its own scale? | INCOIS results panel | Actual monthly-grid comparison when supplied |
| Can someone reproduce the result? | Provenance and downloads | Model/source identity, CSV and NetCDF exports |

These views are consistent with the dimensions exposed in established ocean viewers: geographic maps and graphs against time, depth and distance. This is a reasoned design choice, not a claim that scientists have user-tested this specific interface.

Official example: https://data.marine.copernicus.eu/viewer

## 2. Visual encodings and why they matter

### Temperature

Use the cmocean `thermal` sequential colour scale with a visible °C colourbar. The default 0–32°C range is a starting DISPLAY range, not a physical clipping rule or validated domain bound. Values outside the display range are saturated visually but remain present in data downloads and numeric hover. Adjust limits deliberately and record them in screenshots/links. The code does not clip the underlying temperatures.

### Anomalies and date differences

Use a diverging `balance` colour scale. Keep symmetric limits around zero when interpreting positive/negative changes. The initial ±3°C is a display default, not an anomaly threshold. A date difference is not observed reconstruction error, and an anomaly is not automatically a marine heatwave.

### Uncertainty

Use `amp` for nonnegative interval width. Display width in °C, and show actual empirical coverage/width by depth in Validation. A narrow interval alone is not evidence of a good model. Missing calibration must leave the interval view disabled or unavailable. Nominal 0 m usually lacks raw-Argo calibration and remains blank in that layer.

### Missing data

Plotly heatmaps use `zsmooth:false`, `connectgaps:false` and `hoverongaps:false`. Missing cells stay grey/blank. Profile/time-series lines do not connect missing samples. Gaps are not zero temperature. The UI does not hide them behind a photorealistic ocean texture.

### Depth

Depth axes run downward in physical metres. Native levels are unevenly spaced. Cross-sections/time–depth heatmaps fill bands centred on native levels for display; those bands are a visualization convention, not observations at every intermediate depth. Profiles connect native samples with line segments rather than splines. Neither choice creates extra vertical resolution.

### Geography

Maps use labelled longitude/latitude coordinates. Natural Earth 1:110m outlines provide context; they are not the scientific land/bathymetry mask and are too coarse for detailed coastal decisions. The current view is not an equal-area map. Do not derive ocean areas or distances by counting screen pixels. Transect distance is explicitly computed on a mean-radius sphere.

Colour documentation: https://matplotlib.org/cmocean/
Plotly heatmaps: https://plotly.com/javascript/reference/heatmap/
Natural Earth licensing: https://www.naturalearthdata.com/about/terms-of-use/

## 3. Click-by-click: Explore ocean

1. Start the server with `OCEANEMBED_RESULTS` pointing at your checked bundle.
2. Open the root URL.
3. Read the run-mode badge and synthetic banner first.
4. Choose **Date A**.
5. Choose a **Native depth**.
6. Choose **Potential temperature**, **Seasonal anomaly**, **Calibrated interval width**, or **Valid current-day input fraction**. Unsupported layers are disabled.
7. Set colour limits if necessary; they do not silently auto-rescale with every date.
8. Click **Update view**.
9. Hover on a cell for coordinates and numerical value.
10. Click a coloured cell, or type latitude/longitude and click **Inspect**.
11. Inspect the profile, climatology and any supported interval.
12. Click **Download profile CSV** to export the model profile at that point. Observation points selected separately are not included in that model-profile CSV.
13. To inspect an observation, choose a profile from the Argo selector for that date and click **Reveal observation**.
14. Compare prediction versus observation without confusing a single-profile score with the full benchmark.

The selector is a case-study selector, not a random blind-test claim. Argo profiles can be absent on a selected date. Default synthetic examples are plumbing fixtures, not a depiction of actual float sampling.

## 4. Click-by-click: Compare dates

1. Select **Compare dates** in the left navigation.
2. Choose potential temperature or anomaly; input fraction and interval width are not enabled for date-difference semantics.
3. Choose Date A in the common controls and Date B in the comparison controls.
4. Keep the same depth and shared colour limits.
5. Click **Compare**.
6. Read the A and B maps using the identical colourbar bounds.
7. Read the B−A panel with a separately labelled symmetric difference range.
8. Adjust **Difference range ±°C** and click **Compare** if needed.

The API rejects grid changes. Only cells valid in both dates can have a numerical difference. This compares states from the same published run, not two competing models. A baseline/ensemble model-comparison UI needs separate matched-run bundles and is a future extension; paired model comparisons already exist in the command-line evaluator.

## 5. Click-by-click: Vertical section

1. Select **Vertical section**.
2. Choose date and field in the common controls.
3. Enter endpoint A latitude/longitude and endpoint B latitude/longitude.
4. Choose endpoints whose path lies inside the exported domain.
5. Click **Draw section**.
6. Check the small path map to verify geographic orientation.
7. Read distance along path against depth. Missing coastal/below-bottom samples remain blank.
8. Hover to inspect values and distance/depth coordinates.
9. Use Plotly's camera control to export a PNG when appropriate, then retain the corresponding NetCDF and manifest.

More horizontal samples do not improve the model's underlying 0.25° resolution. The code limits requests to 250 samples. All-four-valid interpolation reduces coast errors but does not establish connectivity across every narrow land barrier; detailed coastal use needs a higher-resolution connectivity check.

## 6. Click-by-click: Time & depth

1. In Explore, select the location of interest.
2. Select **Time & depth**.
3. Choose start/end dates spanning at most 366 calendar days.
4. Choose temperature, anomaly or interval width.
5. Click **Load time window**.
6. Read the time–depth panel and the selected-depth time series.
7. Check that missing export days are blank and are not connected by time-series lines.

Only published dates exist; the viewer cannot manufacture a daily record from a handful of case-study maps. To demonstrate continuous evolution, export a continuous interval. All dates are UTC.

## 7. Click-by-click: Validation

1. Select **Validation**.
2. Choose the evaluation date interval.
3. Keep the full domain for headline metrics, or enter an exploratory bounding box.
4. Click **Apply subset**.
5. Read sample, profile and unique-float counts before interpreting error differences.
6. Inspect RMSE, MAE and bias by depth.
7. Inspect climatology-relative MSE skill; values below zero mean the model loses to climatology on those matched samples.
8. Inspect predicted-versus-observed scatter and residual distribution.
9. Inspect empirical interval coverage against the 0.9 target and interval width together.
10. Read the exact depth table, including anomaly correlation.
11. Scroll to INCOIS results. An absent result explicitly says unpublished.

Metrics use all matching rows. Scatter and histogram use a deterministic capped sample for responsiveness; this cap is labelled and is not the metric sample. Correlated samples limit naive uncertainty interpretation. The UI does not display invented confidence intervals for error metrics; use the paired float/block bootstrap report for inferential comparisons.

Bounding boxes are not verified basin polygons. For official Arabian Sea/Bay of Bengal scores use the frozen polygons and stratification script described in the extensions. Filtering a seen test set is exploratory analysis, not a new independent blind test.

## 8. Click-by-click: Provenance and exports

1. Select **Data & provenance**.
2. Read temperature and anomaly definitions, 0 m convention, uncertainty interpretation and data-availability wording.
3. Inspect checkpoint/statistics/config hashes and the listed source/run metadata.
4. Click **Download manifest JSON**.
5. Use **Download day · NetCDF** for the selected day.
6. Pair downloaded plots with data files, model version, date/depth, colour limits and the observational benchmark used.
7. Use **Copy analysis link** to preserve the current controls in the URL. It restores view state for the same deployed data; it does not independently archive mutable data. Deploy immutable run URLs for durable citations.

NetCDF export includes CF-style coordinates, temperature metadata and the source manifest. The code has not been passed through a formal CF compliance checker; run one before a release advertised as certified CF-compliant. Potential temperature remains potential temperature, not in-situ temperature or heat content.

## 9. Website code map

| File | What to edit |
|---|---|
| `web/index.html` | Navigation, labels, panels, inputs and accessibility text |
| `web/style.css` | Layout, responsive behaviour, fonts, spacing, colours |
| `web/app.js` | Linked controls, Plotly figures, requests and URL state |
| `web/colours.json` | Sampled scientific colormaps; preserve attribution |
| `web/land.json` | Context outlines, not the scientific wet mask |
| `web/vendor/` | Pinned local chart asset and notices |
| `oceanembed/api.py` | Core API plus static assets |
| `oceanembed/research.py` | Map/profile/difference/transect/time/validation/export endpoints |
| `oceanembed/bundle.py` | Verified predictions and evaluation to a public bundle |
| `tests/test_research.py` | Numerical/API checks for new views |
| `requirements-serve.txt` | CPU-only web dependencies |

When changing a field, change its API definition, unit label, plot colour semantics and tests together. Do not rename a reconstruction as a forecast without changing and validating the actual task.

## 10. Data health and real usage parameters

The included input-fraction layer answers only whether the seven channels have finite same-day values. Product quality, source independence, retrieval uncertainty and publication age are separate concepts. The prediction exporter now retains current-day input validity, support and age arrays; production data-health panels should display each channel separately after their semantics are verified.

For an NRT release, add issue time, valid time, first-seen/publication time, source version, stale-input rules, missingness and revision number. The starter's zero retrospective age does not establish immediate data availability. Show a pending/unavailable state rather than a green real-time badge if these records do not exist.

## 11. Features to postpone until the evidence exists

- Photorealistic 3-D ocean animation: optional presentation, not primary scientific evidence.
- D20/thermocline diagnostics: require crossing/gap rules and an honest statement of coarse depth resolution.
- Ocean heat content: requires the appropriate thermodynamic assumptions and additional variables; integrated °C·m is not J/m².
- Density stability/barrier-layer depth: cannot be established from subsurface temperature alone.
- Marine heatwave classification: needs a defensible reference climatology and duration/threshold definition.
- Fisheries, cyclone or navigation recommendations: need application-specific validation; this temperature reconstruction does not establish them.
- "Live" operational mode: requires issue-time-aware data availability and a separate operational test.

## 12. Release checklist

Verify a known map orientation; exact point-profile collocation; mismatched-grid rejection; difference sign B−A; unsupported 0 m calibration; missing-date gaps; nonuniform native depth axis; coordinate bounds; case-study profile identity; data/metric source consistency; empty-data states; downloads; URL restoration; desktop/mobile layout; and synthetic/real labelling.

Do not publish source credentials. Public interaction must not trigger model training. Use immutable outputs, bounded requests, server caching and later raster/object-storage delivery when measured traffic justifies it. The regional grid is small enough to begin with one CPU service.
