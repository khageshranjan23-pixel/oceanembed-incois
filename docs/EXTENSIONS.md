# Complete the high-end v4 experiments and scalable deployment

This file gives explicit implementation steps for features outside the tested reference core. Do not label these as implemented until their checks pass.

## 1. SSS QC, uncertainty and source-age features

1. Select one exact SSS product/version.
2. Read its product manual for uncertainty, RFI, land-contamination and QC fields. Some L4 products do not expose the original retrieval QC; do not invent it.
3. Preserve native uncertainty separately from remapped valid-area support.
4. Decode each flag according to the manual. For a bit field, test required bits with integer bitwise operations, not numeric ordering of flags.
5. Compare counts and maps before/after filtering using training dates.
6. Remap QC/uncertainty with a scientifically justified rule; averaging uncertainty is not always valid error propagation because retrievals are correlated.
7. Add explicit fields to the cube, and update `Cube.input` and the model's first convolution channel count together. A model trained on the old channel schema cannot reuse weights blindly.
8. Carry source/observation age from an availability-aware record in NRT. Missing values need a mask; age is not a substitute for missingness.
9. Train corruption/dropout on observed training-period missingness; tune the schedule on selection data.
10. Keep/reject each quality feature by a matched Argo ablation.

## 2. ASCAT-C coastal wind adapter

The source is Level 2 swath data; the regular-grid remapper is inappropriate.

1. Download the exact coastal product and manual from the official PO.DAAC listing.
2. Confirm retrieval ambiguity selection, rain/ice/land flags, vector conventions, observation times and processing version.
3. Read a swath and flatten valid observation positions, times, U and V into records. If only speed/direction is provided, use the documented meteorological/oceanographic direction convention to compute U/V. Do not assume direction meaning.
4. Normalize longitudes, subset the domain halo, apply QC, and assign each observation to a UTC target day.
5. Use the target edges from `science.edges` to assign rows to 0.25-degree cells. Verify edge inclusivity once so boundary points are not duplicated.
6. Form a QC-filtered mean vector in each cell; use equal weights initially unless the provider supplies an uncertainty model supporting a better weighting. Treat repeated overlapping observations transparently.
7. Save observation count, within-day observation time span, source identifier, fraction of expected sampling if a defensible denominator exists, and representative age. A count-derived presence mask is not geographic area coverage.
8. Leave unobserved cells missing. Never paint a complete daily global field by smoothing across the coastline.
9. Write daily `value(latitude,longitude)` NetCDFs for wind U/V. If using the kit's `support` slot as a presence indicator, record this semantic change and train a source-aware model; do not describe it as area support.
10. Run the baseline/core on the common CCMP/ASCAT availability period. Report coverage and matched-profile scores separately.
11. For a fair retrained-source comparison, use equal training budgets and matching period/locations. Inference-only substitution measures product shift instead.

CCMP includes a background analysis and derived products share inputs. Wind-source experiments do not by themselves prove every other channel has strict satellite-only lineage.

## 3. Full sensor-failure SSL curriculum

The provided SSL masks spatial blocks. Add other failure types incrementally:

- Whole channel over the full window.
- Sensor missing on specific days.
- Swath-like holes drawn from observed source masks.
- Stale input: substitute only an earlier available field and update age.
- SSS coastal degradation sampled from product-supported QC/error patterns.

For each augmentation, save its parameters and plot before/after examples. The SSL target must be an originally valid observation. Do not reconstruct an interpolated pretend truth. Hide related derived quantities that reveal a masked variable. No arbitrary flips/rotations unless all geographic/vector semantics are transformed consistently and the transformed ocean remains scientifically meaningful; easiest is not to use them.

## 4. GradNorm integration

`advanced.GradNormWeights` is provided for nonnegative profile-Huber, gradient-Huber and column-Huber losses. Refactor `teacher_loss` to return the three scalar terms separately, rather than their current fixed weighted sum.

A training-step structure is:

```python
# created once, outside the loop:
balancer = GradNormWeights(n=3, alpha=1.0).to(device)
weight_optimizer = torch.optim.Adam(balancer.parameters(), lr=1e-3)

# after calculating three nonnegative losses on a batch:
weight_optimizer.zero_grad()
gradnorm_loss = balancer.objective(losses, model.d1.parameters())
gradnorm_loss.backward()  # detached model-gradient norms: updates weight parameters only
weight_optimizer.step()

optimizer.zero_grad()
weights = balancer.weights().detach()
model_loss = sum(w * loss for w, loss in zip(weights, losses))
model_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

Warm up with fixed equal weights after normalizing loss magnitudes. Save/restore balancer parameters and initial-loss buffers with the training checkpoint. Omit batches/terms with no valid observations instead of letting a permanently zero loss determine ratios. Log weights and gradients. The provided default script does not enable this path automatically.

Never pass raw continuous Laplace NLL into the ratio-based balancer: it can be negative. Use a separate fixed-weight probabilistic fine-tuning stage or a separately justified method. Compare with the simpler fixed-weight baseline before retaining the feature.

## 5. Regime adapters

`advanced.RegimeAdapters` returns adapted features, balance loss and routing usage. To integrate:

1. Add `self.adapters = RegimeAdapters(width, experts=4, rank=16)` after the decoder feature stage.
2. Before coefficient/residual/scale heads, call `f, balance, usage = self.adapters(f)`.
3. Return the balance term separately to the training loop.
4. Add a small, validation-selected balance weight to the loss; log expert usage by region and season.
5. Start the large backbone from a selected dense checkpoint. Adding a module requires explicit partial loading and checking exactly which new keys are missing. Never globally ignore arbitrary checkpoint mismatches.
6. Train with router warm-up and compare multiple seeds on fixed 2022 selection profiles.
7. Keep adapters only if their incremental gain exceeds measurement uncertainty and operational cost.

The small module calculates every expert then gates outputs. That is simple and correct for a small model but does NOT deliver sparse compute savings. Efficient dispatch is a later optimization, not a scientific requirement.

## 6. Stronger baselines

Do these before claiming best performance:

- Climatology: already included in pair metrics.
- Ridge: already included.
- XGBoost: use the same 32 pointwise features and per-depth targets as `baseline.py`, replace Ridge with `XGBRegressor`, tune tree depth/learning rate/regularization on selection only. Limit training rows deliberately and report the budget.
- MLP: replace the ridge predictor with a 32→128→128→15 network and masked per-depth Huber loss. Use the same training/validation protocol.
- Standard U-Net: create a one-day encoder/decoder with no EOF, temporal or spatial attention. Output 15 anomalies, keep climatology and masks, and tune width under a comparable budget.
- Temporal CNN: keep the spatial baseline and use temporal convolution/pooling, to distinguish attention from simply using history.
- Observer fairness: compare teacher-only variants separately from observation-corrected variants, or attach a comparable observer to strong baselines.

Do not describe the reference core with history=1 as a textbook standard U-Net: it still contains a one-token attention projection and the structured decoder. Name experiments according to what is actually in code.

## 7. Stratified reports

Define basin polygons, coastline-distance threshold and seasons before test access. Use a reviewed GeoJSON polygon file for Bay of Bengal and Arabian Sea, with remaining valid cells in a third group. A point-in-polygon library such as shapely can assign collocated profiles.

For seasons, a conventional starting partition is DJF (NE monsoon), MAM (pre-monsoon), JJAS (SW monsoon), ON (post-monsoon). State this explicitly; oceanographic conventions vary by region. Define dynamic/wind thresholds from training distributions.

Once `pairs.csv` has columns `basin`, `season`, and `coastal_class`, the report core is:

```python
import pandas as pd
from oceanembed.science import metrics
pairs = pd.read_csv('outputs/final-test-metrics/pairs.csv')
# labels must already have been attached from frozen geographic/seasonal definitions
rows = []
for (basin, season, depth), group in pairs.groupby(['basin', 'season', 'depth']):
    rows.append({
        'basin': basin, 'season': season, 'depth': depth,
        'floats': group.wmo.nunique(),
        **metrics(group.theta0, group.temperature, group.climatology)
    })
pd.DataFrame(rows).to_csv('outputs/stratified_metrics.csv', index=False)
```

Show absent strata as insufficient evidence. Do not silently drop poor conditions. Add paired block-bootstrap CIs and sensitivity to float/month/spatial grouping when interpreting small differences.

D20 diagnostics require a valid 20°C crossing and a predetermined multiple-crossing rule; never extrapolate a crossing outside supported depths. Maximum-gradient depth depends strongly on the coarse 15-level grid. State its resolution. Integrated temperature anomaly is not heat content.

## 8. Rich public frontend

After real output exists, an alternative production interface can replace the included Plotly research workspace with Next.js + TypeScript, MapLibre/deck.gl and Plotly. It can use the same read-only API.

Implement in this order:

1. Explorer: date selector, depth selector, tiled temperature raster, coordinate click.
2. Profile: 15-depth curve, explicitly supported calibrated intervals, observed Argo points.
3. Validation: depth/basin/season tables and charts, sample counts, bootstrap intervals, metric definitions.
4. Baselines: matched-test comparison, separately labelled teacher-only and corrected experiments.
5. Sensor-failure: switch between PRECOMPUTED complete/missing-input exports with actual measured metrics; do not invent animation-driven error changes.
6. Argo challenge: prediction displayed first, observation reveal from a fixed documented test pool, deterministic or logged random sample selection.
7. Data health: source versions, missing fractions, ages and NRT/consolidated mode.
8. Architecture and provenance: concise explanation plus downloadable run manifest.
9. 3-D view: only after scientific pages work; interpolation is labelled visualization, not additional vertical resolution.

The updated research site renders matching calibrated intervals and evaluated Argo overlays. To add intervals, read the calibrated evaluator output (or an export containing the matching calibration signature) and display only supported depths. For an arbitrary map cell, calibration is transferred from observed profiles; do not imply local guaranteed coverage.

## 9. Real-time and revised-product pipeline

Historical reconstructed output is the first release. NRT requires a new input contract:

```
source valid_time
source observation_window
provider publication_time (when available)
first_seen_time
retrieval_time
source_version
quality/missingness
```

For a chosen issue time, select only files that were available by the cutoff. If publication times are not archived, begin capturing first-seen times now and call older simulations simulated availability, not verified backtests. Reprocessed satellite products may incorporate later observations even if the input date itself is <=target date.

Use product-specific staleness limits. When substituting an older field, carry actual age and mask; train on comparable stale cases. If all required channels are unusable, withhold the new result or publish an explicitly labelled climatology fallback. Do not show a normal-quality prediction.

Run idempotent jobs:

1. discover data;
2. download with retries/checksums;
3. QC and normalize;
4. write immutable daily input snapshot;
5. infer all models;
6. validate output;
7. atomically publish an immutable run;
8. update a latest-pointer only after success;
9. retain earlier revisions.

Use one NRT-v0 and later consolidated-v1 revision as data improve. Model product changes using training/selection overlap periods. A DUACS geostrophic-current fallback is not an identical replacement for OSCAR total-current estimates.

Monitoring: source missingness/age, product schema changes, range/quality distribution, latent/output drift, delayed Argo errors and coverage. A champion/challenger deployment requires an untouched benchmark for new releases; repeatedly tuning on the old final test erodes its status.

## 10. Scalability

The full regional 15-depth mean field is ~1.46 MB/day as uncompressed float32, before masks/intervals. Ten years of seven surface channels are roughly 2.5 GB, and 15-level targets roughly 5.3 GB. Extra features, validity, age, checkpoints and raw archives increase the requirement. Raw high-resolution global source files can dominate storage; estimate from an actual pilot.

Training:

- Preprocess once, save versioned regional cubes, and memory-map arrays.
- Stream dates/patches; never create seven copies of each day on disk.
- Move to Zarr chunks after benchmarking read amplification for your windows.
- Add mixed precision with finite-gradient checks after a correct float32 baseline. Evaluate any inference precision change against the frozen FP32 outputs.
- Add multiple GPUs only if profiling shows a real need; one moderate GPU can develop this regional model, but exact training time is hardware/data dependent.

Serving:

- Predict each date once, not on every click.
- Separate GPU jobs from CPU web serving.
- Keep immutable outputs in object storage. Cache maps at a CDN.
- Return one selected depth/small profile, not full multi-year arrays in JSON.
- For larger regions serve raster tiles and binary arrays; the starter's JSON is acceptable for its 24k-cell slice but not a global high-frequency feed.
- Add an object-store fetch/cache layer rather than mounting huge scientific training archives in every web instance.
- Use request limits/timeouts and input validation. The API already rejects unsafe date strings and unsupported depth choices.
- Load-test the public read-only API at realistic traffic before choosing instance count. Do not claim a user-capacity figure without measurements.

The important first scalability improvement is removing repeated computation and global downloads. Kubernetes, a large database and multiple GPU replicas are not prerequisites for the SIH regional demo.
