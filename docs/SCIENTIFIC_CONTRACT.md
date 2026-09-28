# Scientific contract and limitations

## Outputs

Target: daily potential temperature referenced to 0 dbar, Celsius, on the inclusive 101x241 coordinate grid, at the 15 specified depths. Nominal 0 m uses the teacher's shallowest available model level. The kit does not predict salinity at depth and does not claim density stability or ocean heat content.

## Information boundaries

- Fitted statistics, EOF, SSL and teacher training: 2016–2021 only.
- Model/observer selection: permitted 2022 selection data.
- Calibration: reserved 2022 Argo calibration float groups.
- Frozen future test: 2023–2025; never tune from these scores.
- Subsurface Argo is used during observer training and evaluation, not inference.
- Inputs can be blended products. No strict satellite-only claim without lineage-reviewed alternatives.
- The test is held out from this model's training, but a blended surface analysis may contain some related observations. This is not complete observing-system independence.
- Historical reprocessed fields do not establish what was available operationally on that date.

## Temperature harmonization

The primary raw Argo benchmark uses delayed-mode adjusted T, S and P with documented QC/error filters. GSW computes Absolute Salinity, potential temperature and depth. Inspect provider reference conventions; do not mix Conservative Temperature, potential temperature and in-situ temperature under one label. Temperature conversion is a thermodynamic operation, not a model accuracy improvement.

INCOIS requires confirmation of its temperature definition. The comparison code will not infer it from ambiguous units. Conversion using INCOIS salinity occurs in the evaluator only.

## Geometry

The remapper assumes rectilinear cell-centred source arrays with edges inferred from centres. This is suitable only after checking source geometry. It is not valid for all grids, swaths or point observations. Valid-area support uses total target area; this is conservative near coasts but is not independent observation coverage.

Bathymetry is an approximate regional validity rule. A mean depth does not completely describe a cell's subgrid wet fractions. For final coastal work, calculate depth-dependent wet-area fractions from higher-resolution bathymetry and teacher native wet masks, choose/freeze support thresholds, and audit sensitivity. The starter rejects many marginal coast cells rather than filling them across land.

Collocation requires all four neighbouring cells valid at depth. This sacrifices coverage for an explicit rule. Reject geographically disconnected stencils if using complex coastlines: all-four-valid alone does not mathematically prove ocean connectivity across narrow land barriers. Add a high-resolution land-intersection check for coastal production evaluation.

## Model differences from full v4 ambition

Included core: residual CNN-style stages, temporal bottleneck attention, optional spatial attention, EOF+residual decoder, optional block SSL, observer, learned residual scale, ensemble export and residual calibration.

Not silently claimed complete: ConvNeXt-specific blocks, multi-scale temporal skip fusion, full sensor-time/swath/staleness SSL curriculum, product-specific SSS reliability encoding, ASCAT coastal adapter, tuned XGBoost/MLP/U-Net benchmark suite, production NRT, a polished multi-page 3-D frontend. These have detailed implementation gates in EXTENSIONS.md.

GradNorm and regime adapters are provided as optional standalone modules and tests, not enabled defaults. This avoids claiming they improved a result that was never run. The default loss weights are provisional, and the default network scale is smaller than the proposal's aspirational parameter count.

## Validation and uncertainty

Primary scores are in degrees Celsius. Show depth-wise errors and matched-sample climatology skill. Absolute temperature correlation can be inflated by geography/season, hence anomaly correlation is also needed. A single pooled-depth RMSE can hide thermocline errors and sampling imbalance.

Laplace b is a scale, not standard deviation: variance is 2b^2. Ensemble variance combines mean conditional variance and variance of member means. This is model-based uncertainty, not a unique measured decomposition into physical aleatoric/epistemic components.

Calibrated intervals use scaled absolute residuals. Pointwise calibration does not imply joint profile coverage. Float/time/spatial dependence and shift limit exchangeable coverage claims. Small counts, shallow gaps and missing sensors must be shown.

If calibration fails sample-count checks, preserve the failure and acquire more data or predefine a scientifically reasonable pooling scheme. Never fill 0 m interval calibration from nonexistent 0 m Argo observations.

## Stronger float-holdout design

Before any observer fitting, hash unique WMO IDs into separate groups. Remove every historical profile belonging to the designated unseen-float group from observer training and selection. Keep a future test subreport for those floats. For a separate independent-platform experiment, use documented observations excluded from input-product analyses where feasible. The default temporal split alone does not establish these stronger claims.

## Reproducibility checklist

Preserve source IDs/versions, downloaded checksums, provider licenses, QC settings, thermodynamic conventions, split manifests, training config, seeds, fitted statistics, weights, observer, calibration, evaluation pairs and code commit. Record the exact software environment on the actual training machine. Do not publish account credentials or source archives contrary to their terms.

## Claim discipline

Synthetic examples are plumbing checks only. GLORYS is a teacher, not observational truth. INCOIS analysed-grid and raw-profile benchmarks can share observations. Literature RMSE is not your result. A missing sensor need not worsen every individual sample; uncertainty need not automatically widen unless the trained/calibrated model supports it. A clean UI does not establish model superiority.
