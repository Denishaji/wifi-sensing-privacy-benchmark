# Wi-Fi Sensing Privacy Benchmark

Reproduction of published IRShield measurements and a synthetic evaluation
of reflecting-surface switching policies against retrained motion detectors.

This is a research prototype. It does not demonstrate a device that
protects a home from Wi-Fi sensing.

## Main finding

A switching policy that appears effective against one detector can retain
motion information detectable by alternative models and temporal features.

At simulated surface strength 0.35, independent switching produced mean
ROC-AUC 0.686 for a nearest-neighbor baseline, versus 0.961 and 0.971 for
logistic and Gaussian detectors with temporal features, across three
simulated environments.

These are synthetic results, separate from the measured-data reproduction.

## What this project contains

- Raw-CSI reproduction matching published processed arrays to numerical precision.
- Motion-threshold evaluation on published IRShield recordings.
- Calibration-only subcarrier selection and temporal reference holdouts.
- Memory-efficient loading using streamed extraction and memory mapping.
- Synthetic binary reflecting-surface controllers.
- 63 attacker-condition evaluations across three fresh simulation seeds.
- Recorded thresholds, false alarms, detection rates and ROC-AUC.

## Quick start: synthetic experiment

Python 3.10 or later.

```powershell
python -m pip install -r requirements.txt
python -B compare_attackers.py
```

Results are written inside the project's `results` directory.
The script refuses to overwrite an existing nonempty experiment directory.

## Measured-data reproduction

```powershell
python -B reproduce_fig06.py
python -B verify_raw_csi.py
python -B evaluate_motion.py
python -B evaluate_holdout.py
```

The download scripts retrieve approximately 243 MB in total.
Temporary extraction requires additional disk space.

The raw-CSI verification compares recomputed values against the published
reference arrays. The holdout experiment uses non-overlapping windows and
selects subcarriers and thresholds using calibration data only.

## Example results

See the `examples` directory for saved reports from completed experiments.
Reports retain their original provenance and any original source hashes.

## Attribution

The IRShield measurements and original processing methodology belong to
Paul Staat and collaborators.

- Paper: https://arxiv.org/abs/2112.01967
- Published data and evaluation scripts:
  https://zenodo.org/records/6367411

The source record specifies CC BY 4.0. This project adapts processing
procedures from that work and adds separate evaluation and simulation code.
It does not claim original IRShield hardware or data collection.

Implementation was developed with AI assistance and checked through
numerical reproduction and experiment runs.

## Limitations

- Published measurements do not validate a device built by this project.
- Reference holdouts share a recording with calibration.
- Non-overlapping windows can remain temporally correlated.
- Equal calibration targets do not guarantee equal test false-alarm rates.
- Synthetic channels are not calibrated to a home or physical surface.
- Three simulation environments do not establish broad robustness.
- Attacker comparisons change both model and features.
- The communication-rate proxy is not measured Wi-Fi throughput.
- No IEEE 802.11bf compliance or universal privacy protection is claimed.