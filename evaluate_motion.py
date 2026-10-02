"""Evaluate a threshold motion detector on published IRShield recordings.

Source: https://zenodo.org/records/6367411
Attribution: Paul Staat and collaborators, CC BY 4.0.

Uses published processed statistics and the authors' tag/time alignment.
This is an offline evaluation of published hardware measurements.
"""

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
NAME = "Fig07_adversarial_motion_sensing_IRShield"
ARCHIVE = ROOT / "data" / f"{NAME}.zip"
OUTPUT = ROOT / "results" / "motion_evaluation_v1"

URL = (
    f"https://zenodo.org/api/records/6367411/files/{NAME}.zip/content"
)
EXPECTED_BYTES = 219723642
EXPECTED_MD5 = "74b833be7d78ce3c751ab757d799f7a6"

SETUPS = ["an2_at1", "an2_at3", "an1_at1", "an1_at3"]
FACTORS = [11, 2]


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)

    if ARCHIVE.exists():
        if (
            ARCHIVE.stat().st_size == EXPECTED_BYTES
            and file_md5(ARCHIVE) == EXPECTED_MD5
        ):
            print("Using verified archive.")
            return
        raise ValueError(f"Existing archive failed verification: {ARCHIVE}")

    partial = ARCHIVE.with_suffix(".zip.part")
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "wifi-privacy-reproduction/0.1"},
    )

    print("Downloading 219.7 MB. No extraction is required.", flush=True)
    digest = hashlib.md5()
    total = 0
    next_progress = 20_000_000

    with urllib.request.urlopen(request, timeout=120) as response:
        with partial.open("wb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break

                total += len(block)
                if total > EXPECTED_BYTES:
                    raise ValueError("Response exceeds expected archive size.")

                stream.write(block)
                digest.update(block)

                if total >= next_progress:
                    print(
                        f"Downloaded {total / 1e6:.1f} MB",
                        flush=True,
                    )
                    next_progress += 20_000_000

    if total != EXPECTED_BYTES:
        raise ValueError("Incomplete download. Run again to restart.")

    if digest.hexdigest() != EXPECTED_MD5:
        raise ValueError("Archive checksum mismatch.")

    partial.replace(ARCHIVE)
    print("Archive checksum verified.")


def load_trace(archive, folder):
    prefix = f"{NAME}/data/{folder}/"

    raw = np.load(
        io.BytesIO(archive.read(prefix + "wrt1.npy")),
        allow_pickle=False,
    )
    processed = np.load(
        io.BytesIO(archive.read(prefix + "wrt1_processed.npy")),
        allow_pickle=False,
    )

    if raw.dtype.names is None or "timestamp" not in raw.dtype.names:
        raise ValueError(f"{folder}: missing timestamp field.")

    if processed.ndim != 3 or processed.shape[:2] != (3, 3):
        raise ValueError(f"{folder}: unexpected processed array shape.")

    if not np.isfinite(processed).all():
        raise ValueError(f"{folder}: nonfinite processed values.")

    offset = 3
    count = processed.shape[-1]

    if not offset < count <= len(raw):
        raise ValueError(f"{folder}: invalid array alignment.")

    timestamps = raw["timestamp"].astype(np.float64)

    # Match the authors' plotting alignment and antenna averaging.
    time_s = (
        timestamps[offset:count] - timestamps[offset]
    ) / 1e6

    values = processed[:, :, offset:].mean(axis=-2).mean(axis=-2)

    if np.any(np.diff(time_s) <= 0):
        raise ValueError(f"{folder}: timestamps are not increasing.")

    return time_s, values


def walking_interval(archive, folder):
    """Match load_tags.py: video times become CSI-relative path times."""
    text = archive.read(
        f"{NAME}/data/{folder}/tags.txt"
    ).decode("utf-8-sig")

    tags = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("\t")
        if len(fields) != 3:
            raise ValueError(f"{folder}: unexpected annotation format.")
        tags.append((float(fields[0]), fields[2].strip()))

    if len(tags) < 4:
        raise ValueError(f"{folder}: insufficient annotations.")

    if tags[0][1] != "start" or tags[-1][1] != "end":
        raise ValueError(f"{folder}: missing start/end annotations.")

    if tags[1][1] != "1" or tags[-2][1] != "1":
        raise ValueError(f"{folder}: unexpected walking-path endpoints.")

    if any(b[0] <= a[0] for a, b in zip(tags, tags[1:])):
        raise ValueError(f"{folder}: annotation times are not increasing.")

    video_start = tags[0][0]

    # Preserve the authors' two-decimal rounding convention.
    path_start = round(tags[1][0] - video_start, 2)
    duration = round(tags[-2][0] - video_start - path_start, 2)

    if duration <= 0:
        raise ValueError(f"{folder}: invalid walking duration.")

    return path_start, duration


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Results already exist: {OUTPUT}\n"
            "Change OUTPUT to a new folder to preserve the previous run."
        )

    download()
    rows = []

    with zipfile.ZipFile(ARCHIVE) as archive:
        for setup in SETUPS:
            for protected in (False, True):
                condition = "IRShield" if protected else "No IRS"
                suffix = "_sur" if protected else ""
                prefix = f"q2_{setup}{suffix}"

                _, reference = load_trace(archive, prefix + "_ref")
                time_s, movement = load_trace(archive, prefix + "_walk")
                start, duration = walking_interval(
                    archive, prefix + "_walk"
                )

                aligned_time = time_s - start
                keep = (aligned_time >= 0) & (aligned_time <= duration)
                walking = movement[keep]

                if not len(walking):
                    raise ValueError(f"{prefix}: no annotated walking samples.")

                if aligned_time[0] > 0 or aligned_time[-1] < duration:
                    raise ValueError(
                        f"{prefix}: trace does not cover the full path."
                    )

                median = float(np.median(reference))
                mad = float(np.median(np.abs(reference - median)))

                for factor in FACTORS:
                    threshold = median + factor * mad

                    motion_alerts = int(np.sum(walking > threshold))
                    reference_alerts = int(np.sum(reference > threshold))

                    rows.append({
                        "setup": setup,
                        "condition": condition,
                        "threshold_factor": factor,
                        "threshold": threshold,
                        "reference_median": median,
                        "reference_mad": mad,
                        "annotated_path_duration_s": duration,
                        "motion_samples": len(walking),
                        "motion_alert_samples": motion_alerts,
                        "motion_alert_fraction": motion_alerts / len(walking),
                        "reference_samples": len(reference),
                        "reference_alert_samples": reference_alerts,
                        "reference_calibration_alert_fraction":
                            reference_alerts / len(reference),
                    })

                print(f"Evaluated {setup}: {condition}", flush=True)

    comparisons = []
    for setup in SETUPS:
        for factor in FACTORS:
            pair = {
                row["condition"]: row
                for row in rows
                if row["setup"] == setup
                and row["threshold_factor"] == factor
            }

            clean = pair["No IRS"]["motion_alert_fraction"]
            protected = pair["IRShield"]["motion_alert_fraction"]

            comparisons.append({
                "setup": setup,
                "threshold_factor": factor,
                "no_irs_motion_alert_fraction": clean,
                "irshield_motion_alert_fraction": protected,
                "reduction_percentage_points": 100 * (clean - protected),
            })

    report = {
        "source": "https://zenodo.org/records/6367411",
        "attribution": "Paul Staat and collaborators",
        "license": "CC BY 4.0",
        "archive_md5": EXPECTED_MD5,
        "numpy_version": np.__version__,
        "method": {
            "statistic": "Published processed CSI statistic, antenna averaged",
            "threshold": "reference median + factor * unscaled reference MAD",
            "factors": FACTORS,
            "motion_interval": "First numbered path marker to last numbered marker",
            "time_alignment": "Authors' plotting and annotation convention",
            "aggregation": "Fraction of retained processed samples above threshold",
        },
        "results": rows,
        "comparisons": comparisons,
        "limitations": [
            "Uses published processed statistics; not recomputed in this script.",
            "Motion alert fractions are descriptive sample-level measurements.",
            "Overlapping windows are correlated, not independent trials.",
            "Walking annotations include a turn and are not frame-level labels.",
            "Reference alert rates reuse threshold-calibration data.",
            "No independent false-positive rate is established.",
            "Each condition uses its own reference-derived threshold.",
            "Lowering the threshold is not a retrained adaptive ML attacker.",
            "Offline subcarrier selection uses the complete recording.",
            "Not a reproduction of the paper's aggregate detection-rate claim.",
            "No new physical defense or home protection is demonstrated.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "motion_metrics.csv", rows)
    write_csv(OUTPUT / "condition_comparison.csv", comparisons)
    (OUTPUT / "motion_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nWalking-interval alert fractions:")
    print("Setup       Factor    No IRS    IRShield    Reduction (pp)")
    for row in comparisons:
        print(
            f"{row['setup']:<12}"
            f"{row['threshold_factor']:>5}"
            f"{row['no_irs_motion_alert_fraction']:>10.1%}"
            f"{row['irshield_motion_alert_fraction']:>12.1%}"
            f"{row['reduction_percentage_points']:>16.1f}"
        )

    print(f"\nSaved results to {OUTPUT}")
    print(
        "Reference crossing rates are calibration diagnostics, "
        "not independent false-alarm rates."
    )


if __name__ == "__main__":
    main()