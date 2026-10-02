"""Memory-efficient held-out evaluation of published IRShield recordings.

Source: https://zenodo.org/records/6367411
Attribution: Paul Staat and collaborators, CC BY 4.0.

Requires:
    evaluate_motion.py
    verify_raw_csi.py

All temporary files and results are stored on E:.
"""

import gc
import json
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from evaluate_motion import (
    ARCHIVE,
    NAME,
    SETUPS,
    EXPECTED_MD5,
    file_md5,
    walking_interval,
    write_csv,
)
from verify_raw_csi import select_subcarriers


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "holdout_v1"
SCRATCH = ROOT / ".tmp"

TARGETS = [0.05, 0.10]
GAP_HALF_SECONDS = 2.0


def load_raw(archive, folder):
    """Stream extraction to E:, then normalize in small batches."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    member = f"{NAME}/data/{folder}/wrt1.npy"

    with tempfile.TemporaryDirectory(
        prefix="csi_",
        dir=str(SCRATCH),
    ) as temporary:
        local_file = Path(temporary) / "recording.npy"

        with archive.open(member, "r") as source:
            with local_file.open("wb") as destination:
                shutil.copyfileobj(
                    source,
                    destination,
                    length=1024 * 1024,
                )

        raw = np.load(
            local_file,
            mmap_mode="r",
            allow_pickle=False,
        )

        try:
            if raw.dtype.names is None:
                raise ValueError(
                    f"{folder}: expected structured CSI data."
                )

            if not {"timestamp", "csi"}.issubset(raw.dtype.names):
                raise ValueError(
                    f"{folder}: missing CSI or timestamps."
                )

            if (
                len(raw) < 4
                or raw["csi"].shape[1:] != (3, 3, 56)
            ):
                raise ValueError(
                    f"{folder}: unexpected CSI shape."
                )

            timestamps = raw["timestamp"].astype(np.float64)

            if not np.isfinite(timestamps).all():
                raise ValueError(
                    f"{folder}: invalid timestamps."
                )

            if np.any(np.diff(timestamps) <= 0):
                raise ValueError(
                    f"{folder}: timestamps must increase."
                )

            # Match the original plotting time origin.
            time_s = (timestamps - timestamps[3]) / 1e6

            amplitude_dtype = np.abs(raw["csi"][:1]).dtype
            normalized = np.empty(
                (len(raw), 3, 3, 56),
                dtype=amplitude_dtype,
            )

            for start in range(0, len(raw), 512):
                stop = min(start + 512, len(raw))
                batch = np.abs(raw["csi"][start:stop])

                if not np.isfinite(batch).all():
                    raise ValueError(
                        f"{folder}: invalid CSI."
                    )

                norm = np.linalg.norm(
                    batch,
                    axis=-1,
                    keepdims=True,
                )

                if np.any(norm == 0):
                    raise ValueError(
                        f"{folder}: zero-amplitude CSI vector."
                    )

                np.divide(
                    batch,
                    norm,
                    out=normalized[start:stop],
                )

            return time_s, normalized

        finally:
            # Windows requires closing the map before deleting the file.
            raw._mmap.close()


def window_features(amplitude, window):
    """Build non-overlapping windows without crossing partitions."""
    starts = np.arange(
        0,
        len(amplitude) - window + 1,
        window,
    )

    if len(starts) < 2:
        raise ValueError("Too few complete windows.")

    # Allocate once instead of collecting a list of arrays.
    statistics = np.empty(
        (len(starts), 3, 3, 56),
        dtype=amplitude.dtype,
    )

    for index, start in enumerate(starts):
        statistics[index] = amplitude[
            start:start + window
        ].std(axis=0, ddof=0)

    # Original processing smooths across subcarriers, not time.
    statistics = np.apply_along_axis(
        lambda values: np.convolve(
            values,
            np.ones(3) / 3,
            mode="same",
        ),
        axis=-1,
        arr=statistics,
    )

    return statistics, starts


def fit_selection(calibration):
    """Choose subcarriers using calibration data only."""
    return {
        f"{rx},{tx}": select_subcarriers(
            calibration[:, rx, tx]
        )
        for rx in range(3)
        for tx in range(3)
    }


def sensing_score(features, selected):
    pair_scores = []

    for rx in range(3):
        for tx in range(3):
            indices = selected[f"{rx},{tx}"]
            pair_scores.append(
                features[:, rx, tx, indices].mean(axis=-1)
            )

    result = np.mean(pair_scores, axis=0)

    if not np.isfinite(result).all():
        raise ValueError("Nonfinite sensing score.")

    return result


def alert_counts(scores, threshold):
    count = int(np.sum(scores > threshold))
    return count, float(count / len(scores))


def evaluate_condition(archive, setup, protected):
    """Release reference CSI before loading walking CSI."""
    condition = "IRShield" if protected else "No IRS"
    folder = f"q2_{setup}" + ("_sur" if protected else "")

    print(
        f"Processing {setup}: {condition}",
        flush=True,
    )

    time_s, amplitude = load_raw(
        archive,
        folder + "_ref",
    )

    midpoint = float((time_s[0] + time_s[-1]) / 2)

    # Timestamps are sorted. Slices avoid boolean-index array copies.
    calibration_stop = int(np.searchsorted(
        time_s,
        midpoint - GAP_HALF_SECONDS,
        side="left",
    ))
    heldout_start = int(np.searchsorted(
        time_s,
        midpoint + GAP_HALF_SECONDS,
        side="left",
    ))

    if (
        calibration_stop < 3
        or len(time_s) - heldout_start < 3
    ):
        raise ValueError("Insufficient reference data.")

    calibration_time = time_s[:calibration_stop]

    # Estimate packet-window size from calibration only.
    window = int(np.ceil(
        1 / np.median(np.diff(calibration_time))
    ))

    if window < 2:
        raise ValueError("Invalid packet-window size.")

    calibration, _ = window_features(
        amplitude[:calibration_stop],
        window,
    )
    heldout, _ = window_features(
        amplitude[heldout_start:],
        window,
    )

    selected = fit_selection(calibration)
    calibration_scores = sensing_score(calibration, selected)
    heldout_scores = sensing_score(heldout, selected)

    calibration_end_s = float(time_s[calibration_stop - 1])
    heldout_start_s = float(time_s[heldout_start])

    # These arrays are no longer needed.
    del amplitude, calibration, heldout, calibration_time, time_s
    gc.collect()

    motion_time, motion_amplitude = load_raw(
        archive,
        folder + "_walk",
    )
    motion_features, starts = window_features(
        motion_amplitude,
        window,
    )

    del motion_amplitude
    gc.collect()

    path_start, duration = walking_interval(
        archive,
        folder + "_walk",
    )
    path_end = path_start + duration

    # Evaluate only windows wholly inside the annotated path.
    keep = (
        (motion_time[starts] >= path_start)
        & (motion_time[starts + window - 1] <= path_end)
    )

    if not np.any(keep):
        raise ValueError(
            f"{folder}: no complete walking windows."
        )

    motion_scores = sensing_score(
        motion_features[keep],
        selected,
    )

    audit = {
        "setup": setup,
        "condition": condition,
        "window_packets": window,
        "calibration_end_s": calibration_end_s,
        "heldout_start_s": heldout_start_s,
        "calibration_windows": len(calibration_scores),
        "heldout_reference_windows": len(heldout_scores),
        "walking_windows": len(motion_scores),
        "walking_path_start_s": path_start,
        "walking_path_end_s": path_end,
        "selected_subcarriers": selected,
    }

    score_rows = []

    for split, scores in [
        ("calibration", calibration_scores),
        ("heldout_reference", heldout_scores),
        ("walking", motion_scores),
    ]:
        for index, value in enumerate(scores):
            score_rows.append({
                "setup": setup,
                "condition": condition,
                "split": split,
                "window_index": index,
                "score": float(value),
            })

    metric_rows = []

    for target in TARGETS:
        threshold = float(np.quantile(
            calibration_scores,
            1 - target,
            method="higher",
        ))

        cal_n, cal_rate = alert_counts(
            calibration_scores,
            threshold,
        )
        ref_n, ref_rate = alert_counts(
            heldout_scores,
            threshold,
        )
        motion_n, motion_rate = alert_counts(
            motion_scores,
            threshold,
        )

        metric_rows.append({
            "setup": setup,
            "condition": condition,
            "calibration_target": target,
            "threshold": threshold,
            "calibration_windows": len(calibration_scores),
            "calibration_alerts": cal_n,
            "calibration_alert_rate": cal_rate,
            "heldout_reference_windows": len(heldout_scores),
            "heldout_reference_alerts": ref_n,
            "heldout_reference_alert_rate": ref_rate,
            "walking_windows": len(motion_scores),
            "walking_alerts": motion_n,
            "walking_alert_rate": motion_rate,
        })

    print(
        f"Completed {setup}: {condition}",
        flush=True,
    )

    return metric_rows, audit, score_rows


def main():
    if not ARCHIVE.is_file():
        raise FileNotFoundError(
            f"Run evaluate_motion.py first. Missing: {ARCHIVE}"
        )

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Results already exist: {OUTPUT}\n"
            "Change OUTPUT to a new folder to preserve earlier results."
        )

    print("Verifying archive checksum...", flush=True)

    if file_md5(ARCHIVE) != EXPECTED_MD5:
        raise ValueError("Archive checksum mismatch.")

    rows = []
    audits = []
    score_rows = []

    with zipfile.ZipFile(ARCHIVE) as archive:
        for setup in SETUPS:
            for protected in (False, True):
                metrics, audit, scores = evaluate_condition(
                    archive,
                    setup,
                    protected,
                )

                rows.extend(metrics)
                audits.append(audit)
                score_rows.extend(scores)

                gc.collect()

    report = {
        "experiment": "Calibration-only selection with temporal holdout",
        "source": "https://zenodo.org/records/6367411",
        "attribution": "Paul Staat and collaborators",
        "source_license": "CC BY 4.0",
        "archive_md5": EXPECTED_MD5,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "protocol": {
            "calibration":
                "First reference half, excluding 2 s near midpoint",
            "reference_evaluation":
                "Second reference half, excluding 2 s near midpoint",
            "windowing":
                "Non-overlapping, approximately one-second packet windows",
            "subcarrier_selection":
                "Calibration only; frozen for evaluation",
            "threshold_selection":
                "Calibration quantile only",
            "calibration_targets": TARGETS,
            "motion_evaluation":
                "Windows wholly inside annotated walking path",
        },
        "results": rows,
        "audit": audits,
        "limitations": [
            "Reference holdout shares a recording with calibration.",
            "Non-overlapping windows can remain temporally correlated.",
            "Small sample counts limit precision for rare alerts.",
            "Equal calibration targets do not imply equal held-out rates.",
            "Packet-count windows only approximate one second.",
            "Annotation alignment follows the published convention.",
            "This differs from the original overlapping-window protocol.",
            "No retrained machine-learning attacker is evaluated.",
            "Published measurements do not validate our own device.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    write_csv(OUTPUT / "metrics.csv", rows)
    write_csv(OUTPUT / "scores.csv", score_rows)

    report_path = OUTPUT / "holdout_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(
        "\nSetup      Condition  Target  "
        "Held-out ref alerts  Walking alerts"
    )

    for row in rows:
        print(
            f"{row['setup']:<11}"
            f"{row['condition']:<11}"
            f"{row['calibration_target']:>6.0%}"
            f"{row['heldout_reference_alert_rate']:>19.1%}"
            f"{row['walking_alert_rate']:>16.1%}"
        )

    print(f"\nSaved report: {report_path}")
    print(
        "Calibration targets are not guarantees "
        "of held-out false-alarm rates."
    )


if __name__ == "__main__":
    main()