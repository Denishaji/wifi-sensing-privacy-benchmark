"""Recalculate IRShield Figure 6 sensing statistics from raw CSI.

Algorithm follows the published IRShield processing code:
https://zenodo.org/records/6367411
Dataset/code attribution: Paul Staat and collaborators, CC BY 4.0.

This is an offline reproduction, not a deployable privacy defense.
"""

import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "Fig06_IRShield_illustration.zip"
OUTPUT = ROOT / "results" / "raw_csi_verification"

EXPECTED_MD5 = "86221eee630f004f0e3004a922bbfc12"
RTOL = 1e-6
ATOL = 1e-10

CONDITIONS = {
    "ref": "No IRS",
    "skip0": "IRS: hold probability 0",
    "wskip60": "IRS: hold probability 0.6",
}


def verify_archive():
    if not ARCHIVE.is_file():
        raise FileNotFoundError(
            f"Archive not found: {ARCHIVE}\n"
            "Run reproduce_fig06.py first."
        )

    digest = hashlib.md5()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    if digest.hexdigest() != EXPECTED_MD5:
        raise ValueError("Archive checksum does not match published metadata.")


def select_subcarriers(smoothed, count=28):
    """Match the authors' offline correlation-ranking procedure."""
    correlation = np.corrcoef(smoothed, rowvar=False)

    if not np.isfinite(correlation).all():
        raise ValueError(
            "Undefined subcarrier correlations; cannot reproduce "
            "the reference selection reliably."
        )

    np.fill_diagonal(correlation, 0)
    correlation *= np.tri(*correlation.shape)

    # Preserve the authors' sorting and tie-handling procedure.
    ranked = np.flip(np.argsort(correlation.flatten()))
    selected = []

    for flat_index in ranked:
        if len(selected) >= count:
            break

        first, second = divmod(
            int(flat_index), correlation.shape[0]
        )

        if first not in selected:
            selected.append(first)

        if len(selected) < count and second not in selected:
            selected.append(second)

    if len(selected) != count:
        raise ValueError("Could not select the expected subcarrier count.")

    return selected


def recompute(raw):
    if raw.dtype.names is None:
        raise ValueError("Expected a structured CSI array.")

    if not {"timestamp", "csi"}.issubset(raw.dtype.names):
        raise ValueError("Missing timestamp or CSI fields.")

    csi = raw["csi"]

    if csi.shape[1:] != (3, 3, 56):
        raise ValueError(f"Unexpected CSI shape: {csi.shape}")

    if not np.isfinite(csi).all():
        raise ValueError("Raw CSI contains NaN or infinity.")

    # Convert before subtraction to avoid unsigned timestamp wraparound.
    timestamps = raw["timestamp"].astype(np.float64)
    intervals = np.diff(timestamps)

    if len(intervals) == 0 or np.any(intervals <= 0):
        raise ValueError("Timestamps must increase strictly.")

    median_interval_us = float(np.median(intervals))

    # Match the authors: approximate one second using median packet spacing.
    window = int(np.ceil(1_000_000 / median_interval_us))

    if not 2 <= window <= len(raw):
        raise ValueError(f"Invalid window size: {window}")

    amplitude = np.abs(csi)
    norms = np.linalg.norm(amplitude, axis=-1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero-amplitude CSI vector.")

    normalized = amplitude / norms

    n_windows = len(raw) - window + 1
    calculated = np.empty((3, 3, n_windows), dtype=np.float64)
    selected_by_pair = {}

    for rx in range(3):
        for tx in range(3):
            # Population standard deviation, matching np.std's default.
            rolling_std = np.array([
                np.std(
                    normalized[start:start + window, rx, tx],
                    axis=0,
                    ddof=0,
                )
                for start in range(n_windows)
            ])

            # The original implementation smooths across SUBCARRIERS,
            # not across time. Preserve this for faithful reproduction.
            smoothed = np.apply_along_axis(
                lambda values: np.convolve(
                    values,
                    np.ones(3) / 3,
                    mode="same",
                ),
                axis=-1,
                arr=rolling_std,
            )

            selected = select_subcarriers(smoothed)

            calculated[rx, tx] = smoothed[:, selected].mean(axis=-1)
            selected_by_pair[f"rx{rx}_tx{tx}"] = selected

    return calculated, {
        "raw_packets": len(raw),
        "window_packets": window,
        "median_packet_interval_us": median_interval_us,
        "processed_windows": n_windows,
        "selected_subcarriers": selected_by_pair,
    }


def main():
    verify_archive()

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Results already exist: {OUTPUT}\n"
            "Change OUTPUT to a new folder to preserve the previous run."
        )

    results = []
    recalculated_arrays = {}

    with zipfile.ZipFile(ARCHIVE) as archive:
        for tag, label in CONDITIONS.items():
            print(f"Recalculating: {label}", flush=True)

            prefix = (
                "Fig06_IRShield_illustration/data/"
                f"p1_an0_at0_{tag}/"
            )

            raw = np.load(
                io.BytesIO(archive.read(prefix + "wrt1.npy")),
                allow_pickle=False,
            )
            reference = np.load(
                io.BytesIO(
                    archive.read(prefix + "wrt1_processed.npy")
                ),
                allow_pickle=False,
            )

            calculated, details = recompute(raw)

            if calculated.shape != reference.shape:
                raise ValueError(
                    f"{label}: shape mismatch: "
                    f"{calculated.shape} versus {reference.shape}"
                )

            if not np.isfinite(reference).all():
                raise ValueError("Reference contains nonfinite values.")

            difference = calculated - reference

            passed = bool(np.allclose(
                calculated,
                reference,
                rtol=RTOL,
                atol=ATOL,
            ))

            result = {
                "condition": label,
                **details,
                "max_absolute_error": float(
                    np.max(np.abs(difference))
                ),
                "root_mean_square_error": float(
                    np.sqrt(np.mean(difference ** 2))
                ),
                "matches_reference": passed,
            }

            results.append(result)
            recalculated_arrays[tag] = calculated

            status = "PASS" if passed else "FAIL"
            print(
                f"  {status}: window={details['window_packets']} packets, "
                f"max error={result['max_absolute_error']:.3e}",
                flush=True,
            )

    all_passed = all(item["matches_reference"] for item in results)

    report = {
        "experiment": "IRShield Figure 6 raw CSI reproduction",
        "source": "https://zenodo.org/records/6367411",
        "attribution": "Paul Staat and collaborators",
        "source_license": "CC BY 4.0",
        "archive_md5": EXPECTED_MD5,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "comparison": {
            "relative_tolerance": RTOL,
            "absolute_tolerance": ATOL,
            "all_conditions_passed": all_passed,
        },
        "conditions": results,
        "limitations": [
            "Reproduces the published algorithm on published recordings.",
            "Not an independent physical experiment.",
            "Subcarrier selection uses the complete recording offline.",
            "The method has not been converted to causal streaming.",
            "No motion-detection accuracy or privacy guarantee is measured.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT / "recomputed_statistics.npz",
        **recalculated_arrays,
    )

    report_path = OUTPUT / "verification.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nVerification report: {report_path}")

    if not all_passed:
        raise SystemExit(
            "Reference comparison failed. Inspect the errors before proceeding."
        )

    print("PASS: all three raw-CSI calculations match the reference.")


if __name__ == "__main__":
    main()
    