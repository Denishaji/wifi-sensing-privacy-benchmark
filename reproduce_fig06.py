"""Reproduce IRShield Figure 6 traces from published processed measurements.

Source: https://zenodo.org/records/6367411
Authors: Paul Staat and collaborators.
Dataset license: CC BY 4.0.

This reproduces published measurements, not a new physical defense.
"""

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "Fig06_IRShield_illustration.zip"
OUTPUT = ROOT / "results" / "fig06"

URL = (
    "https://zenodo.org/api/records/6367411/files/"
    "Fig06_IRShield_illustration.zip/content"
)
EXPECTED_BYTES = 22888783
EXPECTED_MD5 = "86221eee630f004f0e3004a922bbfc12"

CONDITIONS = [
    ("ref", "No IRS", "#2563eb"),
    ("skip0", "IRS: hold probability 0", "#dc2626"),
    ("wskip60", "IRS: hold probability 0.6", "#059669"),
]


def checksum(path):
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
            and checksum(ARCHIVE) == EXPECTED_MD5
        ):
            print("Using previously verified archive.")
            return
        raise ValueError(
            f"Existing archive failed verification: {ARCHIVE}"
        )

    partial = ARCHIVE.with_suffix(".zip.part")
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "wifi-privacy-reproduction/0.1"},
    )

    print("Downloading the 22.9 MB Figure 6 archive...", flush=True)

    with urllib.request.urlopen(request, timeout=90) as response:
        with partial.open("wb") as stream:
            total = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > EXPECTED_BYTES:
                    raise ValueError("Response exceeds expected archive size.")
                stream.write(block)

    if partial.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Incomplete download. Run the script again.")

    if checksum(partial) != EXPECTED_MD5:
        raise ValueError("Downloaded archive checksum does not match.")

    partial.replace(ARCHIVE)
    print("Download checksum verified.")


def load_traces():
    traces = []

    with zipfile.ZipFile(ARCHIVE) as archive:
        for tag, label, color in CONDITIONS:
            prefix = (
                "Fig06_IRShield_illustration/data/"
                f"p1_an0_at0_{tag}/"
            )

            raw = np.load(
                io.BytesIO(archive.read(prefix + "wrt1.npy")),
                allow_pickle=False,
            )
            processed = np.load(
                io.BytesIO(archive.read(prefix + "wrt1_processed.npy")),
                allow_pickle=False,
            )

            if raw.dtype.names is None:
                raise ValueError("Expected a structured raw CSI array.")
            if not {"timestamp", "csi"}.issubset(raw.dtype.names):
                raise ValueError("Missing timestamp or CSI fields.")
            if raw["csi"].shape[1:] != (3, 3, 56):
                raise ValueError("Unexpected antenna/subcarrier layout.")
            if processed.ndim != 3 or processed.shape[:2] != (3, 3):
                raise ValueError("Unexpected processed-statistic layout.")
            if not np.isfinite(processed).all():
                raise ValueError("Processed data contains nonfinite values.")

            # Match the authors' fig_gen.py:
            # omit the first three processed samples, average antenna pairs,
            # and use their timestamp alignment.
            offset = 3
            count = processed.shape[-1]

            if not offset < count <= len(raw):
                raise ValueError("Invalid processed/raw array alignment.")

            timestamps = raw["timestamp"]
            time_s = (
                timestamps[offset:count] - timestamps[offset]
            ).astype(np.float64) / 1e6

            values = processed[:, :, offset:].mean(axis=(0, 1))

            if np.any(np.diff(time_s) <= 0):
                raise ValueError("Timestamps must increase.")

            # Match the published figure's displayed time range.
            keep = (time_s >= 0) & (time_s <= 60)
            time_s = time_s[keep]
            values = values[keep]

            if len(values) == 0:
                raise ValueError("No samples in the figure time range.")

            traces.append({
                "condition": tag,
                "label": label,
                "color": color,
                "time": time_s,
                "value": values,
                "raw_packets": len(raw),
            })

    return traces


def save_svg(traces):
    """Write a standalone graph without installing a plotting package."""
    left, top = 90, 70
    width, height = 820, 340
    ymin, ymax = 0.002, 0.012

    def px(t):
        return left + float(t) / 60 * width

    def py(value):
        return top + height * (1 - (float(value) - ymin) / (ymax - ymin))

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="960" height="560" viewBox="0 0 960 560">',
        '<rect width="960" height="560" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#172033">',
        '<text x="90" y="30" font-size="21">'
        'IRShield: published measured sensing statistics</text>',
        '<text x="90" y="51" font-size="12">'
        'Figure 6 trace reproduction using authors’ processed data</text>',
        '<defs><clipPath id="plot-area">',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}"/>',
        '</clipPath></defs>',
    ]

    for value in np.linspace(ymin, ymax, 6):
        y = py(value)
        svg.extend([
            f'<line x1="{left}" y1="{y}" x2="{left + width}" '
            f'y2="{y}" stroke="#e2e8f0"/>',
            f'<text x="{left - 10}" y="{y + 4}" '
            f'text-anchor="end" font-size="12">{value:.3f}</text>',
        ])

    for seconds in range(0, 61, 10):
        x = px(seconds)
        svg.append(
            f'<text x="{x}" y="{top + height + 22}" '
            f'text-anchor="middle" font-size="12">{seconds}</text>'
        )

    for trace in traces:
        points = " ".join(
            f"{px(t):.2f},{py(v):.2f}"
            for t, v in zip(trace["time"], trace["value"])
        )
        svg.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{trace["color"]}" stroke-width="1.2" '
            'clip-path="url(#plot-area)"/>'
        )

    svg.extend([
        '<text x="500" y="458" text-anchor="middle" '
        'font-size="14">Time (seconds)</text>',
        '<text x="22" y="240" transform="rotate(-90 22 240)" '
        'text-anchor="middle" font-size="14">'
        'Mean sensing statistic</text>',
    ])

    for index, trace in enumerate(traces):
        y = 485 + index * 22
        svg.extend([
            f'<line x1="95" y1="{y}" x2="125" y2="{y}" '
            f'stroke="{trace["color"]}" stroke-width="3"/>',
            f'<text x="135" y="{y + 4}" font-size="13">'
            f'{escape(trace["label"])}</text>',
        ])

    svg.append("</g></svg>")
    (OUTPUT / "fig06_reproduction.svg").write_text(
        "\n".join(svg), encoding="utf-8"
    )


def main():
    download()
    traces = load_traces()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with (OUTPUT / "traces.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "time_seconds", "sensing_statistic"])
        for trace in traces:
            for t, value in zip(trace["time"], trace["value"]):
                writer.writerow([trace["condition"], float(t), float(value)])

    summaries = []
    for trace in traces:
        values = trace["value"]
        summary = {
            "condition": trace["label"],
            "raw_packets": trace["raw_packets"],
            "plotted_samples": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        summaries.append(summary)
        print(
            f"{trace['label']}: "
            f"mean={summary['mean']:.6f}, "
            f"median={summary['median']:.6f}"
        )

    report = {
        "source": "https://zenodo.org/records/6367411",
        "license": "CC BY 4.0",
        "archive_md5": EXPECTED_MD5,
        "method": (
            "Reproduce Figure 6 traces from published processed arrays; "
            "match authors' antenna averaging and timestamp alignment."
        ),
        "raw_csi_processing_recomputed": False,
        "numpy_version": np.__version__,
        "conditions": summaries,
        "limitations": [
            "Published measurements, not measurements from our hardware.",
            "No occupancy or motion classification accuracy calculated.",
            "No adaptive-attacker evaluation.",
            "No new controller or physical privacy protection validated.",
        ],
    }

    (OUTPUT / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    save_svg(traces)

    print(f"\nSaved graph, CSV and JSON to: {OUTPUT}")


if __name__ == "__main__":
    main()