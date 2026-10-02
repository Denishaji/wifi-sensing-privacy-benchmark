"""Fresh-seed synthetic attacker comparison.

No physical privacy protection is claimed.
Requires simulate_surface.py in the same folder.
"""

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

import simulate_surface as sim


OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "attacker_comparison_v1"
)

SEEDS = [101, 202, 303]
SIZES = {"train": 160, "calibration": 200, "test": 200}
STRENGTHS = [0.1, 0.35, 0.7]
TARGET = 0.05

MODELS = (
    "knn_static",
    "logistic_temporal",
    "gaussian_temporal",
)


def scenarios():
    return [("no_surface", 0.0)] + [
        (policy, strength)
        for strength in STRENGTHS
        for policy in ("independent", "persistent")
    ]


def temporal_features(received):
    amplitude = np.abs(received)
    centered = amplitude - amplitude.mean(axis=0)

    power = np.abs(
        np.fft.rfft(centered, axis=0)
    )[1:] ** 2

    total = power.sum(axis=0) + 1e-12

    # Frequency bins are relative to the episode, not physical Hz.
    extras = [
        power[:4].sum(axis=0) / total,
        power[4:12].sum(axis=0) / total,
    ]

    for lag in (1, 4):
        first = centered[:-lag]
        second = centered[lag:]

        denominator = np.sqrt(
            np.mean(first ** 2, axis=0)
            * np.mean(second ** 2, axis=0)
        ) + 1e-12

        extras.append(
            np.mean(first * second, axis=0) / denominator
        )

    return np.concatenate([
        sim.extract_features(received),
        *extras,
    ])


def generate(seed):
    rng = np.random.default_rng(seed)

    direct = sim.complex_normal(rng, (sim.CHANNELS,))
    motion = sim.complex_normal(rng, (sim.CHANNELS,))
    reflected = sim.complex_normal(
        rng, (sim.ELEMENTS, sim.CHANNELS)
    ) / np.sqrt(sim.ELEMENTS)

    data = {item: {} for item in scenarios()}
    labels = {}

    for split_id, (split, count) in enumerate(SIZES.items()):
        y = np.tile([0, 1], count // 2)
        rng.shuffle(y)
        labels[split] = y

        arrays = {item: [] for item in scenarios()}

        for episode, moving in enumerate(y):
            phase = rng.uniform(0, 2 * np.pi)
            frequency = rng.uniform(1, 4)

            wave = np.sin(
                np.linspace(0, 2 * np.pi, sim.PACKETS)
                * frequency + phase
            )

            channel = (
                direct[None, :]
                + 0.10 * sim.complex_normal(
                    rng, (1, sim.CHANNELS)
                )
                + 0.12 * moving
                * wave[:, None]
                * motion[None, :]
            )

            noise = 0.03 * sim.complex_normal(
                rng, (sim.PACKETS, sim.CHANNELS)
            )

            # Each condition receives the same underlying episode.
            for policy, strength in scenarios():
                policy_id = {
                    "no_surface": 0,
                    "independent": 1,
                    "persistent": 2,
                }[policy]

                control_rng = np.random.default_rng(
                    np.random.SeedSequence([
                        seed,
                        split_id,
                        episode,
                        policy_id,
                        99,
                    ])
                )

                states = sim.controller_states(
                    control_rng, policy
                )

                received = (
                    channel
                    + strength * (states @ reflected)
                    + noise
                )

                arrays[(policy, strength)].append(
                    temporal_features(received)
                )

        for item in scenarios():
            data[item][split] = np.asarray(arrays[item])

    return data, labels


def continuous_scores(name, train, y, evaluation):
    # Fit normalization exclusively on training episodes.
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-6)

    x = (train - mean) / scale
    z = (evaluation - mean) / scale

    if name == "logistic_temporal":
        # Fixed settings: no calibration or test hyperparameter tuning.
        weights = np.zeros(x.shape[1])
        bias = 0.0

        for _ in range(500):
            logits = np.clip(
                x @ weights + bias, -40, 40
            )
            probability = 1 / (1 + np.exp(-logits))
            error = probability - y

            weights -= 0.05 * (
                x.T @ error / len(y)
                + 0.01 * weights
            )
            bias -= 0.05 * error.mean()

        # Continuous decision score, not rounded class votes.
        return z @ weights + bias

    if name == "gaussian_temporal":
        # Diagonal Gaussian likelihood ratio.
        # Shrink class variances toward pooled training variance.
        pooled = x.var(axis=0)
        likelihoods = []

        for label in (0, 1):
            group = x[y == label]
            mu = group.mean(axis=0)

            variance = np.maximum(
                0.9 * group.var(axis=0) + 0.1 * pooled,
                1e-4,
            )

            likelihoods.append(
                -0.5 * np.sum(
                    np.log(variance)
                    + (z - mu) ** 2 / variance,
                    axis=1,
                )
            )

        return likelihoods[1] - likelihoods[0]

    raise ValueError(f"Unknown model: {name}")


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Results already exist: {OUTPUT}\n"
            "Choose a new OUTPUT folder to preserve previous results."
        )

    if (
        sim.PACKETS,
        sim.CHANNELS,
        sim.ELEMENTS,
    ) != (64, 16, 16):
        raise ValueError(
            "The original simulator dimensions have changed."
        )

    rows = []

    for seed in SEEDS:
        print(
            f"Evaluating fresh environment {seed}...",
            flush=True,
        )

        data, labels = generate(seed)

        for (policy, strength), parts in data.items():
            evaluation = np.concatenate([
                parts["calibration"],
                parts["test"],
            ])

            for model in MODELS:
                if model == "knn_static":
                    # Original amplitude-statistics baseline.
                    n_static = 3 * sim.CHANNELS

                    scores = sim.attacker_scores(
                        parts["train"][:, :n_static],
                        labels["train"],
                        evaluation[:, :n_static],
                    )
                else:
                    scores = continuous_scores(
                        model,
                        parts["train"],
                        labels["train"],
                        evaluation,
                    )

                if not np.isfinite(scores).all():
                    raise ValueError(
                        "Nonfinite attacker scores."
                    )

                cal, test = np.split(
                    scores,
                    [len(labels["calibration"])],
                )

                cal_reference = cal[
                    labels["calibration"] == 0
                ]

                threshold = float(np.quantile(
                    cal_reference,
                    1 - TARGET,
                    method="higher",
                ))

                prediction = test > threshold
                reference = labels["test"] == 0
                moving = labels["test"] == 1

                rows.append({
                    "seed": seed,
                    "policy": policy,
                    "surface_strength": strength,
                    "attacker": model,
                    "threshold": threshold,
                    "calibration_reference_alert_rate": float(
                        np.mean(cal_reference > threshold)
                    ),
                    "test_reference_alert_rate": float(
                        prediction[reference].mean()
                    ),
                    "test_motion_alert_rate": float(
                        prediction[moving].mean()
                    ),
                    "test_reference_alerts": int(
                        prediction[reference].sum()
                    ),
                    "test_reference_episodes": int(
                        reference.sum()
                    ),
                    "test_motion_alerts": int(
                        prediction[moving].sum()
                    ),
                    "test_motion_episodes": int(
                        moving.sum()
                    ),
                    "test_roc_auc": sim.auc_score(
                        labels["test"], test
                    ),
                })

        del data

    print(
        "\nPolicy       Strength  Attacker             "
        "Mean AUC  AUC range"
    )

    for policy, strength in scenarios():
        for model in MODELS:
            values = [
                row["test_roc_auc"]
                for row in rows
                if row["policy"] == policy
                and row["surface_strength"] == strength
                and row["attacker"] == model
            ]

            print(
                f"{policy:<13}"
                f"{strength:>8.2f}  "
                f"{model:<21}"
                f"{np.mean(values):>8.3f}  "
                f"{min(values):.3f}-{max(values):.3f}"
            )

    report = {
        "scope": "SYNTHETIC ATTACKER COMPARISON",
        "seeds": SEEDS,
        "split_sizes": SIZES,
        "surface_strengths": STRENGTHS,
        "calibration_target": TARGET,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "simulator_sha256": hashlib.sha256(
            Path(sim.__file__).read_bytes()
        ).hexdigest(),
        "feature_definition": (
            "Amplitude mean, standard deviation, difference RMS; "
            "relative spectral energy in bins 1-4 and 5-12; "
            "lag-1 and lag-4 correlation."
        ),
        "training": {
            "knn": (
                "Original 5-neighbor detector, static features"
            ),
            "logistic": (
                "500 steps, learning rate 0.05, "
                "L2 coefficient 0.01"
            ),
            "gaussian": (
                "Diagonal class variances, "
                "0.1 pooled variance shrinkage"
            ),
            "scaling": "Training data only",
            "adaptation": (
                "Separately trained under each surface condition"
            ),
        },
        "results": rows,
        "limitations": [
            "Synthetic channels, not hardware validation.",
            "Fixed geometry within each seed.",
            "No cross-environment transfer test.",
            "Three environments do not establish broad robustness.",
            "New attackers change both model and features versus baseline.",
            "Nearest-neighbor scores retain ties.",
            "Calibration targets do not guarantee test false-alarm rates.",
            "Temporal features use a complete episode.",
            "No streaming latency or real communication performance measured.",
            "These seeds become development-exposed after inspection.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    with (OUTPUT / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT / "attacker_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nSaved results to {OUTPUT}")


if __name__ == "__main__":
    main()