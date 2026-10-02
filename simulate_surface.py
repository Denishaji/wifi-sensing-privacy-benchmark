"""Synthetic reflecting-surface experiment.

All results are SIMULATED.
This model is not calibrated to IRShield or a real home.

Channel:
    received = direct + motion + surface_states @ surface_paths + noise

Surface states are ideal binary reflection coefficients (-1 or +1).
The controller never receives motion labels or attacker measurements.
"""

import csv
import json
import platform
from pathlib import Path

import numpy as np


OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "surface_simulation_v1"
)
SEEDS = [11, 22, 33]
STRENGTHS = [0.1, 0.35, 0.7]

PACKETS = 64
CHANNELS = 16
ELEMENTS = 16
NEIGHBORS = 5

# Independent synthetic episodes within one fixed channel per seed.
SPLIT_SIZES = {
    "train": 160,
    "calibration": 200,
    "test": 200,
}

TARGET_REFERENCE_ALERT_RATE = 0.05


def complex_normal(rng, shape):
    return (
        rng.normal(size=shape)
        + 1j * rng.normal(size=shape)
    ) / np.sqrt(2)


def controller_states(rng, policy):
    """Blind control: depends on random state, never activity labels."""
    if policy == "no_surface":
        return np.zeros((PACKETS, ELEMENTS))

    state = rng.choice([-1.0, 1.0], size=ELEMENTS)
    output = np.empty((PACKETS, ELEMENTS))

    for step in range(PACKETS):
        if policy == "independent":
            state = rng.choice([-1.0, 1.0], size=ELEMENTS)

        elif policy == "persistent":
            # Hold with probability 0.6.
            # Otherwise flip exactly two reflection elements.
            if rng.random() >= 0.6:
                chosen = rng.choice(
                    ELEMENTS, size=2, replace=False
                )
                state[chosen] *= -1

        else:
            raise ValueError(f"Unknown policy: {policy}")

        output[step] = state

    return output


def extract_features(received):
    amplitude = np.abs(received)

    return np.concatenate([
        amplitude.mean(axis=0),
        amplitude.std(axis=0),
        np.sqrt(
            np.mean(np.diff(amplitude, axis=0) ** 2, axis=0)
        ),
    ])


def attacker_scores(train_x, train_y, evaluation_x):
    """Train-scaled 5-nearest-neighbor motion scores."""
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-6)

    train = (train_x - mean) / scale
    evaluation = (evaluation_x - mean) / scale

    scores = []
    for row in evaluation:
        distances = np.sum((train - row) ** 2, axis=1)
        nearest = np.argsort(distances)[:NEIGHBORS]
        scores.append(float(train_y[nearest].mean()))

    return np.asarray(scores)


def auc_score(labels, scores):
    """Pairwise ROC-AUC, including half credit for ties."""
    positive = scores[labels == 1][:, None]
    negative = scores[labels == 0][None, :]

    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both classes are required.")

    return float(np.mean(
        (positive > negative)
        + 0.5 * (positive == negative)
    ))


def scenario_name(policy, strength):
    return f"{policy}_{strength:g}"


def simulate_seed(seed):
    rng = np.random.default_rng(seed)

    scenarios = [("no_surface", 0.0)]
    scenarios += [
        (policy, strength)
        for strength in STRENGTHS
        for policy in ("independent", "persistent")
    ]

    # One synthetic environment per seed.
    direct_attacker = complex_normal(rng, (CHANNELS,))
    motion_path = complex_normal(rng, (CHANNELS,))
    reflected_attacker = (
        complex_normal(rng, (ELEMENTS, CHANNELS))
        / np.sqrt(ELEMENTS)
    )

    # Separate idealized legitimate-receiver channel.
    direct_user = complex_normal(rng, (CHANNELS,))
    reflected_user = (
        complex_normal(rng, (ELEMENTS, CHANNELS))
        / np.sqrt(ELEMENTS)
    )

    # Idealized perfect-channel-knowledge rate proxy at fixed power/noise.
    snr_scale = 100.0
    baseline_rate = float(np.mean(
        np.log2(1 + snr_scale * np.abs(direct_user) ** 2)
    ))

    features = {
        scenario_name(p, s): {}
        for p, s in scenarios
    }
    labels = {}
    communication_ratios = {
        scenario_name(p, s): []
        for p, s in scenarios
    }

    for split_id, (split, count) in enumerate(SPLIT_SIZES.items()):
        y = np.tile([0, 1], count // 2)
        rng.shuffle(y)
        labels[split] = y

        split_features = {
            scenario_name(p, s): []
            for p, s in scenarios
        }

        for episode, moving in enumerate(y):
            phase = rng.uniform(0, 2 * np.pi)
            frequency = rng.uniform(1, 4)

            motion_wave = np.sin(
                np.linspace(0, 2 * np.pi, PACKETS)
                * frequency + phase
            )

            # Same direct channel, motion and noise across policies.
            direct = (
                direct_attacker[None, :]
                + 0.10 * complex_normal(rng, (1, CHANNELS))
                + 0.12 * moving
                * motion_wave[:, None]
                * motion_path[None, :]
            )
            noise = 0.03 * complex_normal(
                rng, (PACKETS, CHANNELS)
            )

            for policy, strength in scenarios:
                key = scenario_name(policy, strength)
                policy_id = {
                    "no_surface": 0,
                    "independent": 1,
                    "persistent": 2,
                }[policy]

                # Reuse the switching sequence across strengths.
                # Randomness is independent of the activity label.
                control_rng = np.random.default_rng(
                    np.random.SeedSequence([
                        seed, split_id, episode, policy_id, 99
                    ])
                )

                states = controller_states(control_rng, policy)
                received = (
                    direct
                    + strength * (states @ reflected_attacker)
                    + noise
                )

                split_features[key].append(
                    extract_features(received)
                )

                if split == "test":
                    user_channel = (
                        direct_user[None, :]
                        + strength * (states @ reflected_user)
                    )
                    rate = float(np.mean(
                        np.log2(
                            1 + snr_scale * np.abs(user_channel) ** 2
                        )
                    ))
                    communication_ratios[key].append(
                        rate / baseline_rate
                    )

        for key in features:
            features[key][split] = np.asarray(
                split_features[key]
            )

    results = []

    for policy, strength in scenarios:
        key = scenario_name(policy, strength)
        data = features[key]

        # Adaptive in this limited sense:
        # fit the detector on data generated under this same policy.
        calibration_scores = attacker_scores(
            data["train"],
            labels["train"],
            data["calibration"],
        )
        test_scores = attacker_scores(
            data["train"],
            labels["train"],
            data["test"],
        )

        calibration_reference = calibration_scores[
            labels["calibration"] == 0
        ]

        threshold = float(np.quantile(
            calibration_reference,
            1 - TARGET_REFERENCE_ALERT_RATE,
            method="higher",
        ))

        # Strict comparison handles ties conservatively.
        test_alerts = test_scores > threshold
        reference_mask = labels["test"] == 0
        motion_mask = labels["test"] == 1

        results.append({
            "seed": seed,
            "policy": policy,
            "surface_strength": strength,
            "calibration_target":
                TARGET_REFERENCE_ALERT_RATE,
            "threshold": threshold,
            "calibration_reference_alert_rate": float(
                np.mean(calibration_reference > threshold)
            ),
            "test_reference_alert_rate": float(
                test_alerts[reference_mask].mean()
            ),
            "test_motion_alert_rate": float(
                test_alerts[motion_mask].mean()
            ),
            "test_roc_auc": auc_score(
                labels["test"], test_scores
            ),
            "test_reference_episodes": int(reference_mask.sum()),
            "test_motion_episodes": int(motion_mask.sum()),
            "ideal_rate_ratio_to_no_surface": float(
                np.mean(communication_ratios[key])
            ),
        })

    return results


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Results already exist: {OUTPUT}\n"
            "Change OUTPUT to a new folder to preserve earlier results."
        )

    rows = []

    for seed in SEEDS:
        print(f"Simulating environment seed {seed}...", flush=True)
        rows.extend(simulate_seed(seed))

    if not all(
        np.isfinite(value)
        for row in rows
        for value in row.values()
        if isinstance(value, (float, int))
    ):
        raise ValueError("Simulation produced nonfinite results.")

    report = {
        "scope": "SYNTHETIC SIMULATION ONLY",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "configuration": {
            "seeds": SEEDS,
            "surface_strengths": STRENGTHS,
            "packets_per_episode": PACKETS,
            "channel_bins": CHANNELS,
            "surface_elements": ELEMENTS,
            "split_sizes": SPLIT_SIZES,
            "attacker": "Train-scaled 5-nearest-neighbor detector",
            "calibration_target":
                TARGET_REFERENCE_ALERT_RATE,
        },
        "results": rows,
        "limitations": [
            "Not calibrated to measured IRShield channels or a home.",
            "Channel bins use simplified random complex coefficients.",
            "Ideal binary reflection states omit hardware imperfections.",
            "No physical switching rate or packet timing is established.",
            "One fixed environment per seed; no cross-room evaluation.",
            "The detector is retrained per policy but is only one attack model.",
            "Discrete nearest-neighbor scores can cause conservative thresholds.",
            "Low threshold detection alone does not establish privacy; inspect AUC.",
            "Ideal rate ratio assumes perfect channel knowledge.",
            "Rate ratio is not throughput, packet loss or connectivity.",
            "No controller is selected or optimized using these test results.",
            "This simulation does not demonstrate a working security device.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    with (OUTPUT / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT / "simulation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(
        "\nSeed  Policy        Strength  Ref alerts  "
        "Motion alerts  AUC    Ideal rate ratio"
    )
    for row in rows:
        print(
            f"{row['seed']:<6}"
            f"{row['policy']:<14}"
            f"{row['surface_strength']:>8.2f}"
            f"{row['test_reference_alert_rate']:>11.1%}"
            f"{row['test_motion_alert_rate']:>15.1%}"
            f"{row['test_roc_auc']:>7.3f}"
            f"{row['ideal_rate_ratio_to_no_surface']:>17.3f}"
        )

    print(f"\nSaved simulated results to {OUTPUT}")


if __name__ == "__main__":
    main()