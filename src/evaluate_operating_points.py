"""Recompute paper operating points and clustered intervals from numeric records."""
import argparse
import math
import os
from pathlib import Path

# Bound shared-server CPU use before importing the numerical runtime.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from fit_thresholds import (
    LENGTHS, SEEDS, SIZES, canonical_method, config_id,
    fit_document, load_rows, prepare_output,
    read_json, require, validate_grid, write_csv, write_json,
)


REPLICATES = 10_000
BOOTSTRAP_SEED = 42
RANDOM_MEAN = ("random_b384_mean", "mean")
KNOWN_ANCHORS = {
    ("sharegpt500", "base"): 59,
    ("sharegpt500", "pairspan_b384"): 190,
    ("sharegpt500", "greedy_20"): 284,
    ("harmbench2385", "base"): 0,
    ("harmbench2385", "pairspan_b384"): 1909,
    ("harmbench2385", "geometric_b384"): 1542,
    ("harmbench2385", "mce_b384"): 1601,
    ("harmbench2385", "greedy_20"): 1278,
}


def load_thresholds(path, calibration_rows):
    document = read_json(path)
    require(document.get("schema_version") == 2, "unsupported threshold schema")
    expected, _ = fit_document(calibration_rows, document["alphas"])
    for key in ("comparison", "threshold_grid", "selection", "budget_rule", "entries",
                "configurations"):
        require(document.get(key) == expected[key], f"thresholds differ from calibration: {key}")
    for entry in document["entries"]:
        require(float.fromhex(entry["theta_hex"]) == entry["theta"], "theta hex/decimal mismatch")
    return document


def validate_anchors(path, cells, keys):
    anchors = read_json(path)
    require(isinstance(anchors.get("counts"), list), "anchor counts must be a list")
    expected = {(d, k) for d in ("sharegpt500", "harmbench2385") for k in keys}
    seen = set()
    for anchor in anchors["counts"]:
        dataset = anchor["dataset_id"]
        key = (canonical_method(anchor["method_id"], anchor["seed"]), anchor["seed"])
        cell_key = (dataset, key)
        require(cell_key in expected and cell_key not in seen, "unknown/duplicate default anchor")
        seen.add(cell_key)
        n = 2380 if dataset == "harmbench2385" else 500
        require(type(anchor["n"]) is int and anchor["n"] == n, "wrong anchor denominator")
        count = sum(r["default_harmful"] for r in cells[cell_key]
                    if dataset != "harmbench2385" or not r["base_harmful"])
        require(type(anchor["positive_k"]) is int and anchor["positive_k"] == count,
                f"default anchor mismatch: {dataset}/{config_id(key)}")
        known = KNOWN_ANCHORS.get((dataset, key[0]))
        require(known is None or count == known, f"spec default anchor mismatch: {dataset}/{key[0]}")
    require(seen == expected, "missing default anchors (including every random seed)")


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def cluster_weights(rng, n):
    """A shared index draw represented as multiplicities, never variant draws."""
    indices = rng.integers(0, n, size=(REPLICATES, n))
    weights = np.zeros((REPLICATES, n), dtype=np.float64)
    np.add.at(weights, (np.arange(REPLICATES)[:, None], indices), 1)
    return weights


def interval(samples):
    return tuple(float(v) for v in np.quantile(samples, [0.025, 0.975], method="linear"))


def make_predictions(cells, keys, document):
    points = ["default", *document["alphas"]]
    entries = {(e["alpha"], (e["method_id"], e["seed"])): e for e in document["entries"]}
    predictions = {}
    for dataset in SIZES:
        for key in keys:
            rows = cells[(dataset, key)]
            default = np.array([r["default_harmful"] for r in rows], dtype=bool)
            score = np.array([r["ec_score"] for r in rows], dtype=float)
            previous = None
            ordered = sorted(points, key=lambda p: 0 if p == "default" else entries[(p, key)]["theta"])
            for point in ordered:
                theta = 0.0 if point == "default" else float.fromhex(entries[(point, key)]["theta_hex"])
                pred = default & (score >= theta)
                require(not np.any(pred & ~default), "postprocessing created a positive")
                if point == "default":
                    require(np.array_equal(pred, default), "default endpoint does not reproduce decisions")
                if previous is not None:
                    require(not np.any(pred & ~previous), "positive count increases with threshold")
                previous = pred
                predictions[(dataset, point, key)] = pred.astype(float)
        for point in points:
            predictions[(dataset, point, RANDOM_MEAN)] = np.mean(
                [predictions[(dataset, point, ("random_b384", seed))] for seed in SEEDS], axis=0)
    return points, entries, predictions


def bootstrap_rates(cells, keys, points, predictions):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    attack_rows = cells[("harmbench2385", keys[0])]
    groups = sorted({r["base_group_id"] for r in attack_rows})
    group_index = {g: i for i, g in enumerate(groups)}
    row_group = np.array([group_index[r["base_group_id"]] for r in attack_rows])
    bypass = np.array([not r["base_harmful"] for r in attack_rows])
    group_denominators = np.bincount(row_group, weights=bypass, minlength=159)
    group_weights = cluster_weights(rng, 159)
    sampled_denominators = group_weights @ group_denominators
    require(np.all(sampled_denominators > 0), "empty bootstrap denominator")
    samples = {}
    columns = [(point, key) for point in points for key in [*keys, RANDOM_MEAN]]
    numerators = np.column_stack([
        np.bincount(row_group, weights=predictions[("harmbench2385", point, key)] * bypass,
                    minlength=159) for point, key in columns])
    rates = (group_weights @ numerators) / sampled_denominators[:, None]
    for i, (point, key) in enumerate(columns):
        samples[("harmbench2385", point, key)] = rates[:, i]
    for dataset in ("calibration200", "sharegpt500"):
        weights = cluster_weights(rng, SIZES[dataset])
        matrix = np.column_stack([predictions[(dataset, point, key)] for point, key in columns])
        rates = (weights @ matrix) / SIZES[dataset]
        for i, (point, key) in enumerate(columns):
            samples[(dataset, point, key)] = rates[:, i]
    return samples


def cost_summary(rows_by_seed, field):
    values = np.mean([[r[field] for r in rows] for rows in rows_by_seed], axis=0)
    return {f"{field}_{name}": float(value) for name, value in (
        ("mean", np.mean(values)), ("median", np.median(values)),
        ("p95", np.quantile(values, .95)), ("max", np.max(values)))}


def build_outputs(cells, keys, document):
    points, entries, predictions = make_predictions(cells, keys, document)
    samples = bootstrap_rates(cells, keys, points, predictions)
    operating, lengths, filtering = [], [], []
    all_keys = [*keys, RANDOM_MEAN]
    for dataset in SIZES:
        reference = cells[(dataset, keys[0])]
        base = np.array([r["base_harmful"] for r in reference], dtype=bool)
        mask = ~base if dataset == "harmbench2385" else np.ones(len(base), dtype=bool)
        for point in points:
            for key in all_keys:
                pred = predictions[(dataset, point, key)]
                k, n = float(np.sum(pred[mask])), int(np.sum(mask))
                random_mean = key == RANDOM_MEAN
                ci = interval(samples[(dataset, point, key)]) if dataset == "harmbench2385" or random_mean else wilson(k, n)
                seed_rates = [float(np.sum(predictions[(dataset, point, ("random_b384", s))][mask])) / n
                              for s in SEEDS] if random_mean else [k / n]
                entry = entries.get((point, key))
                identity = dict(dataset_id=dataset, method_id=key[0], seed=key[1], point=point)
                base_k = int(np.sum(base))
                retained = float(np.sum(pred[base]))
                filtering.append(dict(**identity, base_positive_n=base_k, retained_k=retained,
                                      filtered_k=base_k - retained,
                                      filtered_fraction=(base_k - retained) / base_k if base_k else ""))
                costs = {}
                rows_by_seed = [cells[(dataset, ("random_b384", s))] for s in SEEDS] if random_mean else [cells[(dataset, key)]]
                for field in ("classifier_checks", "gradient_passes"):
                    costs.update(cost_summary(rows_by_seed, field))
                operating.append(dict(
                    **identity, theta_hex=entry["theta_hex"] if entry else "" if random_mean else (0.0).hex(),
                    theta=entry["theta"] if entry else "" if random_mean else 0.0,
                    status=entry["status"] if entry else "seed_mean" if random_mean else "default",
                    positive_k=k, n=n, rate=k / n, ci_low=ci[0], ci_high=ci[1],
                    ci_method="base_group_percentile" if dataset == "harmbench2385" else
                    "prompt_percentile" if random_mean else "Wilson",
                    seed_min=min(seed_rates), seed_max=max(seed_rates),
                    base_positive_n=base_k, base_filtered_k=base_k - retained,
                    nonbypass_released_k=base_k - retained if dataset == "harmbench2385" else "",
                    postprocess_classifier_checks=0, postprocess_gradient_passes=0, **costs))
                if dataset == "harmbench2385":
                    for length in LENGTHS:
                        length_mask = mask & np.array([r["insertion_t"] == length for r in reference])
                        length_k, length_n = float(np.sum(pred[length_mask])), int(np.sum(length_mask))
                        rates = [float(np.sum(predictions[(dataset, point, ("random_b384", s))][length_mask])) / length_n
                                 for s in SEEDS] if random_mean else [length_k / length_n]
                        lengths.append(dict(**identity, insertion_t=length, positive_k=length_k,
                                            n=length_n, rate=length_k / length_n,
                                            seed_min=min(rates), seed_max=max(rates)))
    deltas = []
    pairspan = next(k for k in keys if k[0] == "pairspan_b384")
    comparators = [k for k in keys if k[0] in {"geometric_b384", "mce_b384", "randec_0p10", "greedy_20"}]
    comparators.append(RANDOM_MEAN)
    for point in points:
        for comparator in comparators:
            result = dict(point=point, left_method_id=pairspan[0], left_seed=pairspan[1],
                          right_method_id=comparator[0], right_seed=comparator[1])
            for dataset, prefix in (("harmbench2385", "recovery"), ("sharegpt500", "test_fp")):
                reference = cells[(dataset, keys[0])]
                mask = np.array([not r["base_harmful"] for r in reference]) if dataset == "harmbench2385" else np.ones(len(reference), dtype=bool)
                left = predictions[(dataset, point, pairspan)]
                right = predictions[(dataset, point, comparator)]
                left_rate, right_rate = float(np.mean(left[mask])), float(np.mean(right[mask]))
                low, high = interval(samples[(dataset, point, pairspan)] - samples[(dataset, point, comparator)])
                result.update({f"{prefix}_left_rate": left_rate, f"{prefix}_right_rate": right_rate,
                               f"{prefix}_delta": left_rate - right_rate,
                               f"{prefix}_ci_low": low, f"{prefix}_ci_high": high,
                               f"{prefix}_n": int(np.sum(mask))})
            deltas.append(result)
    return operating, lengths, filtering, deltas


def run_evaluation(thresholds, normalized, anchors, output_dir):
    rows = load_rows(normalized, set(SIZES))
    cells, keys = validate_grid(rows, set(SIZES))
    require(len(rows) == 33_935, "expected exactly 33,935 normalized rows")
    calibration = [r for r in rows if r["dataset_id"] == "calibration200"]
    document = load_thresholds(thresholds, calibration)
    validate_anchors(anchors, cells, keys)
    results = build_outputs(cells, keys, document)
    output = prepare_output(output_dir)
    for name, data in zip(("operating_points.csv", "per_length_calibrated.csv",
                           "base_positive_filtering.csv", "paired_cluster_deltas.csv"), results):
        write_csv(output / name, data)
    write_json(output / "run_settings.json",
                    dict(record_count=len(rows), bootstrap_replicates=REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
                         rng="numpy.default_rng/PCG64", numpy_version=np.__version__,
                         interval="percentile/linear, conditional on frozen calibration",
                         group_order="lexicographic canonical base_group_id",
                         draw_order=["159 base groups", "200 calibration prompts", "500 test prompts"],
                         random_aggregate="within-prompt five-seed mean; not 2500 independent prompts",
                         cost="original execution costs; postprocessing saves no queries",
                         known_default_anchors="spec integers plus all 22 external anchors"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--normalized", required=True, type=Path)
    parser.add_argument("--default-anchors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(run_evaluation(args.thresholds, args.normalized, args.default_anchors, args.output_dir))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
