"""Recompute timing summaries from all five recorded repetitions, not new timing."""

from collections import defaultdict
import csv
import json
import math
import numpy as np


def summarize(root, output):
    reference = json.loads((root / "reference/timing_summaries.json").read_text())
    result = {}
    for method, expected in reference.items():
        with (root / "data/timing" / f"{method}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        groups = defaultdict(list)
        repeat_times = defaultdict(list)
        seen = set()
        for row in rows:
            key = (int(row["t"]), int(row["idx"]))
            identity = (key, int(row["repeat"]))
            if identity in seen or row["method"] != method:
                raise ValueError("Duplicate timing row or wrong method")
            seen.add(identity)
            values = [float(row[name]) for name in ("latency", "queries", "gradient_passes")]
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("Invalid timing measurement")
            groups[key].append(values)
            repeat_times[int(row["repeat"])].append(values[0])
        if len(groups) != 636 or any(len(values) != 5 for values in groups.values()):
            raise ValueError("Expected 636 requests with five timing repetitions")
        medians = np.array([np.median(groups[key], axis=0) for key in sorted(groups)])
        repeat_means = [float(np.mean(repeat_times[i])) for i in range(5)]
        measured = dict(n_prompts=len(groups), mean_prompt_median_latency=float(np.mean(medians[:, 0])),
                        p50_prompt_median_latency=float(np.quantile(medians[:, 0], .5)),
                        p95_prompt_median_latency=float(np.quantile(medians[:, 0], .95)),
                        p99_prompt_median_latency=float(np.quantile(medians[:, 0], .99)),
                        mean_prompt_median_queries=float(np.mean(medians[:, 1])),
                        mean_prompt_median_gradient_passes=float(np.mean(medians[:, 2])),
                        repeat_mean_latencies=repeat_means,
                        repeat_mean_range=max(repeat_means) - min(repeat_means),
                        repeat_mean_relative_range=(max(repeat_means) - min(repeat_means)) / float(np.mean(repeat_means)))
        for name, value in measured.items():
            if not np.allclose(value, expected[name], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Timing summary mismatch: {method}/{name}")
        result[method] = measured
    (output / "timing_summaries.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"methods": len(result), "measurements": 5 * 636 * 5,
            "scope": "Statistics of archived synchronized timing runs; not a new hardware benchmark."}
