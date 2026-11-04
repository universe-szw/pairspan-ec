"""Paper algorithms extracted without changing function bodies.

Only the dependencies of the published EC policies are included. Candidate
evaluation retains the paper's stage/prefix schedule, not a first-hit rewrite.
"""
import hashlib
import itertools
import random
import re
from itertools import combinations
import numpy as np
import torch
from saliency import get_risk_heatmap


def erase_and_check_trace(
    prompt,
    pipeline,
    tokenizer,
    max_erase=20,
    num_adv=2,
    randomized=False,
    prompt_sampling_ratio=0.1,
    mode="suffix",
    llm_name='distilbert',
    max_llm_sequence_len=512,
    candidate_budget_cap=0,
    chunk=96,
):
    """Score-reporting EC trace used for matched-budget analysis.

    This helper mirrors EC candidate construction closely enough for analysis
    while retaining the original binary EC functions unchanged. For insertion,
    candidate_budget_cap should be set for capped-MCE sweeps; an uncapped full
    insertion exhaustive trace can be extremely expensive.
    """
    trace = _empty_trace("RandEC" if randomized else "MCE")
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_length = len(prompt_tokens)
    if prompt_length == 0:
        return _finalize_trace(trace, False)

    candidates = [prompt]
    seen = {prompt}

    def add_candidate(tokens):
        if tokens is None:
            return False
        new_prompt = tokenizer.decode(tokens)
        if new_prompt in seen:
            return False
        seen.add(new_prompt)
        candidates.append(new_prompt)
        return True

    if mode == "suffix":
        for i in range(min(max_erase, prompt_length)):
            add_candidate(prompt_tokens[:-(i + 1)])
    elif mode == "infusion":
        for i in range(min(max_erase, prompt_length)):
            for location in combinations(range(prompt_length), i + 1):
                add_candidate(delete_by_indices(prompt_tokens, location))
                if candidate_budget_cap and len(candidates) >= candidate_budget_cap:
                    break
            if candidate_budget_cap and len(candidates) >= candidate_budget_cap:
                break
    elif mode == "insertion":
        def decode_combination(combination):
            erase_locations = []
            start = 0
            end = 0
            for i in range(len(combination) // 2):
                start = end + combination[(2 * i)]
                end = start + combination[(2 * i) + 1]
                if start >= prompt_length or end > prompt_length:
                    return None
                erase_locations.extend(range(start, end))
            if not erase_locations:
                return None
            return delete_by_indices(prompt_tokens, erase_locations)

        if randomized:
            sample_budget = int(prompt_sampling_ratio * prompt_length * (max_erase + 1) * max(1, num_adv))
            sample_budget = max(1, min(sample_budget, candidate_budget_cap or 2048))
            max_attempts = max(100, sample_budget * 20)
            attempts = 0
            while len(candidates) - 1 < sample_budget and attempts < max_attempts:
                attempts += 1
                combination = []
                for _ in range(num_adv):
                    combination.append(random.randrange(prompt_length))
                    combination.append(random.randrange(max_erase + 1))
                add_candidate(decode_combination(combination))
        else:
            args = []
            for _ in range(num_adv):
                args.append(range(prompt_length))
                args.append(range(max_erase + 1))
            for combination in itertools.product(*args):
                add_candidate(decode_combination(combination))
                if candidate_budget_cap and len(candidates) >= candidate_budget_cap:
                    break
    else:
        raise ValueError("Invalid mode: " + mode)

    if randomized and len(candidates) > 1 and mode in {"suffix", "infusion"}:
        num_to_test = int(prompt_sampling_ratio * (len(candidates) - 1))
        candidates = candidates[:1] + random.sample(candidates[1:], max(1, num_to_test))

    if candidate_budget_cap and len(candidates) > candidate_budget_cap:
        candidates = candidates[:candidate_budget_cap]

    for i in range(0, len(candidates), chunk):
        harmful, harm_scores = _classify_with_harm_scores(candidates[i:i + chunk], pipeline, chunk=chunk)
        _update_trace_scores(trace, harmful, harm_scores)
        if any(harmful):
            return _finalize_trace(trace, True)
    return _finalize_trace(trace, False)


def delete_by_indices(data_list, indices):
    data_list = data_list.copy()
    for index in sorted(indices, reverse=True):
        del data_list[index]
    return data_list


def _classify_with_harm_scores(prompts, pipeline, chunk=64):
    harmful = []
    harm_scores = []
    for i in range(0, len(prompts), chunk):
        outputs = pipeline(prompts[i:i + chunk])
        for out in outputs:
            is_bad = out["label"] == "LABEL_0"
            harmful.append(is_bad)
            harm_scores.append(float(out["score"] if is_bad else 1.0 - out["score"]))
    return harmful, harm_scores


def _empty_trace(candidate_policy, verbose=False):
    trace = {
        "harmful": False,
        "ec_score": 0.0,
        "max_harmful_prob": 0.0,
        "first_hit_margin": None,
        "hit_count": 0,
        "checked_candidate_count": 0,
        "classifier_query_count": 0,
        "candidate_policy": candidate_policy,
    }
    if verbose:
        trace["verbose_events"] = []
    return trace


def _hash_text(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _hash_token_ids(token_ids):
    return hashlib.sha256(",".join(str(int(x)) for x in token_ids).encode("utf-8")).hexdigest()[:16]


def _deleted_token_ids(prompt_tokens, spans):
    token_ids = []
    for start, end in spans or []:
        start = max(0, int(start))
        end = min(len(prompt_tokens), int(end))
        if end > start:
            token_ids.extend(prompt_tokens[start:end])
    return token_ids


def _marker_category(text):
    lower = (text or "").lower()
    if re.search(r"\b(no|not|never|without|avoid|prevent|stop|don't|do not|cannot|can't|shouldn't)\b", lower):
        return "negation"
    if re.search(r"\b(safe|safety|policy|allowed|legal|ethical|responsible|guardrail|moderation)\b", lower):
        return "safety_keyword"
    if re.search(r"\b(teach|learn|education|educational|explain|explanation|research|study|awareness)\b", lower):
        return "education_policy"
    if re.search(r"\b(story|fiction|novel|screenplay|roleplay|character|scene|plot)\b", lower):
        return "fictional_roleplay"
    if re.search(r"\b(write|rewrite|edit|grammar|tone|email|essay|summary|summarize|translate)\b", lower):
        return "writing_context"
    if lower.strip():
        return "generic"
    return "unknown"


def _event_from_candidate(stage, candidate_prompt, spans, score, threshold, prompt_tokens, tokenizer, event_type):
    spans = [(int(start), int(end)) for start, end in (spans or [])]
    deleted_ids = _deleted_token_ids(prompt_tokens, spans)
    deleted_text = tokenizer.decode(deleted_ids) if deleted_ids else ""
    deletion_ratio = len(set(i for start, end in spans for i in range(max(0, start), min(len(prompt_tokens), end)))) / max(1, len(prompt_tokens))
    return {
        "event_type": event_type,
        "stage": stage,
        "score": float(score),
        "margin": float(score) - float(threshold),
        "spans": spans,
        "span_count": len(spans),
        "deletion_token_count": len(deleted_ids),
        "deletion_ratio": float(deletion_ratio),
        "candidate_hash": _hash_text(candidate_prompt),
        "deleted_token_hash": _hash_token_ids(deleted_ids),
        "marker_category": _marker_category(deleted_text),
    }


def _append_limited_verbose_event(trace, event, per_stage_limit=64):
    if "verbose_events" not in trace:
        return
    counts = trace.setdefault("_verbose_stage_counts", {})
    key = (event["stage"], event["event_type"])
    count = counts.get(key, 0)
    if count >= per_stage_limit and event["event_type"] == "hit":
        return
    counts[key] = count + 1
    trace["verbose_events"].append(event)


def _record_verbose_batch(trace, stage, prompts, metas, harmful, scores, prompt_tokens, tokenizer, threshold):
    if "verbose_events" not in trace or not prompts:
        return
    metas = metas or [[] for _ in prompts]
    best_idx = int(np.argmax(scores)) if scores else -1
    if 0 <= best_idx < len(prompts):
        _append_limited_verbose_event(
            trace,
            _event_from_candidate(
                stage,
                prompts[best_idx],
                metas[best_idx],
                scores[best_idx],
                threshold,
                prompt_tokens,
                tokenizer,
                event_type="stage_best",
            ),
            per_stage_limit=8,
        )
    for prompt, spans, is_bad, score in zip(prompts, metas, harmful, scores):
        if bool(is_bad) and float(score) >= threshold:
            _append_limited_verbose_event(
                trace,
                _event_from_candidate(stage, prompt, spans, score, threshold, prompt_tokens, tokenizer, event_type="hit"),
            )


def _update_trace_scores(trace, harmful, harm_scores, threshold=0.5):
    for is_bad, score in zip(harmful, harm_scores):
        score = float(score)
        trace["checked_candidate_count"] += 1
        trace["classifier_query_count"] += 1
        if score > trace["ec_score"]:
            trace["ec_score"] = score
            trace["max_harmful_prob"] = score
        if is_bad or score >= threshold:
            trace["hit_count"] += 1
            if trace["first_hit_margin"] is None:
                trace["first_hit_margin"] = score - threshold


def _base_prompt_trace(prompt, pipeline, candidate_policy="base_prompt", threshold=0.5):
    trace = _empty_trace(candidate_policy)
    harmful, harm_scores = _classify_with_harm_scores([prompt], pipeline, chunk=1)
    _update_trace_scores(trace, harmful, harm_scores, threshold=threshold)
    trace["harmful"] = bool(harmful[0])
    trace["base_harmful"] = bool(harmful[0])
    return trace


def base_classifier_trace(prompt, pipeline, threshold=0.5):
    """Return an explicit pre-EC classifier trace."""
    return _base_prompt_trace(
        prompt,
        pipeline,
        candidate_policy="BaseClassifier",
        threshold=threshold,
    )


def _finalize_trace(trace, harmful):
    trace["harmful"] = bool(harmful)
    if trace["first_hit_margin"] is None:
        trace["first_hit_margin"] = trace["ec_score"] - 0.5
    for key in list(trace.keys()):
        if key.startswith("_"):
            trace.pop(key, None)
    return trace


def _top_risk_spans(risk_scores, max_erase, max_spans=48):
    n = len(risk_scores)
    if n == 0:
        return []
    lens = [1, 2, 4, 8, 16, 32, 64]
    lens = [x for x in lens if x <= max_erase and x <= n]
    if max_erase <= n and max_erase not in lens:
        lens.append(max_erase)
    spans = []
    prefix = np.concatenate([[0.0], np.cumsum(risk_scores)])
    for length in lens:
        if length <= 0 or length > n:
            continue
        scores = (prefix[length:] - prefix[:-length]) / np.sqrt(length)
        if len(scores) == 0:
            continue
        k = min(8, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        for start in top_idx:
            start = int(start)
            spans.append((start, start + length, float(scores[start]), length))
    spans.sort(key=lambda x: (x[2], x[3]), reverse=True)
    dedup = []
    seen = set()
    for start, end, score, length in spans:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((start, end, score))
        if len(dedup) >= max_spans:
            break
    return dedup


def _delete_spans_strict(prompt_tokens, spans):
    erase_locations = []
    for start, end in spans:
        start = max(0, int(start))
        end = min(len(prompt_tokens), int(end))
        if end <= start:
            continue
        erase_locations.extend(range(start, end))
    erase_locations = sorted(set(erase_locations))
    if not erase_locations or len(erase_locations) >= len(prompt_tokens):
        return None
    return delete_by_indices(prompt_tokens, erase_locations)


def _non_overlapping(span_a, span_b):
    return max(span_a[0], span_b[0]) >= min(span_a[1], span_b[1])


def _span_iou(span_a, span_b):
    inter = max(0, min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]))
    union = max(span_a[1], span_b[1]) - min(span_a[0], span_b[0])
    return inter / union if union > 0 else 0.0


def _margin_harmful(harmful, scores, base_harm_score, margin=0.03):
    return _margin_hit_count(
        harmful,
        scores,
        base_harm_score,
        margin=margin,
    ) > 0


def _margin_hit_count(harmful, scores, base_harm_score, margin=0.03):
    return sum(
        bool(is_bad)
        and (
            float(score) >= base_harm_score + margin
            or float(score) >= 0.55
        )
        for is_bad, score in zip(harmful, scores)
    )


def _geometric_span_lens(max_erase, prompt_length):
    raw = [1, 2, 4] + list(range(8, max_erase + 1, 4)) + [max_erase]
    return sorted({x for x in raw if 1 <= x <= max_erase and x < prompt_length})


def _grid_starts(prompt_length, length, max_starts=24):
    if length <= 0 or length >= prompt_length:
        return []
    stride = max(1, min(8, length // 4 if length >= 4 else 1))
    starts = list(range(0, prompt_length - length + 1, stride))
    starts.extend(
        [
            max(0, prompt_length // 4 - length // 2),
            max(0, prompt_length // 2 - length // 2),
            max(0, (3 * prompt_length) // 4 - length // 2),
            prompt_length - length,
        ]
    )
    starts = sorted(set(start for start in starts if 0 <= start <= prompt_length - length))
    if len(starts) <= max_starts:
        return starts
    sampled = [starts[int(round(i))] for i in np.linspace(0, len(starts) - 1, max_starts)]
    return sorted(set(sampled + [0, prompt_length - length]))


def _nms_spans(spans, max_keep=32, iou_threshold=0.75):
    kept = []
    for span in sorted(spans, key=lambda item: item[2], reverse=True):
        if all(_span_iou(span, other) <= iou_threshold for other in kept):
            kept.append(span)
        if len(kept) >= max_keep:
            break
    return kept


def _balanced_start_order(starts):
    """Order grid starts to cover the center, edges, and quartiles first."""
    starts = sorted(set(int(start) for start in starts))
    if len(starts) <= 2:
        return starts
    targets = [
        0.5 * (starts[0] + starts[-1]),
        starts[0],
        starts[-1],
        0.75 * starts[0] + 0.25 * starts[-1],
        0.25 * starts[0] + 0.75 * starts[-1],
    ]
    ordered = []
    remaining = set(starts)
    for target in targets:
        if not remaining:
            break
        selected = min(remaining, key=lambda value: (abs(value - target), value))
        ordered.append(selected)
        remaining.remove(selected)
    ordered.extend(sorted(remaining, key=lambda value: (abs(value - targets[0]), value)))
    return ordered


def _pairspan_single_spans(prompt_length, max_erase, saliency_candidates, limit):
    """Select a deterministic, length-balanced single-span scoring pool."""
    selected = []
    seen = set()

    def add(span):
        start, end = int(span[0]), int(span[1])
        key = (max(0, start), min(prompt_length, end))
        if key[1] <= key[0] or key in seen:
            return
        seen.add(key)
        selected.append(key)

    for span in saliency_candidates:
        if len(selected) >= limit:
            return selected
        add(span)

    buckets = []
    for length in sorted(
        _geometric_span_lens(max_erase=max_erase, prompt_length=prompt_length),
        reverse=True,
    ):
        starts = _balanced_start_order(_grid_starts(prompt_length, length))
        buckets.append([(start, start + length) for start in starts])

    offset = 0
    while len(selected) < limit and any(offset < len(bucket) for bucket in buckets):
        for bucket in buckets:
            if offset < len(bucket):
                add(bucket[offset])
                if len(selected) >= limit:
                    break
        offset += 1
    return selected


def _interleave_pair_families(adaptive, geometric, limit):
    """Alternate adaptive and geometric pairs while removing text duplicates."""
    prompts = []
    metadata = []
    seen = set()
    adaptive_idx = 0
    geometric_idx = 0
    turn = 0
    while len(prompts) < limit and (
        adaptive_idx < len(adaptive) or geometric_idx < len(geometric)
    ):
        family = adaptive if turn % 2 == 0 else geometric
        index = adaptive_idx if turn % 2 == 0 else geometric_idx
        if index >= len(family):
            family = geometric if turn % 2 == 0 else adaptive
            index = geometric_idx if turn % 2 == 0 else adaptive_idx
            if index >= len(family):
                break
            use_adaptive = family is adaptive
        else:
            use_adaptive = family is adaptive
        prompt, spans = family[index]
        if use_adaptive:
            adaptive_idx += 1
        else:
            geometric_idx += 1
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            metadata.append(spans)
        turn += 1
    return prompts, metadata


def pairspan_ec_trace(
    prompt,
    pipeline,
    tokenizer,
    model,
    max_erase=64,
    num_adv=2,
    num_iters=10,
    llm_name="distilbert",
    max_llm_sequence_len=512,
    total_query_cap=384,
    direct_cap=2800,
    single_keep=28,
    saliency_spans=16,
    coarse_keep=36,
    pair_cap=260,
    nms_iou=0.70,
    decision_margin=0.03,
    use_saliency=True,
    use_nms=True,
    use_adaptive_pairs=True,
    use_geometric_pairs=True,
    single_budget_fraction=0.25,
    single_budget_cap=192,
    required_confirmations=1,
    verbose_trace=False,
):
    """Finite-budget pair-span search used by the revised insertion method."""
    if total_query_cap <= 0:
        raise ValueError("PairSpan-EC requires a positive total_query_cap.")
    if not 0.0 <= single_budget_fraction <= 1.0:
        raise ValueError("single_budget_fraction must lie in [0, 1].")
    if single_budget_cap < 0:
        raise ValueError("single_budget_cap must be non-negative.")
    if required_confirmations < 1:
        raise ValueError("required_confirmations must be positive.")
    policy = (
        "PairSpanEC"
        if required_confirmations == 1
        else f"PairSpanEC_Confirm{required_confirmations}"
    )
    trace = _empty_trace(policy, verbose=verbose_trace)
    trace.update(
        {
            "stage_query_counts": {},
            "total_query_cap": int(total_query_cap),
            "greedy_fast_enabled": False,
            "local_refinement": False,
            "pair_rank_rule": "sum_single_span_harmful_probability",
            "pair_family_order": "adaptive_geometric_alternation",
            "gradient_pass_count": 0,
            "required_confirmations": int(required_confirmations),
            "margin_hit_count": 0,
            "component_flags": {
                "saliency": bool(use_saliency),
                "nms": bool(use_nms),
                "adaptive_pairs": bool(use_adaptive_pairs),
                "geometric_pairs": bool(use_geometric_pairs),
            },
        }
    )

    base_outputs = pipeline([prompt])[0]
    base_is_bad = base_outputs["label"] == "LABEL_0"
    base_harm_score = float(
        base_outputs["score"] if base_is_bad else 1.0 - base_outputs["score"]
    )
    trace["base_prob"] = base_harm_score
    trace["base_harmful"] = base_is_bad
    _update_trace_scores(trace, [base_is_bad], [base_harm_score])
    trace["stage_query_counts"]["base"] = 1
    if base_is_bad or total_query_cap == 1:
        return _finalize_trace(trace, base_is_bad)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_length = len(prompt_tokens)
    if prompt_length < 4 or num_adv < 2:
        return _finalize_trace(trace, False)

    saliency_candidates = []
    if use_saliency and saliency_spans > 0:
        try:
            risk_scores, _ = get_risk_heatmap(
                prompt,
                pipeline,
                tokenizer,
                max_length=max_llm_sequence_len,
            )
            trace["gradient_pass_count"] = 1
            saliency_candidates = [
                (start, end)
                for start, end, _ in _top_risk_spans(
                    risk_scores,
                    max_erase=max_erase,
                    max_spans=saliency_spans,
                )
                if end <= prompt_length
            ]
        except Exception as error:
            trace["saliency_error_type"] = type(error).__name__

    requested_single_budget = int((total_query_cap - 1) * single_budget_fraction)
    single_budget = min(
        total_query_cap - trace["classifier_query_count"],
        single_budget_cap,
        requested_single_budget,
    )
    single_spans = _pairspan_single_spans(
        prompt_length,
        max_erase,
        saliency_candidates,
        single_budget,
    )
    single_prompts = []
    single_meta = []
    seen_single = {prompt}
    for span in single_spans:
        erased = _delete_spans_strict(prompt_tokens, [span])
        if not erased:
            continue
        candidate = tokenizer.decode(erased)
        if candidate in seen_single:
            continue
        seen_single.add(candidate)
        single_prompts.append(candidate)
        single_meta.append(span)

    decision_threshold = min(base_harm_score + decision_margin, 0.55)
    if single_prompts:
        harmful, harm_scores = _classify_with_harm_scores(
            single_prompts,
            pipeline,
            chunk=96,
        )
        _update_trace_scores(trace, harmful, harm_scores)
        trace["stage_query_counts"]["single_span"] = len(single_prompts)
        _record_verbose_batch(
            trace,
            "single_span",
            single_prompts,
            [[span] for span in single_meta],
            harmful,
            harm_scores,
            prompt_tokens,
            tokenizer,
            threshold=decision_threshold,
        )
        single_hits = _margin_hit_count(
            harmful,
            harm_scores,
            base_harm_score,
            margin=decision_margin,
        )
        trace["margin_hit_count"] += single_hits
        if trace["margin_hit_count"] >= required_confirmations:
            return _finalize_trace(trace, True)
    else:
        harm_scores = []

    scored_spans = [
        (start, end, float(score), end - start)
        for (start, end), score in zip(single_meta, harm_scores)
    ]
    if use_nms:
        coarse = _nms_spans(
            scored_spans,
            max_keep=min(single_keep, coarse_keep),
            iou_threshold=nms_iou,
        )
    else:
        coarse = sorted(scored_spans, key=lambda item: item[2], reverse=True)[
            : min(single_keep, coarse_keep)
        ]
    adaptive_pairs = []
    if use_adaptive_pairs:
        for i, left in enumerate(coarse):
            for right in coarse[i + 1 :]:
                if not _non_overlapping(left, right):
                    continue
                if (left[1] - left[0]) + (right[1] - right[0]) > max_erase * num_adv:
                    continue
                pair = tuple(sorted((left, right), key=lambda item: item[0]))
                score = float(pair[0][2] + pair[1][2])
                adaptive_pairs.append((pair, score))
    adaptive_pairs.sort(
        key=lambda item: (
            -item[1],
            item[0][0][0],
            item[0][0][1],
            item[0][1][0],
            item[0][1][1],
        )
    )

    adaptive_family = []
    adaptive_seen = set(seen_single)
    for pair, _ in adaptive_pairs[:pair_cap]:
        spans = [(pair[0][0], pair[0][1]), (pair[1][0], pair[1][1])]
        erased = _delete_spans_strict(prompt_tokens, spans)
        if not erased:
            continue
        candidate = tokenizer.decode(erased)
        if candidate in adaptive_seen:
            continue
        adaptive_seen.add(candidate)
        adaptive_family.append((candidate, spans))

    if use_geometric_pairs:
        geometric_prompts, geometric_meta = _direct_pair_grid(
            prompt_tokens,
            tokenizer,
            max_erase=max_erase,
            num_adv=num_adv,
        )
        geometric_family = list(zip(geometric_prompts[:direct_cap], geometric_meta[:direct_cap]))
    else:
        geometric_family = []
    remaining = total_query_cap - trace["classifier_query_count"]
    pair_prompts, pair_meta = _interleave_pair_families(
        adaptive_family,
        geometric_family,
        max(0, remaining),
    )
    trace["candidate_pool_sizes"] = {
        "saliency_spans": len(saliency_candidates),
        "single_scored": len(single_prompts),
        "nms_spans": len(coarse),
        "adaptive_pairs": len(adaptive_family),
        "geometric_pairs": len(geometric_family),
        "pair_evaluated": len(pair_prompts),
    }
    trace["single_query_budget"] = int(single_budget)
    trace["single_budget_fraction"] = float(single_budget_fraction)
    trace["single_budget_cap"] = int(single_budget_cap)

    if pair_prompts:
        harmful, harm_scores = _classify_with_harm_scores(
            pair_prompts,
            pipeline,
            chunk=96,
        )
        _update_trace_scores(trace, harmful, harm_scores)
        trace["stage_query_counts"]["pair_span"] = len(pair_prompts)
        _record_verbose_batch(
            trace,
            "pair_span",
            pair_prompts,
            pair_meta,
            harmful,
            harm_scores,
            prompt_tokens,
            tokenizer,
            threshold=decision_threshold,
        )
        pair_hits = _margin_hit_count(
            harmful,
            harm_scores,
            base_harm_score,
            margin=decision_margin,
        )
        trace["margin_hit_count"] += pair_hits
        result = trace["margin_hit_count"] >= required_confirmations
    else:
        result = False
    if trace["classifier_query_count"] > total_query_cap:
        raise AssertionError("PairSpan-EC exceeded its total classifier-evaluation cap")
    return _finalize_trace(trace, result)


def pairspan_ec(prompt, pipeline, tokenizer, model, **kwargs):
    """Return the PairSpan-EC decision from its canonical trace path."""
    return bool(
        pairspan_ec_trace(
            prompt,
            pipeline,
            tokenizer,
            model,
            **kwargs,
        )["harmful"]
    )


def _direct_pair_grid(prompt_tokens, tokenizer, max_erase=64, num_adv=2):
    """Build the direct geometric two-span family in the submitted order."""
    prompt_length = len(prompt_tokens)
    lengths = _geometric_span_lens(max_erase=max_erase, prompt_length=prompt_length)
    large_first = sorted([x for x in lengths if x >= 8], reverse=True)
    length_pairs = []
    for span_len in large_first:
        if span_len * 2 <= max_erase * num_adv:
            length_pairs.append((span_len, span_len))
    for i, left_len in enumerate(large_first):
        for right_len in large_first[i + 1:]:
            if left_len + right_len <= max_erase * num_adv:
                length_pairs.append((left_len, right_len))

    prompts = []
    metas = []
    seen = set()
    for left_len, right_len in length_pairs:
        for left_start in _grid_starts(prompt_length, left_len):
            left = (left_start, left_start + left_len)
            for right_start in _grid_starts(prompt_length, right_len):
                right = (right_start, right_start + right_len)
                if max(left[0], right[0]) < min(left[1], right[1]):
                    continue
                spans = sorted([left, right], key=lambda item: item[0])
                erased = _delete_spans_strict(prompt_tokens, spans)
                if not erased:
                    continue
                candidate = tokenizer.decode(erased)
                if candidate in seen:
                    continue
                seen.add(candidate)
                prompts.append(candidate)
                metas.append(spans)
    return prompts, metas


def random_pair_grid_ec_trace(
    prompt,
    pipeline,
    tokenizer,
    max_erase=64,
    num_adv=2,
    candidate_budget=384,
    decision_margin=0.03,
    seed=42,
    candidate_policy="RandomPairGrid",
    verbose_trace=False,
):
    """Fixed-query random-order control over the direct geometric pair grid."""
    trace = _empty_trace(candidate_policy, verbose=verbose_trace)
    trace["stage_query_counts"] = {}
    base_outputs = pipeline([prompt])[0]
    base_is_bad = base_outputs["label"] == "LABEL_0"
    base_harm_score = float(base_outputs["score"] if base_is_bad else 1.0 - base_outputs["score"])
    trace["base_prob"] = base_harm_score
    trace["base_harmful"] = base_is_bad
    _update_trace_scores(trace, [base_is_bad], [base_harm_score])
    trace["stage_query_counts"]["base"] = 1
    if base_is_bad or candidate_budget <= 1:
        return _finalize_trace(trace, base_is_bad)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) < 4:
        return _finalize_trace(trace, False)
    pair_prompts, pair_meta = _direct_pair_grid(
        prompt_tokens,
        tokenizer,
        max_erase=max_erase,
        num_adv=num_adv,
    )
    order = list(range(len(pair_prompts)))
    stable_seed = int(hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
    random.Random(seed ^ stable_seed).shuffle(order)
    order = order[:max(0, candidate_budget - 1)]
    selected_prompts = [pair_prompts[i] for i in order]
    selected_meta = [pair_meta[i] for i in order]
    if not selected_prompts:
        return _finalize_trace(trace, False)

    harmful, harm_scores = _classify_with_harm_scores(selected_prompts, pipeline, chunk=96)
    _update_trace_scores(trace, harmful, harm_scores)
    trace["stage_query_counts"]["random_pair_grid"] = len(selected_prompts)
    decision_threshold = min(base_harm_score + decision_margin, 0.55)
    _record_verbose_batch(
        trace,
        "random_pair_grid",
        selected_prompts,
        selected_meta,
        harmful,
        harm_scores,
        prompt_tokens,
        tokenizer,
        threshold=decision_threshold,
    )
    return _finalize_trace(
        trace,
        _margin_harmful(harmful, harm_scores, base_harm_score, margin=decision_margin),
    )


def geometric_prefix_ec_trace(
    prompt,
    pipeline,
    tokenizer,
    max_erase=64,
    num_adv=2,
    candidate_budget=384,
    decision_margin=0.03,
    candidate_policy="GeometricPrefix",
    verbose_trace=False,
):
    """Evaluate a fixed prefix of the direct geometric pair family."""
    trace = _empty_trace(candidate_policy, verbose=verbose_trace)
    trace["stage_query_counts"] = {}
    base_outputs = pipeline([prompt])[0]
    base_is_bad = base_outputs["label"] == "LABEL_0"
    base_harm_score = float(
        base_outputs["score"] if base_is_bad else 1.0 - base_outputs["score"]
    )
    trace["base_prob"] = base_harm_score
    trace["base_harmful"] = base_is_bad
    _update_trace_scores(trace, [base_is_bad], [base_harm_score])
    trace["stage_query_counts"]["base"] = 1
    if base_is_bad or candidate_budget <= 1:
        return _finalize_trace(trace, base_is_bad)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) < 4:
        return _finalize_trace(trace, False)
    pair_prompts, pair_meta = _direct_pair_grid(
        prompt_tokens,
        tokenizer,
        max_erase=max_erase,
        num_adv=num_adv,
    )
    keep = max(0, candidate_budget - 1)
    pair_prompts = pair_prompts[:keep]
    pair_meta = pair_meta[:keep]
    if not pair_prompts:
        return _finalize_trace(trace, False)

    harmful, harm_scores = _classify_with_harm_scores(
        pair_prompts,
        pipeline,
        chunk=96,
    )
    _update_trace_scores(trace, harmful, harm_scores)
    trace["stage_query_counts"]["geometric_prefix"] = len(pair_prompts)
    decision_threshold = min(base_harm_score + decision_margin, 0.55)
    _record_verbose_batch(
        trace,
        "geometric_prefix",
        pair_prompts,
        pair_meta,
        harmful,
        harm_scores,
        prompt_tokens,
        tokenizer,
        threshold=decision_threshold,
    )
    return _finalize_trace(
        trace,
        _margin_harmful(
            harmful,
            harm_scores,
            base_harm_score,
            margin=decision_margin,
        ),
    )


def greedy_ec_trace(prompt, model, tokenizer, num_iters=20, threshold=0.5, candidate_policy="GreedyEC"):
    """GreedyEC trace with max harmful probability over checked candidates."""
    trace = _empty_trace(candidate_policy)
    device = model.device
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    with torch.inference_mode():
        output_softmax = torch.nn.functional.softmax(model(tokens)[0], dim=1)
        score = float(output_softmax[0, 0].item())
        is_bad = bool(output_softmax[0].argmax().item() == 0)
    _update_trace_scores(trace, [is_bad], [score], threshold=threshold)
    if is_bad and score >= threshold:
        return _finalize_trace(trace, True)

    for _ in range(num_iters):
        if tokens.shape[1] < 3:
            return _finalize_trace(trace, False)
        tokens_batch = []
        for i in range(1, tokens.shape[1] - 1):
            erased_seq = torch.cat((tokens[:, :i], tokens[:, i + 1:]), dim=1)
            tokens_batch.append(erased_seq)
        tokens_batch = torch.cat(tokens_batch, dim=0)
        sub_batch = 256
        max_score = -1.0
        best_idx = -1
        hit = False
        with torch.inference_mode():
            for j in range(0, tokens_batch.shape[0], sub_batch):
                batch = tokens_batch[j:j + sub_batch]
                output_softmax = torch.nn.functional.softmax(model(batch)[0], dim=1)
                scores = output_softmax[:, 0].detach().cpu().numpy().astype(float).tolist()
                harmful = (output_softmax.argmax(dim=1) == 0).detach().cpu().numpy().astype(bool).tolist()
                _update_trace_scores(trace, harmful, scores, threshold=threshold)
                curr_max_idx = int(np.argmax(scores))
                curr_max_score = float(scores[curr_max_idx])
                if curr_max_score > max_score:
                    max_score = curr_max_score
                    best_idx = j + curr_max_idx
                if any(score >= threshold for score in scores):
                    hit = True
        if hit or max_score >= threshold:
            return _finalize_trace(trace, True)
        tokens = tokens_batch[best_idx].unsqueeze(0)
    return _finalize_trace(trace, False)


def grad_ec_trace(prompt, model, tokenizer, word_embeddings=None, num_iters=50, threshold=0.5):
    """GradEC trace over the same discrete checks used by GradEC.

    The continuous relaxation drives the mask update, but the reported score is
    the maximum harmful probability among actually re-encoded prompts checked by
    the binary GradEC decision path.
    """
    trace = _empty_trace("GradEC")
    device = getattr(model, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if word_embeddings is None:
        word_embeddings = model.get_input_embeddings()
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.inference_mode():
        model_output = model(tokens)
        logits = model_output.logits if hasattr(model_output, "logits") else model_output[0]
        scores = torch.nn.functional.softmax(logits, dim=1)
        harm_score = float(scores[0, 0].item())
        is_bad = bool(scores[0].argmax().item() == 0)
    _update_trace_scores(trace, [is_bad], [harm_score], threshold=threshold)
    if is_bad and harm_score >= threshold:
        return _finalize_trace(trace, True)

    mask_logits = torch.ones_like(tokens).float()
    mask_logits = torch.nn.Parameter(mask_logits.requires_grad_().to(device))
    optimizer = torch.optim.SGD(params=[mask_logits], lr=10.0, momentum=0.0, weight_decay=1e-2)

    for _ in range(num_iters):
        optimizer.zero_grad()
        mask_sigmoid = torch.sigmoid(mask_logits)
        binary_mask = (mask_sigmoid >= 0.5).long()
        masked_tokens = binary_mask * tokens
        masked_prompt = tokenizer.decode((masked_tokens)[0][1:-1])
        reencoded = torch.tensor(tokenizer.encode(masked_prompt)).unsqueeze(0).to(device)
        with torch.inference_mode():
            model_output = model(reencoded)
            logits = model_output.logits if hasattr(model_output, "logits") else model_output[0]
            scores = torch.nn.functional.softmax(logits, dim=1)
            harm_score = float(scores[0, 0].item())
            is_bad = bool(scores[0].argmax().item() == 0)
        _update_trace_scores(trace, [is_bad], [harm_score], threshold=threshold)
        if is_bad and harm_score >= threshold:
            return _finalize_trace(trace, True)

        embeddings = word_embeddings(tokens)
        embeddings = mask_sigmoid.unsqueeze(2) * embeddings + (1 - mask_sigmoid.unsqueeze(2)) * word_embeddings(0 * tokens)
        output = model(inputs_embeds=embeddings, labels=torch.tensor([0]).to(device))
        loss = output.loss
        loss.backward()
        optimizer.step()

    mask_sigmoid = torch.sigmoid(mask_logits)
    binary_mask = (mask_sigmoid >= 0.5).long()
    masked_tokens = binary_mask * tokens
    masked_prompt = tokenizer.decode((masked_tokens)[0][1:-1])
    reencoded = torch.tensor(tokenizer.encode(masked_prompt)).unsqueeze(0).to(device)
    with torch.inference_mode():
        model_output = model(reencoded)
        logits = model_output.logits if hasattr(model_output, "logits") else model_output[0]
        scores = torch.nn.functional.softmax(logits, dim=1)
        harm_score = float(scores[0, 0].item())
        is_bad = bool(scores[0].argmax().item() == 0)
    _update_trace_scores(trace, [is_bad], [harm_score], threshold=threshold)
    return _finalize_trace(trace, is_bad and harm_score >= threshold)
