import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_metrics(
    predicted_matches: List[Dict],
    ground_truth: Dict[str, str]
) -> Dict[str, float]:
    
    correct = 0
    total_predicted = len(predicted_matches)
    total_actual = len(ground_truth)

    for match in predicted_matches:
        bank_id = match['bank_id']
        check_id = match['check_id']
        if ground_truth.get(bank_id) == check_id:
            correct += 1

    precision = correct / total_predicted if total_predicted > 0 else 0.0
    recall = correct / total_actual if total_actual > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'correct': correct,
        'total_predicted': total_predicted,
        'total_actual': total_actual
    }


def analyze_errors(
    predicted_matches: List[Dict],
    ground_truth: Dict[str, str],
    bank_df: pd.DataFrame,
    check_df: pd.DataFrame
) -> Dict:
    
    errors = []
    correct_matches = []

    for match in predicted_matches:
        bank_id = match['bank_id']
        check_id = match['check_id']
        is_correct = ground_truth.get(bank_id) == check_id

        if not is_correct:
            true_check = ground_truth.get(bank_id, 'UNKNOWN')
            errors.append({
                **match,
                'true_check_id': true_check,
                'is_correct': False
            })
        else:
            correct_matches.append({**match, 'is_correct': True})

    # Categorize errors
    error_categories = defaultdict(int)
    for err in errors:
        if err['amount_diff'] > 0.01:
            error_categories['amount_mismatch'] += 1
        if err['date_diff_days'] > 5:
            error_categories['large_date_gap'] += 1
        if err['bank_type'] != err['check_type']:
            error_categories['type_mismatch'] += 1
        if err.get('text_sim', 0) < 0.3:
            error_categories['low_text_similarity'] += 1

    # Confidence distribution
    correct_confs = [m['confidence'] for m in correct_matches]
    error_confs = [e['confidence'] for e in errors]

    return {
        'total_errors': len(errors),
        'total_correct': len(correct_matches),
        'error_categories': dict(error_categories),
        'avg_correct_confidence': round(np.mean(correct_confs), 4) if correct_confs else 0,
        'avg_error_confidence': round(np.mean(error_confs), 4) if error_confs else 0,
        'errors': errors[:10],  # Top 10 errors for inspection
    }


def learning_curve(
    bank_df: pd.DataFrame,
    check_df: pd.DataFrame,
    ground_truth: Dict[str, str],
    ml_matcher,
    unique_matches: List[Dict],
    steps: List[float] = None
) -> List[Dict]:
    
    if steps is None:
        steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = []
    unique_bank_ids = {m['bank_id'] for m in unique_matches}
    unique_check_ids = {m['check_id'] for m in unique_matches}

    for frac in steps:
        # Reset matcher weights
        from src.ml_matcher import MLMatcher
        matcher = MLMatcher(svd_components=ml_matcher.svd_components)
        matcher.tfidf = ml_matcher.tfidf
        matcher.svd = ml_matcher.svd
        matcher.is_fitted = True

        # Use fraction of unique matches as validated training data
        n_train = int(len(unique_matches) * frac)
        train_matches = unique_matches[:n_train]

        if train_matches:
            # Mark all as correct (they are unique amount matches)
            validated = []
            for m in train_matches:
                vm = {**m, 'is_correct': ground_truth.get(m['bank_id']) == m['check_id']}
                validated.append(vm)
            matcher.add_validated_matches(validated)

        # Match remaining transactions
        ml_matches = matcher.match(
            bank_df, check_df,
            exclude_bank_ids=unique_bank_ids,
            exclude_check_ids=unique_check_ids
        )

        all_matches = unique_matches + ml_matches
        metrics = compute_metrics(all_matches, ground_truth)

        results.append({
            'training_fraction': frac,
            'n_training': n_train,
            'n_total_matches': len(all_matches),
            **metrics
        })

    return results


def generate_report(
    metrics: Dict,
    error_analysis: Dict,
    learning_results: List[Dict],
    feature_weights: Dict[str, float]
) -> str:
    
    lines = []
    lines.append("=" * 70)
    lines.append("FINANCIAL RECONCILIATION SYSTEM - PERFORMANCE REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Overall Metrics
    lines.append("OVERALL METRICS")
    lines.append("-" * 40)
    lines.append(f"  Precision:  {metrics['precision']:.4f}  ({metrics['correct']}/{metrics['total_predicted']} correct)")
    lines.append(f"  Recall:     {metrics['recall']:.4f}  ({metrics['correct']}/{metrics['total_actual']} found)")
    lines.append(f"  F1 Score:   {metrics['f1']:.4f}")
    lines.append("")

    # Feature Weights
    lines.append("LEARNED FEATURE WEIGHTS")
    lines.append("-" * 40)
    for feat, weight in sorted(feature_weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(weight * 50)
        lines.append(f"  {feat:25s} {weight:.4f}  {bar}")
    lines.append("")

    # Error Analysis
    lines.append("ERROR ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"  Total correct matches:  {error_analysis['total_correct']}")
    lines.append(f"  Total errors:           {error_analysis['total_errors']}")
    lines.append(f"  Avg correct confidence: {error_analysis['avg_correct_confidence']:.4f}")
    lines.append(f"  Avg error confidence:   {error_analysis['avg_error_confidence']:.4f}")
    lines.append("")
    if error_analysis['error_categories']:
        lines.append("  Error categories:")
        for cat, count in sorted(error_analysis['error_categories'].items(), key=lambda x: -x[1]):
            lines.append(f"    {cat:30s} {count}")
    lines.append("")

    # Learning Curve
    lines.append("LEARNING CURVE (Performance vs Training Data)")
    lines.append("-" * 60)
    lines.append(f"  {'Training %':>12s}  {'N Train':>8s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>8s}")
    for lr in learning_results:
        lines.append(
            f"  {lr['training_fraction']*100:>10.0f}%  "
            f"{lr['n_training']:>8d}  "
            f"{lr['precision']:>10.4f}  "
            f"{lr['recall']:>8.4f}  "
            f"{lr['f1']:>8.4f}"
        )
    lines.append("")

    # Sample Errors
    if error_analysis['errors']:
        lines.append("SAMPLE ERRORS (up to 10)")
        lines.append("-" * 60)
        for i, err in enumerate(error_analysis['errors'][:10]):
            lines.append(f"  Error {i+1}:")
            lines.append(f"    Predicted: {err['bank_id']} -> {err['check_id']}")
            lines.append(f"    Actual:    {err['bank_id']} -> {err['true_check_id']}")
            lines.append(f"    Confidence: {err['confidence']:.4f}")
            lines.append(f"    Bank desc:  {err.get('bank_desc', 'N/A')}")
            lines.append(f"    Check desc: {err.get('check_desc', 'N/A')}")
            lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
