
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def find_unique_amount_matches(
    bank_df: pd.DataFrame,
    check_df: pd.DataFrame,
    date_threshold_days: int = 10
) -> Tuple[List[Dict], List[Dict]]:
    
    bank_amt_counts = bank_df['amount'].value_counts()
    check_amt_counts = check_df['amount'].value_counts()

    # Find amounts that are unique in BOTH sources
    bank_unique_amts = set(bank_amt_counts[bank_amt_counts == 1].index)
    check_unique_amts = set(check_amt_counts[check_amt_counts == 1].index)
    unique_amts = bank_unique_amts & check_unique_amts

    matched_pairs = []
    flagged_pairs = []

    for amt in unique_amts:
        bank_row = bank_df[bank_df['amount'] == amt].iloc[0]
        check_row = check_df[check_df['amount'] == amt].iloc[0]

        # Calculate date difference
        date_diff = abs((bank_row['date'] - check_row['date']).days)

        # Calculate confidence score
        confidence = _calculate_unique_confidence(bank_row, check_row, date_diff)

        match_info = {
            'bank_id': bank_row['transaction_id'],
            'check_id': check_row['transaction_id'],
            'amount': amt,
            'bank_date': bank_row['date'],
            'check_date': check_row['date'],
            'date_diff_days': date_diff,
            'bank_desc': bank_row['description'],
            'check_desc': check_row['description'],
            'bank_type': bank_row['type_normalized'],
            'check_type': check_row['type_normalized'],
            'confidence': confidence,
            'match_method': 'unique_amount'
        }

        matched_pairs.append(match_info)

        # Flag potential issues
        if date_diff > date_threshold_days:
            match_info['flag'] = f'Date difference ({date_diff} days) exceeds threshold ({date_threshold_days} days)'
            flagged_pairs.append(match_info)
        elif bank_row['type_normalized'] != check_row['type_normalized']:
            match_info['flag'] = f'Type mismatch: {bank_row["type_normalized"]} vs {check_row["type_normalized"]}'
            flagged_pairs.append(match_info)

    # Sort by confidence (highest first)
    matched_pairs.sort(key=lambda x: x['confidence'], reverse=True)

    return matched_pairs, flagged_pairs


def _calculate_unique_confidence(
    bank_row: pd.Series,
    check_row: pd.Series,
    date_diff: int
) -> float:
    
    # Amount score: always perfect for unique matches
    amount_score = 1.0

    # Date proximity score: exponential decay
    date_score = np.exp(-date_diff / 5.0)

    # Type agreement score
    type_score = 1.0 if bank_row['type_normalized'] == check_row['type_normalized'] else 0.3

    # Basic description similarity (word overlap)
    bank_words = set(bank_row['description_clean'].split())
    check_words = set(check_row['description_clean'].split())
    if bank_words and check_words:
        overlap = len(bank_words & check_words)
        total = len(bank_words | check_words)
        desc_score = overlap / total if total > 0 else 0.0
    else:
        desc_score = 0.0

    # Weighted combination
    confidence = (
        0.4 * amount_score +
        0.3 * date_score +
        0.2 * type_score +
        0.1 * desc_score
    )

    return round(min(confidence, 1.0), 4)
