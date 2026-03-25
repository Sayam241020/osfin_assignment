"""
Data Loading and Preprocessing Module

Handles loading bank statements and check register CSVs,
normalizing fields for reconciliation comparison.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


def load_bank_statements(filepath: str) -> pd.DataFrame:
    """Load and preprocess bank statement transactions.

    Args:
        filepath: Path to bank_statements.csv

    Returns:
        DataFrame with normalized columns
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float).round(2)
    df['type_normalized'] = df['type'].map({'DEBIT': 'DR', 'CREDIT': 'CR'}).fillna(df['type'])
    df['description_clean'] = df['description'].apply(clean_description)
    df['source'] = 'bank'
    return df


def load_check_register(filepath: str) -> pd.DataFrame:
    """Load and preprocess check register transactions.

    Args:
        filepath: Path to check_register.csv

    Returns:
        DataFrame with normalized columns
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float).round(2)
    df['type_normalized'] = df['type'].str.upper().str.strip()
    # Normalize CR/DR variants
    df['type_normalized'] = df['type_normalized'].replace({
        'DEBIT': 'DR', 'CREDIT': 'CR',
        'D': 'DR', 'C': 'CR',
        'Dr': 'DR', 'Cr': 'CR'
    })
    df['description_clean'] = df['description'].apply(clean_description)
    df['source'] = 'check'
    return df


def clean_description(text: str) -> str:
    """Normalize a transaction description for comparison.

    - Lowercases
    - Removes special characters except spaces
    - Collapses multiple spaces
    - Strips leading/trailing whitespace

    Args:
        text: Raw description string

    Returns:
        Cleaned description string
    """
    if pd.isna(text):
        return ''
    text = str(text).lower()
    # Remove common prefixes like check numbers, reference codes
    text = re.sub(r'#\d+', '', text)
    text = re.sub(r'\b\d{4,}\b', '', text)  # Remove long numeric codes
    # Remove special characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_ground_truth_mapping(bank_df: pd.DataFrame, check_df: pd.DataFrame) -> dict:
    """Derive ground truth mapping from transaction IDs.

    Since both CSVs describe the same 308 transactions, we match
    by the numeric suffix of transaction_id (B0001 -> R0001, etc.).

    Args:
        bank_df: Bank statements DataFrame
        check_df: Check register DataFrame

    Returns:
        Dict mapping bank transaction_id -> check transaction_id
    """
    ground_truth = {}
    bank_nums = {}
    check_nums = {}

    for _, row in bank_df.iterrows():
        tid = row['transaction_id']
        num = int(re.search(r'\d+', tid).group())
        bank_nums[num] = tid

    for _, row in check_df.iterrows():
        tid = row['transaction_id']
        num = int(re.search(r'\d+', tid).group())
        check_nums[num] = tid

    for num in bank_nums:
        if num in check_nums:
            ground_truth[bank_nums[num]] = check_nums[num]

    return ground_truth
