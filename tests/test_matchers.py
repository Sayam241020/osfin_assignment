"""
Unit Tests for Financial Reconciliation System

Tests critical functions in data loading, unique matching,
ML matching, and evaluation modules.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_loader import clean_description, get_ground_truth_mapping
from src.unique_matcher import find_unique_amount_matches, _calculate_unique_confidence
from src.ml_matcher import MLMatcher
from src.evaluator import compute_metrics


class TestCleanDescription(unittest.TestCase):
    """Test description cleaning and normalization."""

    def test_basic_cleaning(self):
        self.assertEqual(clean_description("WALMART STORE #1234"), "walmart store")

    def test_lowercase(self):
        self.assertEqual(clean_description("GROCERY STORE"), "grocery store")

    def test_remove_special_chars(self):
        result = clean_description("Payment - ACH/Wire")
        self.assertNotIn("-", result)
        self.assertNotIn("/", result)

    def test_collapse_spaces(self):
        result = clean_description("too   many    spaces")
        self.assertEqual(result, "too many spaces")

    def test_empty_string(self):
        self.assertEqual(clean_description(""), "")

    def test_nan_input(self):
        self.assertEqual(clean_description(float('nan')), "")


class TestGroundTruth(unittest.TestCase):
    """Test ground truth mapping derivation."""

    def test_basic_mapping(self):
        bank_df = pd.DataFrame({
            'transaction_id': ['B0001', 'B0002', 'B0003']
        })
        check_df = pd.DataFrame({
            'transaction_id': ['R0001', 'R0002', 'R0003']
        })
        mapping = get_ground_truth_mapping(bank_df, check_df)
        self.assertEqual(mapping['B0001'], 'R0001')
        self.assertEqual(mapping['B0002'], 'R0002')
        self.assertEqual(mapping['B0003'], 'R0003')
        self.assertEqual(len(mapping), 3)


class TestUniqueAmountMatching(unittest.TestCase):
    """Test unique amount matching logic."""

    def setUp(self):
        """Create test DataFrames."""
        self.bank_df = pd.DataFrame({
            'transaction_id': ['B0001', 'B0002', 'B0003', 'B0004'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'description': ['Grocery', 'Gas', 'Electric', 'Food'],
            'amount': [100.0, 200.0, 300.0, 200.0],  # 200 is non-unique
            'type': ['DEBIT', 'DEBIT', 'CREDIT', 'DEBIT'],
            'type_normalized': ['DR', 'DR', 'CR', 'DR'],
            'description_clean': ['grocery', 'gas', 'electric', 'food'],
            'balance': [1000, 800, 1100, 900]
        })
        self.check_df = pd.DataFrame({
            'transaction_id': ['R0001', 'R0002', 'R0003', 'R0004'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03']),
            'description': ['Groceries', 'Fuel', 'Power bill', 'Dinner'],
            'amount': [100.0, 200.0, 300.0, 200.0],
            'type': ['DR', 'DR', 'CR', 'DR'],
            'type_normalized': ['DR', 'DR', 'CR', 'DR'],
            'description_clean': ['groceries', 'fuel', 'power bill', 'dinner'],
            'category': ['food', 'transport', 'utilities', 'food'],
            'notes': ['', '', '', '']
        })

    def test_finds_unique_amounts(self):
        matches, _ = find_unique_amount_matches(self.bank_df, self.check_df)
        # Only 100.0 and 300.0 are unique in both; 200.0 appears twice
        matched_amounts = {m['amount'] for m in matches}
        self.assertIn(100.0, matched_amounts)
        self.assertIn(300.0, matched_amounts)
        self.assertNotIn(200.0, matched_amounts)

    def test_confidence_scores(self):
        matches, _ = find_unique_amount_matches(self.bank_df, self.check_df)
        for m in matches:
            self.assertGreaterEqual(m['confidence'], 0.0)
            self.assertLessEqual(m['confidence'], 1.0)


class TestMLMatcher(unittest.TestCase):
    """Test ML matcher functionality."""

    def setUp(self):
        self.bank_df = pd.DataFrame({
            'transaction_id': ['B0001', 'B0002', 'B0003'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'description': ['Walmart grocery shopping', 'Shell gas station', 'Netflix subscription'],
            'amount': [150.0, 50.0, 15.99],
            'type_normalized': ['DR', 'DR', 'DR'],
            'description_clean': ['walmart grocery shopping', 'shell gas station', 'netflix subscription'],
        })
        self.check_df = pd.DataFrame({
            'transaction_id': ['R0001', 'R0002', 'R0003'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
            'description': ['Groceries at Walmart', 'Gas fill up Shell', 'Netflix monthly'],
            'amount': [150.0, 50.0, 15.99],
            'type_normalized': ['DR', 'DR', 'DR'],
            'description_clean': ['groceries at walmart', 'gas fill up shell', 'netflix monthly'],
        })

    def test_similarity_matrix_shape(self):
        matcher = MLMatcher(svd_components=5)
        matcher.fit(self.bank_df, self.check_df)
        sim = matcher.compute_similarity_matrix(self.bank_df, self.check_df)
        self.assertEqual(sim.shape, (3, 3))

    def test_match_returns_correct_count(self):
        matcher = MLMatcher(svd_components=5)
        matcher.fit(self.bank_df, self.check_df)
        matches = matcher.match(self.bank_df, self.check_df)
        self.assertEqual(len(matches), 3)

    def test_match_has_confidence(self):
        matcher = MLMatcher(svd_components=5)
        matcher.fit(self.bank_df, self.check_df)
        matches = matcher.match(self.bank_df, self.check_df)
        for m in matches:
            self.assertIn('confidence', m)
            self.assertGreaterEqual(m['confidence'], 0.0)
            self.assertLessEqual(m['confidence'], 1.0)


class TestEvaluationMetrics(unittest.TestCase):
    """Test precision, recall, F1 computation."""

    def test_perfect_matching(self):
        predicted = [
            {'bank_id': 'B0001', 'check_id': 'R0001'},
            {'bank_id': 'B0002', 'check_id': 'R0002'},
        ]
        ground_truth = {'B0001': 'R0001', 'B0002': 'R0002'}
        metrics = compute_metrics(predicted, ground_truth)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)

    def test_partial_matching(self):
        predicted = [
            {'bank_id': 'B0001', 'check_id': 'R0001'},
            {'bank_id': 'B0002', 'check_id': 'R0003'},  # Wrong
        ]
        ground_truth = {'B0001': 'R0001', 'B0002': 'R0002', 'B0003': 'R0003'}
        metrics = compute_metrics(predicted, ground_truth)
        self.assertAlmostEqual(metrics['precision'], 0.5)
        self.assertAlmostEqual(metrics['recall'], 1/3, places=3)

    def test_no_matches(self):
        metrics = compute_metrics([], {'B0001': 'R0001'})
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)
        self.assertEqual(metrics['f1'], 0.0)


if __name__ == '__main__':
    unittest.main()
