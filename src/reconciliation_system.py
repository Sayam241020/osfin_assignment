import os
import pandas as pd
from typing import Dict, List, Tuple

from src.data_loader import load_bank_statements, load_check_register, get_ground_truth_mapping
from src.unique_matcher import find_unique_amount_matches
from src.ml_matcher import MLMatcher
from src.evaluator import compute_metrics, analyze_errors, learning_curve, generate_report


class ReconciliationSystem:

    def __init__(
        self,
        bank_path: str,
        check_path: str,
        svd_components: int = 50,
        verbose: bool = True
    ):
        
        self.bank_path = bank_path
        self.check_path = check_path
        self.svd_components = svd_components
        self.verbose = verbose
        self.bank_df = None
        self.check_df = None
        self.ground_truth = None
        self.unique_matches = []
        self.ml_matches = []
        self.all_matches = []
        self.ml_matcher = None
        self.metrics = None
        self.error_analysis = None
        self.learning_results = None

    def log(self, msg: str):
        if self.verbose:
            print(f"[RECON] {msg}")

    def run(self) -> Dict:
        
        self.log("=" * 60)
        self.log("FINANCIAL RECONCILIATION SYSTEM")
        self.log("=" * 60)

        # Step 1: Load data
        self._load_data()

        # Step 2: Unique amount matching
        self._unique_match()

        # Step 3: ML-based matching
        self._ml_match()

        # Step 4: Iterative learning
        self._iterative_learning()

        # Step 5: Final evaluation
        self._evaluate()

        # Step 6: Learning curve analysis
        self._learning_curve()

        # Step 7: Generate report
        report = self._generate_report()

        return {
            'metrics': self.metrics,
            'all_matches': self.all_matches,
            'report': report
        }

    def _load_data(self):
        """Load and preprocess both CSV files."""
        self.log("\nPhase 0: Loading data...")
        self.bank_df = load_bank_statements(self.bank_path)
        self.check_df = load_check_register(self.check_path)
        self.ground_truth = get_ground_truth_mapping(self.bank_df, self.check_df)
        self.log(f"  Bank transactions:  {len(self.bank_df)}")
        self.log(f"  Check transactions: {len(self.check_df)}")
        self.log(f"  Ground truth pairs: {len(self.ground_truth)}")

    def _unique_match(self):
        """Phase 1: Match transactions with unique amounts."""
        self.log("\nPhase 1: Unique Amount Matching...")
        self.unique_matches, flagged = find_unique_amount_matches(
            self.bank_df, self.check_df
        )
        metrics = compute_metrics(self.unique_matches, self.ground_truth)
        self.log(f"  Unique matches found: {len(self.unique_matches)}")
        self.log(f"  Flagged for review:   {len(flagged)}")
        self.log(f"  Precision: {metrics['precision']:.4f}")
        self.log(f"  Accuracy:  {metrics['correct']}/{len(self.unique_matches)}")
        if flagged:
            self.log(f"  Flags:")
            for f in flagged[:5]:
                self.log(f"    {f['bank_id']} -> {f['check_id']}: {f.get('flag', 'N/A')}")

    def _ml_match(self):
        self.log("\nPhase 2: ML-Based Matching...")

        # Initialize and fit the ML matcher
        self.ml_matcher = MLMatcher(svd_components=self.svd_components)
        self.ml_matcher.fit(self.bank_df, self.check_df)

        # Exclude already-matched transactions
        matched_bank_ids = {m['bank_id'] for m in self.unique_matches}
        matched_check_ids = {m['check_id'] for m in self.unique_matches}

        self.ml_matches = self.ml_matcher.match(
            self.bank_df, self.check_df,
            exclude_bank_ids=matched_bank_ids,
            exclude_check_ids=matched_check_ids
        )

        self.all_matches = self.unique_matches + self.ml_matches
        metrics = compute_metrics(self.all_matches, self.ground_truth)
        self.log(f"  ML matches found: {len(self.ml_matches)}")
        self.log(f"  Total matches:    {len(self.all_matches)}")
        self.log(f"  Combined Precision: {metrics['precision']:.4f}")
        self.log(f"  Combined Recall:    {metrics['recall']:.4f}")
        self.log(f"  Combined F1:        {metrics['f1']:.4f}")

    def _iterative_learning(self, rounds: int = 3):
        
        self.log("\nPhase 3: Iterative Learning...")

        matched_bank_ids = {m['bank_id'] for m in self.unique_matches}
        matched_check_ids = {m['check_id'] for m in self.unique_matches}

        for round_num in range(1, rounds + 1):
            self.log(f"\n  --- Learning Round {round_num} ---")

            # Validate current ML matches against ground truth
            validated = []
            for m in self.ml_matches:
                is_correct = self.ground_truth.get(m['bank_id']) == m['check_id']
                validated.append({**m, 'is_correct': is_correct})

            # Feed validated matches to improve the model
            self.ml_matcher.add_validated_matches(validated)

            self.log(f"  Feature weights: {self.ml_matcher.get_feature_weights()}")

            # Re-match with updated weights
            self.ml_matches = self.ml_matcher.match(
                self.bank_df, self.check_df,
                exclude_bank_ids=matched_bank_ids,
                exclude_check_ids=matched_check_ids
            )

            self.all_matches = self.unique_matches + self.ml_matches
            metrics = compute_metrics(self.all_matches, self.ground_truth)
            self.log(f"  Precision: {metrics['precision']:.4f}")
            self.log(f"  Recall:    {metrics['recall']:.4f}")
            self.log(f"  F1:        {metrics['f1']:.4f}")

    def _evaluate(self):
        """Compute final metrics and error analysis."""
        self.log("\nPhase 4: Final Evaluation...")
        self.metrics = compute_metrics(self.all_matches, self.ground_truth)
        self.error_analysis = analyze_errors(
            self.all_matches, self.ground_truth,
            self.bank_df, self.check_df
        )

    def _learning_curve(self):
        self.log("\nPhase 5: Learning Curve Analysis...")
        self.learning_results = learning_curve(
            self.bank_df, self.check_df,
            self.ground_truth, self.ml_matcher,
            self.unique_matches
        )

    def _generate_report(self) -> str:
        """Generate and return the performance report."""
        report = generate_report(
            self.metrics,
            self.error_analysis,
            self.learning_results,
            self.ml_matcher.get_feature_weights()
        )
        return report

    def save_results(self, output_dir: str = "results"):
        
        os.makedirs(output_dir, exist_ok=True)

        # Save matches CSV
        matches_df = pd.DataFrame(self.all_matches)
        cols_to_save = [
            'bank_id', 'check_id', 'amount', 'confidence',
            'match_method', 'date_diff_days', 'bank_desc', 'check_desc',
            'bank_type', 'check_type'
        ]
        cols_available = [c for c in cols_to_save if c in matches_df.columns]
        matches_df[cols_available].to_csv(
            os.path.join(output_dir, 'matches.csv'), index=False
        )
        self.log(f"\n  Matches saved to {output_dir}/matches.csv")

        # Save report
        report = self._generate_report()
        with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
            f.write(report)
        self.log(f"  Report saved to {output_dir}/report.txt")

        # Save learning curve
        lr_df = pd.DataFrame(self.learning_results)
        lr_df.to_csv(os.path.join(output_dir, 'learning_curve.csv'), index=False)
        self.log(f"  Learning curve saved to {output_dir}/learning_curve.csv")
