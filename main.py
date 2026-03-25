#!/usr/bin/env python3
"""
Financial Reconciliation System - Main CLI

Demonstrates the full reconciliation workflow:
  1. Load bank statements and check register data
  2. Match transactions with unique amounts (Phase 1)
  3. ML-based matching for remaining transactions (Phase 2)
  4. Iterative learning from validated matches (Phase 3)
  5. Evaluate and report results

Usage:
    python main.py
    python main.py --bank data/bank_statements.csv --check data/check_register.csv
"""

import argparse
import os
import sys

from src.reconciliation_system import ReconciliationSystem


def main():
    parser = argparse.ArgumentParser(
        description='Financial Reconciliation System Using Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --bank bank_statements.csv --check check_register.csv
  python main.py --svd-dim 100 --output results/
        """
    )
    parser.add_argument(
        '--bank', type=str, default='bank_statements.csv',
        help='Path to bank statements CSV (default: bank_statements.csv)'
    )
    parser.add_argument(
        '--check', type=str, default='check_register.csv',
        help='Path to check register CSV (default: check_register.csv)'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory for results (default: results/)'
    )
    parser.add_argument(
        '--svd-dim', type=int, default=50,
        help='Number of SVD dimensions for text embeddings (default: 50)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress messages'
    )
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.bank):
        print(f"Error: Bank statements file not found: {args.bank}")
        sys.exit(1)
    if not os.path.exists(args.check):
        print(f"Error: Check register file not found: {args.check}")
        sys.exit(1)

    # Run reconciliation
    system = ReconciliationSystem(
        bank_path=args.bank,
        check_path=args.check,
        svd_components=args.svd_dim,
        verbose=not args.quiet
    )

    result = system.run()

    # Print report
    print("\n")
    print(result['report'])

    # Save results
    system.save_results(args.output)

    # Summary
    metrics = result['metrics']
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Total transactions matched: {metrics['total_predicted']}/{metrics['total_actual']}")
    print(f"  Correctly matched:          {metrics['correct']}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\n  Results saved to: {args.output}/")
    print(f"  - matches.csv:       All matched transaction pairs")
    print(f"  - report.txt:        Detailed performance report")
    print(f"  - learning_curve.csv: Learning curve data")


if __name__ == '__main__':
    main()
