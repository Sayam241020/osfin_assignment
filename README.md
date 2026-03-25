# Financial Reconciliation System Using Machine Learning

An unsupervised ML-based system that automatically matches transactions between bank statements and internal check registers, inspired by [Chew, 2020](https://doi.org/10.1145/3383455.3422539).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the reconciliation system
python main.py

# Run with custom paths
python main.py --bank bank_statements.csv --check check_register.csv --output results/

# Run unit tests
python -m pytest tests/ -v
```

## Architecture

```
├── main.py                       # CLI entry point
├── src/
│   ├── data_loader.py            # CSV loading & normalization
│   ├── unique_matcher.py         # Phase 1: unique amount matching
│   ├── ml_matcher.py             # Phase 2: hybrid ML matching
│   ├── evaluator.py              # Metrics & analysis
│   └── reconciliation_system.py  # Pipeline orchestrator
├── tests/
│   └── test_matchers.py          # Unit tests
├── bank_statements.csv           # Input data (308 transactions)
├── check_register.csv            # Input data (308 transactions)
└── results/                      # Output directory
    ├── matches.csv               # Matched pairs with confidence
    ├── report.txt                # Performance report
    └── learning_curve.csv        # Learning curve data
```

## Approach: Hybrid ML (Option C)

This system uses a **custom hybrid approach** combining ideas from the research paper with modern techniques:

### Phase 1: Unique Amount Matching
Transactions with amounts that appear exactly once in both sources are matched directly with high confidence. Each match is scored using date proximity, type agreement, and description overlap.

### Phase 2: ML-Based Matching (TF-IDF + SVD)
For remaining transactions, a multi-feature similarity approach:

1. **Text Similarity** — TF-IDF character n-grams (2-4) on cleaned descriptions, reduced via SVD (50 dimensions), scored by cosine similarity. This captures textual variations between sources (e.g., "WALMART STORE" vs "Groceries at Walmart").

2. **Amount Similarity** — Gaussian kernel on amount differences (σ=1.0), handling small rounding discrepancies.

3. **Date Similarity** — Exponential decay (τ=3 days) on date differences, accounting for typical 0-5 day recording lags.

4. **Type Similarity** — Normalized type comparison (DEBIT/CREDIT → DR/CR) with partial credit for mismatches.

5. **Optimal Assignment** — Hungarian algorithm ensures globally optimal 1-to-1 matching.

### Phase 3: Iterative Learning
The system re-weights features based on validated match accuracy, implementing the match → review → improve cycle. Feature weights are adjusted to increase discrimination between correct and incorrect matches.

## Design Decisions

- **Character n-grams over word tokens**: More robust to abbreviations and different wordings (e.g., "CHVRON" vs "Chevron gas")
- **Hungarian algorithm over greedy matching**: Guarantees globally optimal assignment
- **Feature weight learning**: Adapts to dataset-specific patterns without labeled data
- **SVD dimensionality reduction**: Captures latent semantic relationships between description terms, as suggested by the research paper

## Output

Results are saved to the `results/` directory:
- `matches.csv`: All 308 matched pairs with confidence scores
- `report.txt`: Detailed performance analysis (precision, recall, F1)
- `learning_curve.csv`: Performance improvement with training data

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, scipy
