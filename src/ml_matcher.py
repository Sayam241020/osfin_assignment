import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional


class MLMatcher:

    def __init__(self, svd_components: int = 50, random_state: int = 42):
        
        self.svd_components = svd_components
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=5000,
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(
            n_components=svd_components,
            random_state=random_state
        )
        self.feature_weights = {
            'text_similarity': 0.35,
            'amount_similarity': 0.30,
            'date_similarity': 0.20,
            'type_similarity': 0.15
        }
        self.validated_matches = []
        self.is_fitted = False

    def fit(self, bank_df: pd.DataFrame, check_df: pd.DataFrame):
        
        all_descriptions = pd.concat([
            bank_df['description_clean'],
            check_df['description_clean']
        ]).values

        tfidf_matrix = self.tfidf.fit_transform(all_descriptions)

        n_components = min(self.svd_components, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True

    def compute_similarity_matrix(
        self,
        bank_df: pd.DataFrame,
        check_df: pd.DataFrame
    ) -> np.ndarray:
        
        if not self.is_fitted:
            self.fit(bank_df, check_df)

        # 1. Text similarity via TF-IDF + SVD + cosine
        text_sim = self._compute_text_similarity(bank_df, check_df)

        # 2. Amount similarity (Gaussian kernel)
        amount_sim = self._compute_amount_similarity(bank_df, check_df)

        # 3. Date similarity (exponential decay)
        date_sim = self._compute_date_similarity(bank_df, check_df)

        # 4. Type similarity (binary match)
        type_sim = self._compute_type_similarity(bank_df, check_df)

        # Weighted combination
        combined = (
            self.feature_weights['text_similarity'] * text_sim +
            self.feature_weights['amount_similarity'] * amount_sim +
            self.feature_weights['date_similarity'] * date_sim +
            self.feature_weights['type_similarity'] * type_sim
        )

        return combined

    def match(
        self,
        bank_df: pd.DataFrame,
        check_df: pd.DataFrame,
        exclude_bank_ids: set = None,
        exclude_check_ids: set = None
    ) -> List[Dict]:
        # Filter out already-matched transactions
        if exclude_bank_ids:
            bank_df = bank_df[~bank_df['transaction_id'].isin(exclude_bank_ids)].copy()
        if exclude_check_ids:
            check_df = check_df[~check_df['transaction_id'].isin(exclude_check_ids)].copy()

        if len(bank_df) == 0 or len(check_df) == 0:
            return []

        bank_df = bank_df.reset_index(drop=True)
        check_df = check_df.reset_index(drop=True)

        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(bank_df, check_df)

        # Use Hungarian algorithm to find optimal assignment
        # Convert similarity to cost (Hungarian minimizes cost)
        cost_matrix = 1.0 - sim_matrix
        bank_indices, check_indices = linear_sum_assignment(cost_matrix)

        matches = []
        for b_idx, c_idx in zip(bank_indices, check_indices):
            confidence = sim_matrix[b_idx, c_idx]
            bank_row = bank_df.iloc[b_idx]
            check_row = check_df.iloc[c_idx]

            match_info = {
                'bank_id': bank_row['transaction_id'],
                'check_id': check_row['transaction_id'],
                'amount': bank_row['amount'],
                'check_amount': check_row['amount'],
                'amount_diff': abs(bank_row['amount'] - check_row['amount']),
                'bank_date': bank_row['date'],
                'check_date': check_row['date'],
                'date_diff_days': abs((bank_row['date'] - check_row['date']).days),
                'bank_desc': bank_row['description'],
                'check_desc': check_row['description'],
                'bank_type': bank_row['type_normalized'],
                'check_type': check_row['type_normalized'],
                'confidence': round(float(confidence), 4),
                'match_method': 'ml_hybrid',
                'text_sim': round(float(self._last_text_sim[b_idx, c_idx]), 4),
                'amount_sim': round(float(self._last_amount_sim[b_idx, c_idx]), 4),
                'date_sim': round(float(self._last_date_sim[b_idx, c_idx]), 4),
                'type_sim': round(float(self._last_type_sim[b_idx, c_idx]), 4),
            }
            matches.append(match_info)

        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches

    def add_validated_matches(self, validated: List[Dict]):
        self.validated_matches.extend(validated)
        self._update_weights()

    def _update_weights(self):
        if len(self.validated_matches) < 5:
            return

        correct = [m for m in self.validated_matches if m.get('is_correct', True)]
        incorrect = [m for m in self.validated_matches if not m.get('is_correct', True)]

        if not correct:
            return

        # Calculate average feature values for correct matches
        features = ['text_sim', 'amount_sim', 'date_sim', 'type_sim']
        weight_keys = ['text_similarity', 'amount_similarity', 'date_similarity', 'type_similarity']

        correct_means = {}
        for i, feat in enumerate(features):
            vals = [m.get(feat, 0.5) for m in correct if feat in m]
            correct_means[weight_keys[i]] = np.mean(vals) if vals else 0.5

        incorrect_means = {}
        if incorrect:
            for i, feat in enumerate(features):
                vals = [m.get(feat, 0.5) for m in incorrect if feat in m]
                incorrect_means[weight_keys[i]] = np.mean(vals) if vals else 0.5

        # Adjust weights: increase for features that discriminate well
        new_weights = {}
        for key in weight_keys:
            base = correct_means.get(key, 0.5)
            if incorrect_means:
                penalty = incorrect_means.get(key, 0.5)
                discriminative_power = base - penalty + 0.5  # Shift to positive
            else:
                discriminative_power = base
            new_weights[key] = max(0.05, discriminative_power)

        # Normalize
        total = sum(new_weights.values())
        for key in new_weights:
            new_weights[key] = round(new_weights[key] / total, 4)

        self.feature_weights = new_weights

    def _compute_text_similarity(
        self, bank_df: pd.DataFrame, check_df: pd.DataFrame
    ) -> np.ndarray:
        bank_tfidf = self.tfidf.transform(bank_df['description_clean'])
        check_tfidf = self.tfidf.transform(check_df['description_clean'])

        bank_svd = self.svd.transform(bank_tfidf)
        check_svd = self.svd.transform(check_tfidf)

        sim = cosine_similarity(bank_svd, check_svd)
        # Normalize to [0, 1]
        sim = (sim + 1) / 2
        self._last_text_sim = sim
        return sim

    def _compute_amount_similarity(
        self, bank_df: pd.DataFrame, check_df: pd.DataFrame
    ) -> np.ndarray:
        bank_amounts = bank_df['amount'].values.reshape(-1, 1)
        check_amounts = check_df['amount'].values.reshape(1, -1)

        # Absolute difference
        diff = np.abs(bank_amounts - check_amounts)

        # Gaussian similarity: sigma = 1.0 for tight matching
        sigma = 1.0
        sim = np.exp(-(diff ** 2) / (2 * sigma ** 2))

        self._last_amount_sim = sim
        return sim

    def _compute_date_similarity(
        self, bank_df: pd.DataFrame, check_df: pd.DataFrame
    ) -> np.ndarray:
        bank_dates = bank_df['date'].values.astype('datetime64[D]').astype(int).reshape(-1, 1)
        check_dates = check_df['date'].values.astype('datetime64[D]').astype(int).reshape(1, -1)

        # Absolute date difference in days
        diff = np.abs(bank_dates - check_dates)

        # Exponential decay with tau = 3 days
        tau = 3.0
        sim = np.exp(-diff / tau)

        self._last_date_sim = sim
        return sim

    def _compute_type_similarity(
        self, bank_df: pd.DataFrame, check_df: pd.DataFrame
    ) -> np.ndarray:

        bank_types = bank_df['type_normalized'].values.reshape(-1, 1)
        check_types = check_df['type_normalized'].values.reshape(1, -1)

        sim = (bank_types == check_types).astype(float)
        # Give partial credit for mismatches (some may be data entry errors)
        sim = np.where(sim == 1.0, 1.0, 0.2)

        self._last_type_sim = sim
        return sim

    def get_feature_weights(self) -> Dict[str, float]:
        
        return self.feature_weights.copy()
