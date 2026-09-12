from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib


class TabularSequenceDataset(Dataset):
    """Convert transactional CSV into sequence windows per account.

    Parameters
    - csv_path: path to CSV
    - id_col: column identifying user/account
    - time_col: column with timestamp (will be sorted)
    - seq_len: sliding window length (number of transactions)
    - step: window step
    - features: list of feature columns to use; if None infer all except id/time/label
    - label_col: name of binary label column
    - categorical_cols: list of categorical columns
    - impute_strategy: 'median'|'mean'|'most_frequent'
    - normalize: whether to StandardScale continuous features
    - smote: whether to apply SMOTE at sequence-aggregate level (optional)
    """

    def __init__(
        self,
        csv_path: str,
        id_col: str = "account_id",
        time_col: str = "transaction_time",
        seq_len: int = 50,
        step: int = 1,
        features: Optional[List[str]] = None,
        label_col: str = "label",
        categorical_cols: Optional[List[str]] = None,
        impute_strategy: str = "median",
        normalize: bool = True,
        smote: bool = False,
        seed: int = 42,
    ):
        self.csv_path = csv_path
        self.id_col = id_col
        self.time_col = time_col
        self.seq_len = seq_len
        self.step = step
        self.label_col = label_col
        self.categorical_cols = categorical_cols or []
        self.impute_strategy = impute_strategy
        self.normalize = normalize
        self.smote = smote
        self.seed = seed

        self.df = pd.read_csv(csv_path)
        self._fit_transforms(features)
        self.X_windows, self.y = self._build_windows()

    def _fit_transforms(self, features: Optional[List[str]]):
        drop_cols = {self.id_col, self.time_col, self.label_col}
        if features is None:
            features = [c for c in self.df.columns if c not in drop_cols]
        self.features = features

        # Separate numeric / categorical
        self.cat_cols = [c for c in self.features if c in self.categorical_cols]
        self.num_cols = []
        for c in self.features:
            if c in self.cat_cols:
                continue
            col = self.df[c]
            if pd.api.types.is_numeric_dtype(col):
                self.num_cols.append(c)
                continue
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().all():
                self.df[c] = coerced
                self.num_cols.append(c)
            elif coerced.notna().sum() / len(coerced) >= 0.95:
                self.df[c] = coerced
                self.num_cols.append(c)
            else:
                self.cat_cols.append(c)

        # Imputer for numeric
        self.num_imputer = SimpleImputer(strategy=self.impute_strategy)
        if len(self.num_cols) > 0:
            self.num_imputer.fit(self.df[self.num_cols])

        # One-hot encoder for categorical
        if len(self.cat_cols) > 0:
            # sklearn changed the OneHotEncoder 'sparse' argument name to 'sparse_output' in newer versions.
            try:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_df = self.df[self.cat_cols].fillna("__NA__").astype(str)
            self.ohe.fit(cat_df)
            self.cat_feature_names = list(self.ohe.get_feature_names_out(self.cat_cols))
        else:
            self.ohe = None
            self.cat_feature_names = []

        # Scaler for numeric
        if self.normalize and len(self.num_cols) > 0:
            self.scaler = StandardScaler()
            num_filled = pd.DataFrame(self.num_imputer.transform(self.df[self.num_cols]), columns=self.num_cols)
            self.scaler.fit(num_filled)
        else:
            self.scaler = None

    def _transform_row(self, df_slice: pd.DataFrame) -> np.ndarray:
        # df_slice: (seq_len, ) rows
        num_part = np.empty((len(df_slice), 0))
        if len(self.num_cols) > 0:
            vals = self.num_imputer.transform(df_slice[self.num_cols])
            if self.scaler is not None:
                vals = self.scaler.transform(vals)
            num_part = vals

        cat_part = np.empty((len(df_slice), 0))
        if self.ohe is not None:
            cat_vals = df_slice[self.cat_cols].fillna("__NA__").astype(str)
            cat_part = self.ohe.transform(cat_vals)

        combined = np.hstack([num_part, cat_part]) if (num_part.size or cat_part.size) else np.zeros((len(df_slice), 0))
        return combined.astype(np.float32)

    def _build_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        grouped = self.df.groupby(self.id_col)
        for _, group in grouped:
            group_sorted = group.sort_values(self.time_col)
            vals = group_sorted
            n = len(vals)
            for start in range(0, max(1, n - self.seq_len + 1), self.step):
                end = start + self.seq_len
                if end > n:
                    # pad by repeating last row
                    window = vals.iloc[max(0, n - self.seq_len):n]
                else:
                    window = vals.iloc[start:end]

                Xw = self._transform_row(window)
                # Determine label for window: if any transaction labeled positive
                yw = int(window[self.label_col].astype(int).sum() > 0)
                X_list.append(Xw)
                y_list.append(yw)

        X_arr = np.stack([self._pad_or_trim(x) for x in X_list], axis=0)
        y_arr = np.array(y_list, dtype=np.int64)

        if self.smote:
            X_arr, y_arr = self._sequence_smote_upsample(X_arr, y_arr)

        return X_arr, y_arr

    def _pad_or_trim(self, arr: np.ndarray) -> np.ndarray:
        # Ensure shape (seq_len, feature_dim)
        s, f = arr.shape
        if s == self.seq_len:
            return arr
        if s < self.seq_len:
            pad = np.repeat(arr[-1:, :], repeats=(self.seq_len - s), axis=0)
            return np.vstack([arr, pad])
        return arr[: self.seq_len, :]

    def _sequence_smote_upsample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Simple approach: aggregate per-sequence features and run SMOTE on aggregates,
        # then map synthetic aggregates back to nearest real sequences (heuristic).
        from sklearn.neighbors import NearestNeighbors

        agg = X.mean(axis=1)
        sm = SMOTE(random_state=self.seed)
        X_syn, y_syn = sm.fit_resample(agg, y)

        # If SMOTE created new samples, find nearest real sequence for each synthetic aggregate
        if len(X_syn) > len(agg):
            nbrs = NearestNeighbors(n_neighbors=1).fit(agg)
            dists, idxs = nbrs.kneighbors(X_syn[len(agg) :])
            synthetic_sequences = X[idxs.flatten()]
            X_new = np.vstack([X, synthetic_sequences])
            y_new = np.hstack([y, np.ones(len(synthetic_sequences), dtype=np.int64)])
            return X_new, y_new
        return X, y

    def get_class_weights(self) -> torch.Tensor:
        # For dynamic loss weighting
        labels = self.y
        classes, counts = np.unique(labels, return_counts=True)
        weights = {c: max(1.0, float(len(labels)) / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
        weight_list = [weights.get(i, 1.0) for i in range(max(classes) + 1)]
        return torch.tensor(weight_list, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X_windows[idx]), int(self.y[idx])

    def get_dataloader(self, batch_size: int = 128, shuffle: bool = True, num_workers: int = 0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    # small usage example
    ds = TabularSequenceDataset(
        csv_path="data/sample_transactions.csv",
        id_col="account_id",
        time_col="transaction_time",
        seq_len=20,
        categorical_cols=["merchant_category"],
        label_col="is_fraud",
        smote=False,
    )
    dl = ds.get_dataloader(batch_size=16)
    xb, yb = next(iter(dl))
    print("Batch X shape:", xb.shape)
    print("Batch y shape:", yb.shape)
