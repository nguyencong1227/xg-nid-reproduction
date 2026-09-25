"""Flow-level feature loading, preprocessing and the zero-day split.

The 82 statistical flow features of CIC-IoT2023 as produced by the XG-NID
extractor (nfstream + Additional_Features.py).  Packet-level columns
(`udps.*`) are dropped: NI-Diff operates at the flow level (Sec. II).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Domain constraints (Sec. IV, Attack Model)
#   "we put constraints on the symbolic features and only modify those
#    statistical features such as average length, inter arrival time, etc."
# --------------------------------------------------------------------------- #
MUTABLE_FEATURES: Tuple[str, ...] = (
    'src2dst_duration_ms', 'dst2src_duration_ms',
    # packet-size statistics
    'bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps',
    'bidirectional_max_ps',
    'src2dst_min_ps', 'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps',
    'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps', 'dst2src_max_ps',
    # inter-arrival-time statistics
    'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms',
    'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms',
    'src2dst_min_piat_ms', 'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms',
    'src2dst_max_piat_ms',
    'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms',
    'dst2src_max_piat_ms',
    'packet_size_variation',
    'Rolling_Duration_Destination', 'Rolling_Duration_SourceDestination',
)

# (min, mean, max) triples that must stay ordered in the raw feature space
ORDERED_TRIPLES: Tuple[Tuple[str, str, str], ...] = (
    ('bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_max_ps'),
    ('src2dst_min_ps', 'src2dst_mean_ps', 'src2dst_max_ps'),
    ('dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_max_ps'),
    ('bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_max_piat_ms'),
    ('src2dst_min_piat_ms', 'src2dst_mean_piat_ms', 'src2dst_max_piat_ms'),
    ('dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_max_piat_ms'),
)


class FlowScaler:
    """log1p (optional) followed by z-score, with a torch-side inverse.

    Kept differentiable so the constrained gradient attack can hop between the
    standardised space it optimises in and the raw space the constraints live in.
    """

    def __init__(self, log1p: bool = True, clip: float = 10.0):
        self.log1p = log1p
        # ours: a handful of flag counters (cwr, ece, rare protocols) are almost
        # always zero, so their z-scores reach ~280 on the few non-zero flows and
        # would dominate the VAE reconstruction loss.  0.03% of values are affected.
        self.clip = clip
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.log_mask: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'FlowScaler':
        Xf = X.astype(np.float64)
        self.log_mask = (np.nanmin(Xf, axis=0) >= 0.0) if self.log1p else np.zeros(Xf.shape[1], bool)
        Z = self._fwd_np(Xf)
        self.mean = Z.mean(0)
        self.std = Z.std(0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def _fwd_np(self, X: np.ndarray) -> np.ndarray:
        Z = X.copy()
        if self.log1p:
            m = self.log_mask
            Z[:, m] = np.log1p(np.clip(Z[:, m], 0.0, None))
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._fwd_np(np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
        Z = (Z - self.mean) / self.std
        if self.clip:
            Z = np.clip(Z, -self.clip, self.clip)
        return Z.astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        X = Z.astype(np.float64) * self.std + self.mean
        if self.log1p:
            m = self.log_mask
            X[:, m] = np.expm1(np.clip(X[:, m], None, 50.0))
        return X

    # ---- torch views, used by the attacks ---------------------------------- #
    def to_torch(self, device) -> 'FlowScaler':
        self._t_mean = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        self._t_std = torch.as_tensor(self.std, dtype=torch.float32, device=device)
        self._t_logm = torch.as_tensor(self.log_mask, device=device)
        return self

    def t_inverse(self, Z: torch.Tensor) -> torch.Tensor:
        X = Z * self._t_std + self._t_mean
        if self.log1p:
            X = torch.where(self._t_logm, torch.expm1(X.clamp(max=50.0)), X)
        return X

    def t_forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.log1p:
            X = torch.where(self._t_logm, torch.log1p(X.clamp(min=0.0)), X)
        Z = (X - self._t_mean) / self._t_std
        return Z.clamp(-self.clip, self.clip) if self.clip else Z


@dataclass
class FlowData:
    """Everything the pipeline needs after preprocessing.

    All X arrays live in the standardised space; labels are *remapped* to the
    contiguous set of known classes (0..K-1) used by the DNN classifier.

    The constraint sets are carried per-instance rather than read from module
    globals, so a second dataset with a different schema only has to supply its
    own column names.
    """
    feat_names: List[str]
    scaler: FlowScaler
    known_names: List[str]
    zero_day_names: List[str]
    benign_idx: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test_id: np.ndarray
    y_test_id: np.ndarray
    X_test_ood: np.ndarray
    y_test_ood_orig: np.ndarray
    # realism constraints, by column name
    mutable_names: Tuple[str, ...] = MUTABLE_FEATURES
    triple_names: Tuple[Tuple[str, str, str], ...] = ORDERED_TRIPLES
    product_names: Tuple[Tuple[str, str, str], ...] = ()
    # optional: the fine-grained class each known/zero-day label came from
    known_supers: Optional[List[str]] = None
    zero_day_supers: Optional[List[str]] = None
    n_known_fine: Optional[int] = None

    @property
    def n_features(self) -> int:
        return len(self.feat_names)

    @property
    def n_classes(self) -> int:
        return len(self.known_names)

    def _idx(self, groups) -> List[Tuple[int, ...]]:
        pos = {f: i for i, f in enumerate(self.feat_names)}
        return [tuple(pos[c] for c in g) for g in groups if all(c in pos for c in g)]

    def mutable_mask(self) -> np.ndarray:
        return np.array([f in self.mutable_names for f in self.feat_names], dtype=bool)

    def ordered_triples_idx(self) -> List[Tuple[int, int, int]]:
        return self._idx(self.triple_names)

    def product_constraints_idx(self) -> List[Tuple[int, int, int]]:
        """(total, count, mean) triples that must satisfy total == count * mean."""
        return self._idx(self.product_names)

    def summary(self) -> str:
        head = (f"features={self.n_features}  classifier head={self.n_classes}-way"
                f"  zero-day classes={len(self.zero_day_names)}")
        if self.n_known_fine:
            head += f"  (trained on data of {self.n_known_fine} fine classes)"
        head += f"\nknown={self.known_names}"
        zs = sorted(set(self.zero_day_supers or []))
        head += f"\nzero-day supers={zs}" if zs else f"\nzero-day={self.zero_day_names}"
        return (
            head + "\n"
            f"train={len(self.X_train):,}  val={len(self.X_val):,}  "
            f"test-ID={len(self.X_test_id):,}  test-OOD={len(self.X_test_ood):,}\n"
            f"mutable features={int(self.mutable_mask().sum())}/{self.n_features}"
        )


# --------------------------------------------------------------------------- #
def load_raw(cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return Xtr, ytr, Xte, yte, feature names -- original 8-class labels."""
    import pandas as pd

    cache = cfg.path(cfg.data.cache_npz)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return (z['Xtr'], z['ytr'], z['Xte'], z['yte'], [str(c) for c in z['feats']])

    probe = pd.read_csv(cfg.path(cfg.data.train_csv), nrows=5)
    str_cols = list(probe.select_dtypes(exclude=[np.number]).columns)
    use_cols = [c for c in probe.columns if c not in str_cols]
    tr = pd.read_csv(cfg.path(cfg.data.train_csv), usecols=use_cols)
    te = pd.read_csv(cfg.path(cfg.data.test_csv), usecols=use_cols)
    lc = cfg.data.label_col
    feats = [c for c in tr.columns if c != lc]
    # zero-variance identifier columns were already zeroed out upstream
    feats = [c for c in feats if not (tr[c].nunique(dropna=False) <= 1 and te[c].nunique(dropna=False) <= 1)]
    Xtr = tr[feats].to_numpy(np.float32)
    Xte = te[feats].to_numpy(np.float32)
    ytr = tr[lc].to_numpy(np.int64)
    yte = te[lc].to_numpy(np.int64)
    np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        feats=np.array(feats, dtype=object))
    return Xtr, ytr, Xte, yte, feats


def build(cfg, zero_day: Optional[Sequence[str]] = None) -> FlowData:
    """Split into known / zero-day, fit the scaler on known training flows only."""
    Xtr, ytr, Xte, yte, feats = load_raw(cfg)
    names = cfg.data.class_names
    zd_names = list(zero_day if zero_day is not None else cfg.data.zero_day_classes)
    name2orig = {n: i for i, n in enumerate(names)}
    zd_ids = {name2orig[n] for n in zd_names}
    known_names = [n for n in names if n not in zd_names]
    remap = {name2orig[n]: i for i, n in enumerate(known_names)}

    tr_known = ~np.isin(ytr, list(zd_ids))
    te_known = ~np.isin(yte, list(zd_ids))

    scaler = FlowScaler(log1p=cfg.data.log1p, clip=cfg.data.clip).fit(Xtr[tr_known])

    Xk = scaler.transform(Xtr[tr_known])
    yk = np.array([remap[v] for v in ytr[tr_known]], dtype=np.int64)

    rng = np.random.default_rng(cfg.data.seed)
    perm = rng.permutation(len(Xk))
    n_val = int(round(cfg.data.val_frac * len(Xk)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    return FlowData(
        feat_names=feats,
        scaler=scaler,
        known_names=known_names,
        zero_day_names=zd_names,
        benign_idx=known_names.index(cfg.data.benign_class),
        X_train=Xk[tr_idx], y_train=yk[tr_idx],
        X_val=Xk[val_idx], y_val=yk[val_idx],
        X_test_id=scaler.transform(Xte[te_known]),
        y_test_id=np.array([remap[v] for v in yte[te_known]], dtype=np.int64),
        X_test_ood=scaler.transform(Xte[~te_known]),
        y_test_ood_orig=yte[~te_known],
    )


def loaders(X: np.ndarray, y: Optional[np.ndarray], batch_size: int,
            shuffle: bool = True, device: str = 'cpu'):
    from torch.utils.data import DataLoader, TensorDataset
    tx = torch.from_numpy(np.ascontiguousarray(X))
    ds = TensorDataset(tx) if y is None else TensorDataset(tx, torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(device == 'cuda'), drop_last=False)


# --------------------------------------------------------------------------- #
# Official CIC-IoT2023 release (39 features, 34 fine classes)
# --------------------------------------------------------------------------- #
def build_raw(cfg, zero_day: Optional[Sequence[str]] = None) -> FlowData:
    """Paper-faithful split: hold out whole *super* classes, but keep the fine
    labels for the classifier.

    Holding out Brute force / Spoofing / Recon / Web-based leaves exactly the
    20 fine classes the paper trains on.
    """
    from .data_cic_raw import (MUTABLE_FEATURES as RAW_MUTABLE,
                               ORDERED_TRIPLES as RAW_TRIPLES,
                               PRODUCT_CONSTRAINTS as RAW_PRODUCTS)

    z = np.load(cfg.path(cfg.data.raw_cache_npz), allow_pickle=True)
    Xtr, ytr, Xte, yte = z['Xtr'], z['ytr'], z['Xte'], z['yte']
    feats = [str(c) for c in z['feats']]
    fine = [str(c) for c in z['fine']]
    supers = [str(c) for c in z['supers']]

    zd_supers = set(zero_day if zero_day is not None else cfg.data.zero_day_classes)
    is_zd = np.array([s in zd_supers for s in supers], dtype=bool)
    if not is_zd.any():
        raise ValueError(f'no fine class maps to zero-day supers {sorted(zd_supers)}')

    known_ids = [i for i in range(len(fine)) if not is_zd[i]]
    zd_ids = [i for i in range(len(fine)) if is_zd[i]]

    gran = getattr(cfg.data, 'label_granularity', 'super')
    if gran == 'super':
        # 4-way head over the known super classes, trained on the data of all
        # 20 remaining fine classes
        head_names: List[str] = []
        for i in known_ids:
            if supers[i] not in head_names:
                head_names.append(supers[i])
        remap = {i: head_names.index(supers[i]) for i in known_ids}
        benign_label = cfg.data.benign_class
    elif gran == 'fine':
        head_names = [fine[i] for i in known_ids]
        remap = {orig: k for k, orig in enumerate(known_ids)}
        benign_label = None
    else:
        raise ValueError(f'unknown label_granularity: {gran}')

    tr_known = ~is_zd[ytr]
    te_known = ~is_zd[yte]

    scaler = FlowScaler(log1p=cfg.data.log1p, clip=cfg.data.clip).fit(Xtr[tr_known])
    Xk = scaler.transform(Xtr[tr_known])
    yk = np.array([remap[v] for v in ytr[tr_known]], dtype=np.int64)

    rng = np.random.default_rng(cfg.data.seed)
    perm = rng.permutation(len(Xk))
    n_val = int(round(cfg.data.val_frac * len(Xk)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    benign_fine = [i for i in known_ids if supers[i] == cfg.data.benign_class]
    if not benign_fine:
        raise ValueError(f'benign super class {cfg.data.benign_class!r} not among known')
    benign_idx = (head_names.index(benign_label) if benign_label is not None
                  else remap[benign_fine[0]])

    return FlowData(
        feat_names=feats,
        scaler=scaler,
        known_names=head_names,
        zero_day_names=[fine[i] for i in zd_ids],
        benign_idx=benign_idx,
        X_train=Xk[tr_idx], y_train=yk[tr_idx],
        X_val=Xk[val_idx], y_val=yk[val_idx],
        X_test_id=scaler.transform(Xte[te_known]),
        y_test_id=np.array([remap[v] for v in yte[te_known]], dtype=np.int64),
        X_test_ood=scaler.transform(Xte[~te_known]),
        y_test_ood_orig=yte[~te_known],
        mutable_names=RAW_MUTABLE,
        triple_names=RAW_TRIPLES,
        product_names=RAW_PRODUCTS,
        known_supers=[supers[i] for i in known_ids],
        zero_day_supers=[supers[i] for i in zd_ids],
        n_known_fine=len(known_ids),
    )


def build_dataset(cfg, zero_day: Optional[Sequence[str]] = None) -> FlowData:
    """Dispatch on cfg.data.dataset."""
    if cfg.data.dataset == 'cic_raw':
        return build_raw(cfg, zero_day)
    if cfg.data.dataset == 'xgnid':
        return build(cfg, zero_day)
    raise ValueError(f'unknown dataset: {cfg.data.dataset}')
