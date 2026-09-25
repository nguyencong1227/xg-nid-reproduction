"""Loader for the *official* CIC-IoT2023 release (39 features, 34 classes).

This is the dataset the paper actually uses. Rows come from
`CSV/<AttackClass>/<name>.pcap.csv`; the fine-grained class is the folder name,
and there is no `Label` column in this release.

Mapping the 34 fine classes onto the 8 super classes reproduces the paper's
split exactly: holding out Brute force + Spoofing + Recon + Web-based leaves
**20 classes** for the classifier, which is what Sec. IV states.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# 34 fine classes -> 8 super classes
# --------------------------------------------------------------------------- #
SUPER_CLASS: Dict[str, str] = {
    'Benign_Final': 'Benign',
    # DDoS (12)
    'DDoS-ACK_Fragmentation': 'DDos', 'DDoS-HTTP_Flood': 'DDos',
    'DDoS-ICMP_Flood': 'DDos', 'DDoS-ICMP_Fragmentation': 'DDos',
    'DDoS-PSHACK_FLOOD': 'DDos', 'DDoS-RSTFINFLOOD': 'DDos',
    'DDoS-SYN_Flood': 'DDos', 'DDoS-SlowLoris': 'DDos',
    'DDoS-SynonymousIP_Flood': 'DDos', 'DDoS-TCP_Flood': 'DDos',
    'DDoS-UDP_Flood': 'DDos', 'DDoS-UDP_Fragmentation': 'DDos',
    # DoS (4)
    'DoS-HTTP_Flood': 'Dos', 'DoS-SYN_Flood': 'Dos',
    'DoS-TCP_Flood': 'Dos', 'DoS-UDP_Flood': 'Dos',
    # Mirai (3)
    'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    # Recon (5)
    'Recon-HostDiscovery': 'Recon', 'Recon-OSScan': 'Recon',
    'Recon-PingSweep': 'Recon', 'Recon-PortScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    # Spoofing (2)
    'DNS_Spoofing': 'Spoofing', 'MITM-ArpSpoofing': 'Spoofing',
    # Web-based (6)
    'BrowserHijacking': 'WebBased', 'CommandInjection': 'WebBased',
    'SqlInjection': 'WebBased', 'Uploading_Attack': 'WebBased',
    'XSS': 'WebBased', 'Backdoor_Malware': 'WebBased',
    # Brute force (1)
    'DictionaryBruteForce': 'BruteForce',
}

FEATURES: Tuple[str, ...] = (
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
    'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Variance',
)

# Statistical features an attacker can move (Sec. IV). Everything else --
# protocol indicators, flag rates, flag counts, packet count -- is symbolic and
# frozen: perturbing it would break the flow's semantics.
MUTABLE_FEATURES: Tuple[str, ...] = (
    'Header_Length', 'Time_To_Live', 'Rate',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Variance',
)

ORDERED_TRIPLES: Tuple[Tuple[str, str, str], ...] = (('Min', 'AVG', 'Max'),)

# `Tot sum` is the summed packet size over `Number` packets, so it has to track
# AVG exactly. Enforced as (total, count, mean) after every attack step.
PRODUCT_CONSTRAINTS: Tuple[Tuple[str, str, str], ...] = (('Tot sum', 'Number', 'AVG'),)


def class_dirs(root: str) -> List[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def _count_rows(path: str, bufsize: int = 1 << 22) -> int:
    """Data rows in a CSV (buffered newline count, minus the header)."""
    n = 0
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            n += b.count(b'\n')
    return max(n - 1, 0)


def load(root: str, per_class: int = 45000, seed: int = 42,
         classes: Optional[Sequence[str]] = None,
         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Read the per-attack CSVs, sampling up to `per_class` rows per *fine* class.

    The sample is uniform over the whole class: a first pass counts rows per
    file, then each file contributes in proportion to its size.  Reading only
    the first few files instead would bias towards whichever pcap happened to
    sort first.

    Returns X, y_fine, fine_names, super_names -- where y_fine indexes
    `fine_names` and `super_names[i]` is the super class of `fine_names[i]`.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    names = list(classes) if classes else class_dirs(root)
    Xs, ys, fine_names = [], [], []

    for cname in names:
        if cname not in SUPER_CLASS:
            if verbose:
                print(f'  skip unknown entry {cname}')
            continue
        files = sorted(glob.glob(os.path.join(root, cname, '*.csv')))
        if not files:
            continue
        sizes = [_count_rows(fp) for fp in files]
        total = sum(sizes)
        if total == 0:
            continue
        keep = min(per_class, total)

        parts = []
        for fp, n in zip(files, sizes):
            if n == 0:
                continue
            take = int(round(keep * n / total))
            df = pd.read_csv(fp, usecols=list(FEATURES))
            if take < len(df):
                df = df.iloc[rng.choice(len(df), max(take, 0), replace=False)]
            if len(df):
                parts.append(df)
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        if len(d) > keep:                       # rounding can overshoot by a few
            d = d.iloc[rng.choice(len(d), keep, replace=False)]

        X = np.nan_to_num(d[list(FEATURES)].to_numpy(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
        Xs.append(X)
        ys.append(np.full(len(X), len(fine_names), dtype=np.int64))
        fine_names.append(cname)
        if verbose:
            print(f'  [{len(fine_names):2d}] {cname:26s} {len(X):>7,} / {total:>10,} '
                  f'-> {SUPER_CLASS[cname]}', flush=True)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    return X, y, fine_names, [SUPER_CLASS[n] for n in fine_names]


def global_first_occurrence(root: str, classes: Optional[Sequence[str]] = None,
                            verbose: bool = True):
    """Mark, per class, which rows are the *first* global occurrence of their
    feature vector.

    CIC-IoT2023 is 53.5% exact duplicates, and the duplication crosses class
    boundaries: 62% of DoS-TCP_Flood rows share a 39-dim vector with a
    DDoS-TCP_Flood row. Those pairs are contradictory labels that no model can
    separate. Collapsing them globally (first occurrence wins, in class order)
    removes the contradiction and takes the corpus to ~21.8M rows -- which is
    what the paper's "over 20 million flow-level samples" appears to describe.
    """
    import pandas as pd

    names = [n for n in (list(classes) if classes else class_dirs(root))
             if n in SUPER_CLASS]
    per_class_hashes, files_of = {}, {}
    for cname in names:
        files = sorted(glob.glob(os.path.join(root, cname, '*.csv')))
        hs = []
        for fp in files:
            df = pd.read_csv(fp, usecols=list(FEATURES))
            hs.append(pd.util.hash_pandas_object(df, index=False).to_numpy())
        per_class_hashes[cname] = np.concatenate(hs) if hs else np.zeros(0, np.uint64)
        files_of[cname] = files
        if verbose:
            print(f'  hashed {cname:26s} {len(per_class_hashes[cname]):>10,}', flush=True)

    order = [c for c in names if len(per_class_hashes[c])]
    allh = np.concatenate([per_class_hashes[c] for c in order])
    _, first_idx = np.unique(allh, return_index=True)
    keep = np.zeros(len(allh), dtype=bool)
    keep[first_idx] = True

    masks, off = {}, 0
    for c in order:
        n = len(per_class_hashes[c])
        masks[c] = keep[off:off + n]
        off += n
    if verbose:
        print(f'  global dedup: {len(allh):,} -> {int(keep.sum()):,} '
              f'({100 * (1 - keep.sum() / len(allh)):.1f}% removed)', flush=True)
    return masks, files_of


def load_dedup(root: str, per_class: int = 45000, seed: int = 42,
               classes: Optional[Sequence[str]] = None, verbose: bool = True):
    """`load`, but drawing from the globally deduplicated corpus."""
    import pandas as pd

    masks, files_of = global_first_occurrence(root, classes, verbose)
    rng = np.random.default_rng(seed)
    Xs, ys, fine_names = [], [], []
    for cname, files in files_of.items():
        m = masks.get(cname)
        if m is None or not m.any():
            continue
        parts, off = [], 0
        for fp in files:
            df = pd.read_csv(fp, usecols=list(FEATURES))
            parts.append(df[m[off:off + len(df)]])
            off += len(df)
        d = pd.concat(parts, ignore_index=True)
        total = len(d)
        if total > per_class:
            d = d.iloc[rng.choice(total, per_class, replace=False)]
        X = np.nan_to_num(d[list(FEATURES)].to_numpy(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
        Xs.append(X)
        ys.append(np.full(len(X), len(fine_names), dtype=np.int64))
        fine_names.append(cname)
        if verbose:
            print(f'  [{len(fine_names):2d}] {cname:26s} {len(X):>7,} / {total:>9,} '
                  f'unique -> {SUPER_CLASS[cname]}', flush=True)
    return (np.concatenate(Xs), np.concatenate(ys), fine_names,
            [SUPER_CLASS[n] for n in fine_names])


def build_cache(root: str, out_npz: str, per_class: int = 40000,
                seed: int = 42, test_frac: float = 0.2, dedup: bool = False) -> str:
    """One-off: sample, split train/test stratified by fine class, save an npz."""
    loader = load_dedup if dedup else load
    X, y, fine_names, super_names = loader(root, per_class, seed)
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in range(len(fine_names)):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        cut = int(round(test_frac * len(idx)))
        te_idx.append(idx[:cut])
        tr_idx.append(idx[cut:])
    tr = np.concatenate(tr_idx)
    te = np.concatenate(te_idx)
    rng.shuffle(tr)
    rng.shuffle(te)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        Xtr=X[tr], ytr=y[tr], Xte=X[te], yte=y[te],
        feats=np.array(FEATURES, dtype=object),
        fine=np.array(fine_names, dtype=object),
        supers=np.array(super_names, dtype=object))
    print(f'cache: {out_npz}  train {len(tr):,}  test {len(te):,}  '
          f'classes {len(fine_names)}  features {X.shape[1]}')
    return out_npz
