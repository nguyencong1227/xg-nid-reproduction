"""Dataset preprocessing (XG-NID Sec. 3.2.1, Tables 3 and 4).

Two jobs:

1. **Attacker-MAC filtering.**  A CIC-IoT2023 attack capture also contains the
   victim's ordinary background chatter.  A flow only counts as an attack if
   one of its MACs belongs to an attacker device (Table 3); a flow only counts
   as benign if *neither* MAC does.
2. **Rebalancing.**  20% is held out for test, capped at 4,000 flows per class,
   and the remainder is under/oversampled to 20,000 flows per class, keeping
   each attack subclass proportionally represented.

The per-file quota pass reads only the two MAC columns first, so the full
46M-flow dataset can be planned without ever holding a whole class in memory.
"""
from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ATTACKER_MACS, LABEL_DICT, class_from_pcap_name


@dataclass
class FileInfo:
    path: str
    cls: str
    subclass: str
    n_rows: int = 0


def _mac_mask(df: pd.DataFrame, cls: str) -> pd.Series:
    src = df["src_mac"].astype(str).str.lower()
    dst = df["dst_mac"].astype(str).str.lower()
    touches_attacker = src.isin(ATTACKER_MACS) | dst.isin(ATTACKER_MACS)
    return ~touches_attacker if cls == "Benign" else touches_attacker


def scan(feature_dir: str, pattern: str = "*.csv",
         benign_from_attack_captures: bool = False) -> list[FileInfo]:
    """Discover feature CSVs and count how many flows survive MAC filtering.

    ``benign_from_attack_captures`` additionally harvests the flows in an attack
    capture that touch *no* attacker MAC and treats them as Benign.  The paper
    sources Benign only from the dedicated benign captures, so this is off by
    default; it exists so a small subset of captures can still be exercised
    end-to-end with a background class present.
    """
    infos: list[FileInfo] = []
    for path in sorted(glob.glob(os.path.join(feature_dir, pattern))):
        subclass = os.path.splitext(os.path.basename(path))[0]
        cls = class_from_pcap_name(subclass)
        if cls is None:
            print(f"[step2b] skipping {subclass}: no class mapping")
            continue
        macs = pd.read_csv(path, usecols=["src_mac", "dst_mac"])
        mask = _mac_mask(macs, cls)
        n = int(mask.sum())
        infos.append(FileInfo(path=path, cls=cls, subclass=subclass, n_rows=n))
        print(f"[step2b] {subclass:28s} {cls:11s} {n:>9,} flows after MAC filter")
        if benign_from_attack_captures and cls != "Benign":
            n_bg = int((~mask).sum())
            if n_bg:
                infos.append(FileInfo(path=path, cls="Benign",
                                      subclass=subclass + "-background", n_rows=n_bg))
                print(f"[step2b] {subclass + '-background':28s} {'Benign':11s} "
                      f"{n_bg:>9,} flows after MAC filter")
    return infos


def _sample_file(info: FileInfo, quota: int, rng: np.random.Generator,
                 chunksize: int = 50_000) -> pd.DataFrame:
    """Reservoir-sample up to ``quota`` MAC-filtered rows out of one file."""
    keep: pd.DataFrame | None = None
    for chunk in pd.read_csv(info.path, chunksize=chunksize):
        chunk = chunk[_mac_mask(chunk, info.cls)]
        if chunk.empty:
            continue
        keep = chunk if keep is None else pd.concat([keep, chunk], ignore_index=True)
        # Trim eagerly so memory stays bounded at ~2x quota.
        if len(keep) > 2 * quota and quota > 0:
            keep = keep.sample(n=quota, random_state=int(rng.integers(1 << 31)))
    if keep is None:
        return pd.DataFrame()
    if quota and len(keep) > quota:
        keep = keep.sample(n=quota, random_state=int(rng.integers(1 << 31)))
    keep = keep.copy()
    keep["subclass"] = info.subclass
    return keep


def _resample_to(df: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
    """Under- or oversample ``df`` to exactly ``target`` rows (paper Table 4)."""
    if len(df) == 0 or target <= 0:
        return df.iloc[:0]
    seed = int(rng.integers(1 << 31))
    if len(df) >= target:
        return df.sample(n=target, random_state=seed)
    reps = target // len(df)
    out = pd.concat([df] * reps, ignore_index=True)
    remainder = target - len(out)
    if remainder:
        out = pd.concat([out, df.sample(n=remainder, replace=True, random_state=seed)],
                        ignore_index=True)
    return out


def build_dataset(
    feature_dir: str,
    out_dir: str,
    train_per_class: int = 20_000,
    test_cap: int = 4_000,
    test_frac: float = 0.2,
    seed: int = 42,
    pattern: str = "*.csv",
    benign_from_attack_captures: bool = False,
) -> tuple[str, str]:
    """Assemble balanced ``train.csv`` / ``test.csv`` with an integer ``Label``."""
    rng = np.random.default_rng(seed)
    infos = scan(feature_dir, pattern, benign_from_attack_captures)
    if not infos:
        raise RuntimeError(f"no usable feature CSVs found in {feature_dir}")

    by_class: dict[str, list[FileInfo]] = {}
    for info in infos:
        by_class.setdefault(info.cls, []).append(info)

    train_parts, test_parts = [], []
    for cls, files in sorted(by_class.items()):
        total = sum(f.n_rows for f in files)
        if total == 0:
            print(f"[step2b] {cls}: no flows survived filtering, skipped")
            continue
        need = train_per_class + test_cap

        frames = []
        for f in files:
            # Proportional subclass representation, but never more than exists.
            quota = min(f.n_rows, max(1, math.ceil(need * f.n_rows / total)))
            part = _sample_file(f, quota, rng)
            if not part.empty:
                frames.append(part)
        pool = pd.concat(frames, ignore_index=True)
        pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        n_test = min(test_cap, int(round(test_frac * total)))
        n_test = min(n_test, max(1, len(pool) - 1))
        test = pool.iloc[:n_test]
        rest = pool.iloc[n_test:]
        train = _resample_to(rest, train_per_class, rng)

        for part, bucket in ((train, train_parts), (test, test_parts)):
            part = part.copy()
            part["Label"] = LABEL_DICT[cls]
            part["class_name"] = cls
            bucket.append(part)
        print(f"[step2b] {cls:11s} pool={len(pool):>7,}  test={len(test):>6,}  train={len(train):>6,}")

    os.makedirs(out_dir, exist_ok=True)
    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[step2b] wrote {len(train_df):,} train / {len(test_df):,} test rows to {out_dir}")
    return train_path, test_path
