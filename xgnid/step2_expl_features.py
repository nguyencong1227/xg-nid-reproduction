"""Component 2 - Explainable Feature Extractor (XG-NID Sec. 3.1.2, Algorithm 1).

Conventional NFStream statistics describe a flow in isolation, so they cannot
see a brute-force attempt spread over hundreds of short flows.  Algorithm 1
fixes that by keeping a sliding window per *destination* and re-deriving the
Table 1 statistics at every time step.

Two window modes are supported:

``count`` (default, ``window=350``)
    The window holds the last N flows seen for that key.  This is what the
    released GNN4ID tool does, and what produced the numbers in the paper.
``time`` (``window="60s"``)
    A true time window, which is what the prose in Sec. 3.1.2 literally
    describes.  Kept available because the two disagree, and the time-based
    variant is the one that transfers across captures with different flow
    rates.

Both keys from the reference implementation are produced: ``*_Destination``
(the Table 1 definition, "received at the destination") and
``*_SourceDestination`` (per source-destination pair, which is what actually
isolates a single attacker).
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd

from .config import DNS_PORTS, EXPIRATION_IDS, HTTP_PORTS, PROTOCOLS, ROLLING_WINDOW, VULNERABLE_PORTS

# (output suffix, source column, aggregation) applied under each grouping key.
_ROLLING_SPEC: list[tuple[str, str, str]] = [
    ("Rolling_UDP_Requests", "_is_udp", "sum"),
    ("Rolling_TCP_Requests", "_is_tcp", "sum"),
    ("Rolling_ICMP_Requests", "_is_icmp", "sum"),
    ("Rolling_ACK_Packets", "bidirectional_ack_packets", "sum"),
    ("Rolling_FIN_Packets", "bidirectional_fin_packets", "sum"),
    ("Rolling_rst_Packets", "bidirectional_rst_packets", "sum"),
    ("Rolling_psh_Packets", "bidirectional_psh_packets", "sum"),
    ("Rolling_SYN_Packets", "bidirectional_syn_packets", "sum"),
    ("Rolling_http_port", "_is_http_port", "sum"),
    ("Rolling_DNS_request", "_is_dns_dst", "sum"),
    ("Rolling_DNS_request2", "_is_dns_src", "sum"),
    ("Rolling_vulnerable_port", "_is_vuln_port", "sum"),
    ("Rolling_Duration", "bidirectional_duration_ms", "mean"),
    ("Rolling_packets", "src2dst_packets", "sum"),
    ("Rolling_bipackets", "bidirectional_packets", "sum"),
]

_HELPER_COLUMNS = [
    "_is_udp", "_is_tcp", "_is_icmp", "_is_http_port",
    "_is_dns_dst", "_is_dns_src", "_is_vuln_port",
    "_dst_ip_key", "_src_dst_key",
]


def _rolling(
    df: pd.DataFrame, key: str, col: str, agg: str, window, time_mode: bool
) -> pd.Series:
    """Grouped rolling aggregation that stays aligned with ``df.index``."""
    grouped = df.groupby(key, sort=False, observed=True)[col]
    if time_mode:
        # rolling on a time offset needs the timestamp as the index
        s = df[col].copy()
        s.index = df["_ts"]
        out = s.groupby(df[key].to_numpy(), sort=False).rolling(window, min_periods=1)
        out = getattr(out, agg)()
        out = out.reset_index(level=0, drop=True)
        out.index = df.index
        return out
    out = getattr(grouped.rolling(window, min_periods=1), agg)()
    return out.reset_index(level=0, drop=True).reindex(df.index)


def _rolling_nunique(df: pd.DataFrame, key: str, col: str, window, time_mode: bool) -> pd.Series:
    counter: Callable = lambda x: len(np.unique(x))
    if time_mode:
        s = df[col].astype("float64")
        s.index = df["_ts"]
        out = (s.groupby(df[key].to_numpy(), sort=False)
                .rolling(window, min_periods=1).apply(counter, raw=True))
        out = out.reset_index(level=0, drop=True)
        out.index = df.index
        return out
    out = (df.groupby(key, sort=False, observed=True)[col]
             .rolling(window, min_periods=1).apply(counter, raw=True))
    return out.reset_index(level=0, drop=True).reindex(df.index)


def add_explainable_features(
    df: pd.DataFrame,
    window=ROLLING_WINDOW,
    http_ports: list[int] = HTTP_PORTS,
    dns_ports: list[int] = DNS_PORTS,
    vulnerable_ports: list[int] = VULNERABLE_PORTS,
    one_hot: bool = True,
    keys: tuple[str, ...] = ("Destination", "SourceDestination"),
) -> pd.DataFrame:
    """Append the Table 1 temporal features to an extracted-flow frame."""
    if "bidirectional_first_seen_ms" not in df.columns:
        raise KeyError("frame must contain 'bidirectional_first_seen_ms' to be time-ordered")

    time_mode = isinstance(window, str)
    df = df.sort_values("bidirectional_first_seen_ms", kind="mergesort").reset_index(drop=True)
    if time_mode:
        df["_ts"] = pd.to_datetime(df["bidirectional_first_seen_ms"], unit="ms")

    # Grouping keys.  Plain string concatenation is enough and avoids the
    # LabelEncoder round-trip in the reference implementation.
    df["_dst_ip_key"] = df["dst_ip"].astype(str)
    df["_src_dst_key"] = df["src_ip"].astype(str) + "-" + df["dst_ip"].astype(str)
    key_column = {"Destination": "_dst_ip_key", "SourceDestination": "_src_dst_key"}

    # Boolean indicators, as float so the rolling kernels stay on the fast path.
    df["_is_udp"] = (df["protocol"] == 17).astype("float64")
    df["_is_tcp"] = (df["protocol"] == 6).astype("float64")
    df["_is_icmp"] = (df["protocol"] == 1).astype("float64")
    df["_is_http_port"] = df["dst_port"].isin(http_ports).astype("float64")
    df["_is_dns_dst"] = df["dst_port"].isin(dns_ports).astype("float64")
    df["_is_dns_src"] = df["src_port"].isin(dns_ports).astype("float64")
    df["_is_vuln_port"] = df["dst_port"].isin(vulnerable_ports).astype("float64")

    # Spread of packet sizes across the four directional extremes.
    df["packet_size_variation"] = df[
        ["src2dst_min_ps", "src2dst_max_ps", "dst2src_min_ps", "dst2src_max_ps"]
    ].std(axis=1)

    new_cols: dict[str, pd.Series] = {}
    for suffix in keys:
        key = key_column[suffix]
        for name, col, agg in _ROLLING_SPEC:
            if suffix == "Destination" and name in ("Rolling_vulnerable_port",):
                # Table 1 defines this one per destination only; the reference
                # code computes it per pair.  Emit both, they are cheap.
                pass
            new_cols[f"{name}_{suffix}"] = _rolling(df, key, col, agg, window, time_mode)

    # Table 1: "unique source ports used to communicate with a specific destination"
    if "SourceDestination" in keys:
        new_cols["Unique_Ports_In_SourceDestination"] = _rolling_nunique(
            df, "_src_dst_key", "dst_port", window, time_mode
        )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    if one_hot:
        df = one_hot_categoricals(df)

    drop = [c for c in _HELPER_COLUMNS + ["_ts"] if c in df.columns]
    return df.drop(columns=drop)


def one_hot_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot ``expiration_id`` and ``protocol`` over fixed vocabularies.

    Fixed categories matter: without them a capture that happens to contain no
    IGMP traffic would produce a narrower feature matrix than its siblings.
    """
    out = df.copy()
    out["expiration_id"] = pd.Categorical(out["expiration_id"], categories=EXPIRATION_IDS)
    out["protocol"] = pd.Categorical(out["protocol"], categories=PROTOCOLS)
    return pd.get_dummies(
        out, prefix=["Exp", "proto"], columns=["expiration_id", "protocol"], dtype=int
    )


def process_csv(path: str, out_path: str | None = None, **kwargs) -> str:
    """Read an extracted-flow CSV, add temporal features, write it back out."""
    df = pd.read_csv(path)
    df = add_explainable_features(df, **kwargs)
    out_path = out_path or path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
