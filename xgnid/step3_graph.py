"""Component 3 - Graph Generator (XG-NID Sec. 3.1.3, Algorithm 2).

Each flow becomes one heterogeneous graph:

* one ``flow`` node carrying the flow-level statistics (Eq. 2),
* one ``packet`` node per packet, whose features are the 1500 payload bytes
  (Sec. 3.1.1) and optionally the eight TCP flags,
* ``contain`` edges flow -> packet with [direction, ip_size, transport_size,
  payload_size] (Eq. 3), and
* ``link`` edges packet_i -> packet_{i+1} with the inter-arrival time (Eq. 4).

``ToUndirected`` is applied last: without the reverse ``contain`` edge the flow
node would never receive a message from its packets.

Two deliberate departures from the released GNN4ID code, both switchable:

* ``link`` edge attributes are shaped ``[n-1, 1]`` rather than ``[n-1]``.  The
  1-D version makes ``GATConv(edge_dim=-1)`` infer the edge-feature width from
  the *number of packets*, which silently changes the model between graphs.
* Flow features are standardised and payload bytes are divided by 255
  (``paper_defaults=True`` restores the raw values).  Raw NFStream columns span
  microseconds to megabytes, and the first GATConv sees them before any
  normalisation layer.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData, InMemoryDataset

from .config import PACKET_FEATURE_COLUMNS, PAYLOAD_DIM

# Identity, timestamp and leakage columns never enter the flow node vector.
# The bidirectional_* counters are dropped because Component 2 already turned
# them into rolling statistics; keeping both double-counts them.
DROP_COLUMNS = [
    "id", "src_ip", "src_port", "dst_ip", "dst_port", "ip_version",
    "src_mac", "src_oui", "dst_mac", "dst_oui", "vlan_id", "tunnel_id",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "bidirectional_duration_ms", "bidirectional_packets", "bidirectional_bytes",
    "bidirectional_syn_packets", "bidirectional_cwr_packets",
    "bidirectional_ece_packets", "bidirectional_urg_packets",
    "bidirectional_ack_packets", "bidirectional_psh_packets",
    "bidirectional_rst_packets", "bidirectional_fin_packets",
    # bookkeeping added by earlier steps
    "Label", "class_name", "subclass", "expiration_id", "protocol",
]

FLAG_COLUMNS = ["udps.syn", "udps.cwr", "udps.ece", "udps.urg",
                "udps.ack", "udps.psh", "udps.rst", "udps.fin"]


@dataclass
class GraphSpec:
    """Everything needed to rebuild an identical feature space at inference."""
    flow_columns: list[str]
    include_packet_flags: bool
    include_payload: bool
    payload_dim: int
    payload_scale: str          # "unit" (bytes/255) or "raw"
    standardize: bool
    mean: list[float] | None = None
    std: list[float] | None = None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @staticmethod
    def load(path: str) -> "GraphSpec":
        with open(path) as fh:
            return GraphSpec(**json.load(fh))

    @property
    def flow_dim(self) -> int:
        return len(self.flow_columns)

    @property
    def packet_dim(self) -> int:
        return (8 if self.include_packet_flags else 0) + (
            self.payload_dim if self.include_payload else 0)


# Columns contributed by Component 2; excluding them ablates the paper's
# contribution #5 (the temporal/explainable feature set).
TEMPORAL_PREFIXES = ("Rolling_", "Unique_Ports_", "packet_size_variation")


def select_flow_columns(df: pd.DataFrame,
                        exclude_prefixes: tuple[str, ...] = ()) -> list[str]:
    """Numeric flow-level columns, in a stable order."""
    drop = set(DROP_COLUMNS) | set(PACKET_FEATURE_COLUMNS)
    cols = [c for c in df.columns
            if c not in drop and not c.startswith(tuple(exclude_prefixes))]
    numeric = df[cols].select_dtypes(include=["number", "bool"]).columns
    return [c for c in cols if c in set(numeric)]


def fit_spec(
    df: pd.DataFrame,
    include_packet_flags: bool = False,
    include_payload: bool = True,
    payload_dim: int = PAYLOAD_DIM,
    paper_defaults: bool = False,
    exclude_prefixes: tuple[str, ...] = (),
) -> GraphSpec:
    """Derive the feature space (and normalisation statistics) from train data."""
    if not include_payload and not include_packet_flags:
        raise ValueError(
            "packet nodes would have zero features: enable include_packet_flags "
            "when include_payload is False")
    cols = select_flow_columns(df, exclude_prefixes)
    spec = GraphSpec(
        flow_columns=cols,
        include_packet_flags=include_packet_flags,
        include_payload=include_payload,
        payload_dim=payload_dim,
        payload_scale="raw" if paper_defaults else "unit",
        standardize=not paper_defaults,
    )
    if spec.standardize:
        x = df[cols].to_numpy(dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        spec.mean, spec.std = mean.tolist(), std.tolist()
    return spec


def _parse_list(value) -> list:
    """Packet lists are JSON (this pipeline) or a Python repr (legacy GNN4ID)."""
    if isinstance(value, list):
        return value
    s = str(value)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return [t.strip() for t in s.strip("[]").replace("'", "").split(",")]


def _payload_to_bytes(hex_str: str, dim: int) -> np.ndarray:
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError:
        raw = b"\x00"
    arr = np.frombuffer(raw[:dim], dtype=np.uint8)
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size), constant_values=0)
    return arr


def row_to_graph(row: pd.Series, spec: GraphSpec, label: int | None = None,
                 to_undirected: bool = True) -> HeteroData:
    """Algorithm 2: build the heterogeneous graph for a single flow."""
    payloads = _parse_list(row["udps.payload_data"])
    n = len(payloads)

    # --- flow node -----------------------------------------------------
    flow_vec = np.asarray([row[c] for c in spec.flow_columns], dtype=np.float64)
    flow_vec = np.nan_to_num(flow_vec, nan=0.0, posinf=0.0, neginf=0.0)
    if spec.standardize and spec.mean is not None:
        flow_vec = (flow_vec - np.asarray(spec.mean)) / np.asarray(spec.std)

    # --- packet nodes --------------------------------------------------
    blocks = []
    if spec.include_packet_flags:
        flags = np.asarray(
            [[float(v) for v in _parse_list(row[c])[:n]] for c in FLAG_COLUMNS],
            dtype=np.float32,
        ).T                                          # [n, 8]
        blocks.append(flags)
    if spec.include_payload:
        payload = np.stack([_payload_to_bytes(p, spec.payload_dim) for p in payloads])
        payload = payload.astype(np.float32)
        if spec.payload_scale == "unit":
            payload = payload / 255.0
        blocks.append(payload)
    packet_x = np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0), np.float32)

    # --- edges ---------------------------------------------------------
    contain_attr = np.asarray(
        [[float(v) for v in _parse_list(row[c])[:n]]
         for c in ("udps.packet_direction", "udps.ip_size",
                   "udps.transport_size", "udps.payload_size")],
        dtype=np.float32,
    ).T                                              # [n, 4]
    delta = np.asarray([float(v) for v in _parse_list(row["udps.delta_time"])[:n]],
                       dtype=np.float32)

    data = HeteroData()
    data["flow"].x = torch.tensor(flow_vec, dtype=torch.float32).view(1, -1)
    data["packet"].x = torch.tensor(packet_x, dtype=torch.float32)
    data["flow", "contain", "packet"].edge_index = torch.stack(
        [torch.zeros(n, dtype=torch.long), torch.arange(n, dtype=torch.long)]
    )
    data["flow", "contain", "packet"].edge_attr = torch.tensor(contain_attr)
    data["packet", "link", "packet"].edge_index = torch.stack(
        [torch.arange(0, n - 1, dtype=torch.long), torch.arange(1, n, dtype=torch.long)]
    )
    data["packet", "link", "packet"].edge_attr = torch.tensor(delta[1:]).view(-1, 1)
    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)

    return T.ToUndirected()(data) if to_undirected else data


def dataframe_to_graphs(df: pd.DataFrame, spec: GraphSpec,
                        label_column: str | None = "Label",
                        progress: bool = True) -> list[HeteroData]:
    from tqdm import tqdm

    it = df.iterrows()
    if progress:
        it = tqdm(it, total=len(df), desc="graphs", unit="flow")
    return [
        row_to_graph(row, spec, None if label_column is None else row[label_column])
        for _, row in it
    ]


class XGNIDGraphDataset(InMemoryDataset):
    """Materialises a flow CSV into one collated PyG file.

    ``spec`` must be fitted on the training split and reused for the test split,
    otherwise the two get different feature spaces.
    """

    def __init__(self, root: str, csv_path: str | None = None,
                 spec: GraphSpec | None = None, name: str = "data",
                 max_rows: int | None = None, transform=None):
        self._csv_path = csv_path
        self._spec = spec
        self._name = name
        self._max_rows = max_rows
        super().__init__(root, transform=transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self._name}.pt"]

    def process(self):
        if self._csv_path is None or self._spec is None:
            raise RuntimeError(
                f"{self.processed_paths[0]} is missing; pass csv_path and spec to build it")
        df = pd.read_csv(self._csv_path)
        if self._max_rows is not None:
            df = df.iloc[: self._max_rows]
        label_column = "Label" if "Label" in df.columns else None
        self.save(dataframe_to_graphs(df, self._spec, label_column), self.processed_paths[0])
