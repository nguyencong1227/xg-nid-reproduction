"""Component 1 - Flow and Feature Generator (XG-NID Sec. 3.1.1).

Turns a raw pcap into one CSV row per flow, where each row carries

* the 76 NFStream flow-level statistics, and
* 14 packet-level lists (payload hex, delta time, direction, layer sizes and
  the eight TCP flags), one entry per packet in the flow.

Flows are cut at 20 packets and expire after 120 s idle, exactly as the paper
specifies, so that inference can happen in near real time.

Unlike the released GNN4ID script this module

* truncates each payload to ``PAYLOAD_DIM`` bytes at capture time (the graph
  builder discards everything past byte 1500 anyway, so keeping more only
  inflates the CSV by ~20x on full-MTU flows), and
* serialises the per-packet lists as JSON instead of relying on Python's
  ``repr``, which makes the parser in Component 3 unambiguous.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Iterable

from nfstream import NFPlugin, NFStreamer

from .config import (
    ACCOUNTING_MODE,
    ACTIVE_TIMEOUT_S,
    IDLE_TIMEOUT_S,
    MAX_PACKETS_PER_FLOW,
    PACKET_FEATURE_COLUMNS,
    PAYLOAD_DIM,
    class_from_pcap_name,
)

_FLAGS = ("syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin")


class PacketFeatures(NFPlugin):
    """Collects the per-packet modality and forces expiry at ``limit`` packets."""

    def _payload_hex(self, packet) -> str:
        if packet.payload_size > 0:
            raw = packet.ip_packet[-packet.payload_size :][:PAYLOAD_DIM]
            return raw.hex()
        return "00"

    def on_init(self, packet, flow):
        udps = flow.udps
        udps.payload_data = [self._payload_hex(packet)]
        udps.delta_time = [packet.delta_time]
        udps.packet_direction = [packet.direction]
        udps.ip_size = [packet.ip_size]
        udps.transport_size = [packet.transport_size]
        udps.payload_size = [packet.payload_size]
        for flag in _FLAGS:
            setattr(udps, flag, [getattr(packet, flag)])
        if self.limit == 1:
            flow.expiration_id = -1

    def on_update(self, packet, flow):
        udps = flow.udps
        udps.payload_data.append(self._payload_hex(packet))
        udps.delta_time.append(packet.delta_time)
        udps.packet_direction.append(packet.direction)
        udps.ip_size.append(packet.ip_size)
        udps.transport_size.append(packet.transport_size)
        udps.payload_size.append(packet.payload_size)
        for flag in _FLAGS:
            getattr(udps, flag).append(getattr(packet, flag))
        if self.limit == flow.bidirectional_packets:
            flow.expiration_id = -1  # -1 forces expiration


def _flow_columns(flow) -> list[str]:
    """Base NFStream columns, i.e. every attribute that is not a udps list."""
    return [k for k in flow.keys() if not k.startswith("udps.")]


def extract_pcap(
    pcap_path: str,
    out_dir: str,
    limit: int = MAX_PACKETS_PER_FLOW,
    idle_timeout: int = IDLE_TIMEOUT_S,
    active_timeout: int = ACTIVE_TIMEOUT_S,
    label: str | None = None,
    flush_every: int = 5_000,
    progress_every: int = 50_000,
) -> str:
    """Extract one pcap to ``<out_dir>/<pcap stem>.csv`` and return the path."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(pcap_path)
    for suffix in (".pcap", ".pcapng"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out_path = os.path.join(out_dir, stem + ".csv")

    if label is None:
        label = class_from_pcap_name(pcap_path)

    streamer = NFStreamer(
        source=pcap_path,
        accounting_mode=ACCOUNTING_MODE,
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
        statistical_analysis=True,
        n_dissections=0,
        udps=PacketFeatures(limit=limit),
    )

    started = time.time()
    header: list[str] | None = None
    base_cols: list[str] = []
    rows: list[list] = []
    n = 0

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        for flow in streamer:
            if header is None:
                base_cols = _flow_columns(flow)
                header = base_cols + PACKET_FEATURE_COLUMNS
                if label is not None:
                    header.append("class_name")
                writer.writerow(header)

            row = [getattr(flow, c) for c in base_cols]
            row += [json.dumps(getattr(flow.udps, c.split(".", 1)[1]))
                    for c in PACKET_FEATURE_COLUMNS]
            if label is not None:
                row.append(label)
            rows.append(row)
            n += 1

            if len(rows) >= flush_every:
                writer.writerows(rows)
                rows.clear()
            if progress_every and n % progress_every == 0:
                print(f"  {stem}: {n:,} flows ({time.time() - started:.0f}s)",
                      file=sys.stderr, flush=True)
        if rows:
            writer.writerows(rows)

    if header is None:  # pcap produced no flow at all
        open(out_path, "w").close()

    print(f"[step1] {stem}: {n:,} flows -> {out_path} "
          f"({time.time() - started:.1f}s)", file=sys.stderr, flush=True)
    return out_path


def extract_many(pcaps: Iterable[str], out_dir: str, **kwargs) -> list[str]:
    return [extract_pcap(p, out_dir, **kwargs) for p in pcaps]
