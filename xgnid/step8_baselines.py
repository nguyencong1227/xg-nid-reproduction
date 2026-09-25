"""Baseline comparisons (XG-NID Sec. 4.1.1, Tables 5 and 6).

The paper's argument is that either modality alone leaves a blind spot, so the
baselines are deliberately single-modality:

* **flow-only** - the flow node vector, i.e. the classical NIDS feature set.
  Expected to be weak on the payload-specific classes (web-based, brute force).
* **packet-only** - the concatenated payload bytes of the flow's packets, the
  Payload-Byte representation of Farrukh et al. (2022).  Expected to be weak on
  the volumetric classes.

Both are evaluated on the same split as the HGNN, and each result also carries
the subset metric the paper reports: ``payload_specific_f1`` for flow models
and ``flow_specific_f1`` for packet models.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .config import LABEL_DICT, PAYLOAD_SPECIFIC_CLASSES
from .step3_graph import GraphSpec, _parse_list, _payload_to_bytes, select_flow_columns

PAYLOAD_SPECIFIC_LABELS = {LABEL_DICT[c] for c in PAYLOAD_SPECIFIC_CLASSES}


def flow_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x = df[columns].to_numpy(dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def payload_matrix(df: pd.DataFrame, n_packets: int = 20,
                   payload_dim: int = 1500) -> np.ndarray:
    """Flatten the first ``n_packets`` payloads into one Payload-Byte vector."""
    out = np.zeros((len(df), n_packets * payload_dim), dtype=np.float32)
    for r, (_, row) in enumerate(df.iterrows()):
        for i, hexstr in enumerate(_parse_list(row["udps.payload_data"])[:n_packets]):
            out[r, i * payload_dim: (i + 1) * payload_dim] = _payload_to_bytes(
                hexstr, payload_dim)
    return out


def _models(seed: int) -> dict:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "MLP": MLPClassifier(random_state=seed, max_iter=300),
        "KNN": KNeighborsClassifier(),
        "DNN": MLPClassifier(hidden_layer_sizes=(256, 128, 64), random_state=seed, max_iter=300),
    }


def _score(y_true, y_pred, subset_labels: set[int]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    res = {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }
    # The paper's "Payload-Specific" / "Flow-Specific" columns score only the
    # classes in the subset.  `labels=` is required: without it the macro
    # average silently includes the out-of-subset classes at zero support.
    labels = sorted(subset_labels)
    res["subset_f1"] = (
        float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        if labels else None
    )
    res["subset_support"] = int(np.isin(y_true, labels).sum())
    return res


def run_baselines(
    dataset_dir: str,
    out_dir: str,
    modalities: tuple[str, ...] = ("flow", "packet"),
    max_train: int | None = 20_000,
    n_packets: int = 20,
    payload_dim: int = 1500,
    seed: int = 42,
) -> dict:
    """Train every baseline on each modality and write ``baselines.json``."""
    os.makedirs(out_dir, exist_ok=True)
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    test = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
    if max_train is not None and len(train) > max_train:
        train = train.sample(n=max_train, random_state=seed)
    y_tr = train["Label"].to_numpy()
    y_te = test["Label"].to_numpy()

    present = set(int(v) for v in np.concatenate([y_tr, y_te]))
    payload_specific = PAYLOAD_SPECIFIC_LABELS & present
    flow_specific = present - PAYLOAD_SPECIFIC_LABELS

    results: dict[str, dict] = {}
    for modality in modalities:
        if modality == "flow":
            cols = select_flow_columns(train)
            x_tr, x_te = flow_matrix(train, cols), flow_matrix(test, cols)
            subset = payload_specific      # Table 5: "Payload-Specific" column
            subset_name = "payload_specific_f1"
        else:
            x_tr = payload_matrix(train, n_packets, payload_dim)
            x_te = payload_matrix(test, n_packets, payload_dim)
            subset = flow_specific         # Table 6: "Flow-Specific" column
            subset_name = "flow_specific_f1"

        scaler = StandardScaler().fit(x_tr)
        x_tr_s, x_te_s = scaler.transform(x_tr), scaler.transform(x_te)
        print(f"[baselines] {modality}: train {x_tr.shape}, test {x_te.shape}")

        for name, model in _models(seed).items():
            t0 = time.time()
            # Trees are scale-invariant; everything else needs the scaler.
            xa, xb = ((x_tr, x_te) if name in ("RandomForest", "AdaBoost")
                      else (x_tr_s, x_te_s))
            try:
                model.fit(xa, y_tr)
                pred = model.predict(xb)
            except Exception as exc:                      # e.g. KNN OOM on payloads
                print(f"[baselines] {modality}/{name} failed: {exc}")
                continue
            score = _score(y_te, pred, subset)
            np.save(os.path.join(out_dir, f"pred_{modality}_{name}.npy"), pred)
            score[subset_name] = score.pop("subset_f1")
            score["seconds"] = round(time.time() - t0, 1)
            results[f"{modality}/{name}"] = score
            print(f"[baselines] {modality:7s} {name:19s} "
                  f"P={score['precision']:.3f} R={score['recall']:.3f} "
                  f"F1={score['f1']:.3f} {subset_name}="
                  f"{score[subset_name] if score[subset_name] is None else round(score[subset_name], 3)}"
                  f"  ({score['seconds']}s)")

    np.save(os.path.join(out_dir, "y_test.npy"), y_te)
    path = os.path.join(out_dir, "baselines.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[baselines] wrote {path}")
    return results
