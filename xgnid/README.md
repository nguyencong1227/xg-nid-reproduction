# XG-NID — reimplementation

A from-scratch implementation of

> Y. A. Farrukh, S. Wali, I. Khan, N. D. Bastian.
> *XG-NID: Dual-modality network intrusion detection using a heterogeneous graph
> neural network and large language model.*
> Expert Systems with Applications 287:128089, 2025. ([arXiv:2408.16021](https://arxiv.org/abs/2408.16021))

starting from raw pcap files. Everything runs through one CLI.

## Component map

| # | Paper section | Module | What it does |
|---|---|---|---|
| 1 | 3.1.1 Flow and Feature Generator | `step1_flow_extract.py` | pcap → flow CSV; 76 NFStream flow features + 14 per-packet lists, flows cut at 20 packets, 120 s idle timeout |
| 2 | 3.1.2 Explainable Feature Extractor (Alg. 1, Table 1) | `step2_expl_features.py` | rolling per-destination temporal features |
| – | 3.2.1 Dataset preprocessing (Tables 3, 4) | `step2b_preprocess.py` | attacker-MAC filtering, class rebalancing, train/test split |
| 3 | 3.1.3 Graph Generator (Alg. 2, Eqs. 2–4) | `step3_graph.py` | flow CSV → heterogeneous PyG graphs |
| 4 | 3.1.4 HGNN Model (Eqs. 5–9) | `step4_model.py` | 2× GATConv + BN + LeakyReLU → global mean pool → MLP → log-softmax |
| – | 4.1 Performance analysis | `step5_train.py` | training loop, metrics, confusion matrix |
| – | 4.1.1 Baselines (Tables 5, 6) | `step8_baselines.py` | flow-only and packet-only classical models |
| – | real-time inference (3.1.1) | `step9_infer.py` | pcap → per-flow prediction with the trained model |
| – | Tables 5, 6 rendering | `step10_report.py` | markdown comparison table |
| 5 | 3.1.5 Integrated Gradient Explainer (Eq. 10) | `step6_ig_explain.py` | per-prediction attributions over nodes and edges |
| 6 | 3.1.6 Generative Explainer (Alg. 3) | `step7_llm_explain.py` | builds `Q_flow` / `Q_payload`, optional Llama 3 generation |

## Graph structure (Fig. 2)

```
        flow node  (85 features: 46 NFStream stats + 32 temporal + 7 one-hot)
             |
        contain edge  [direction, ip_size, transport_size, payload_size]
             |
   packet_1 -- link -- packet_2 -- link -- ... -- packet_n   (n <= 20)
   (1500 payload bytes each)      [delta_time]
```

`ToUndirected` is applied at the end, which adds the reverse `contain` edge.
Without it the flow node receives no messages and its embedding is the input
projection alone.

## Environment

```bash
conda env create -f environment.yml     # or see the header of that file
conda activate xg-nid
```

Created on this host as `/home/congnc/anaconda3/envs/xg-nid` rather than in the
first `envs_dirs` entry (`/storage/congnc/.conda/envs`): `/storage` holds the
graph datasets and has less headroom. It is still activatable by name because
`/home/congnc/anaconda3/envs` is also on `envs_dirs`.

Pinned set: Python 3.11, torch 2.6.0+cu124, torch-geometric 2.8.0.post1,
nfstream 6.6.0, pandas 3.0.1, numpy 1.26.4, scikit-learn 1.2.2 (see
`requirements-xgnid.txt` for why each version is pinned), plus transformers
5.17 / accelerate 1.15 for Component 6.

## Ablation suite

```bash
python experiments.py --list                    # 12 experiments, 6 graph builds
python experiments.py --run baseline,seed1
python experiments.py --all
python experiments.py --report                  # rebuild work/xgnid/EXPERIMENTS.md
```

Graph building is keyed on the flags that change the graphs, so model-only
variants reuse an existing graph directory; finished stages are skipped on
re-run. The suite includes repeated seeds of the reference configuration
deliberately: on the two-capture demo split the seed spread (macro-F1
0.926-0.960) is *wider* than the gap to the `no_edge_attr` and `flow_only`
ablations, so those ablations are not separable here.

## Pipeline

```bash
python -m xgnid.cli extract   --pcaps 'data/cic_pcap/*.pcap' --out work/xgnid/01_flows
python -m xgnid.cli features  --in  work/xgnid/01_flows      --out work/xgnid/02_features
python -m xgnid.cli dataset   --in  work/xgnid/02_features   --out work/xgnid/03_dataset
python -m xgnid.cli graphs    --dataset work/xgnid/03_dataset --out work/xgnid/04_graphs
python -m xgnid.cli train     --graphs  work/xgnid/04_graphs  --out work/xgnid/05_run
python -m xgnid.cli baselines --dataset work/xgnid/03_dataset --out work/xgnid/06_baselines
python -m xgnid.cli report    --run work/xgnid/05_run --baselines work/xgnid/06_baselines
python -m xgnid.cli explain   --run work/xgnid/05_run --graphs work/xgnid/04_graphs --num 6
python -m xgnid.cli infer     --pcap data/cic_pcap/XSS.pcap --run work/xgnid/05_run
```

`explain --llm meta-llama/Meta-Llama-3-8B-Instruct` runs the prompts through the
model the paper used; without `--llm` the prompts are written out unanswered.

For the full CIC-IoT2023 run, drop `--train-per-class` / `--test-cap` so the
defaults (20,000 / 4,000 per class, Table 4) apply, and remove
`--benign-from-background` once the real benign captures are present.

## Deviations from the released GNN4ID code

Each is a switch, not a hard-coded choice.

1. **`link` edge attributes are shaped `[n-1, 1]` instead of `[n-1]`.**
   Cosmetic only. PyG's `GATConv` reshapes a 1-D `edge_attr` to `[-1, 1]`
   before the lazy `lin_edge` (`gat_conv.py` lines 361-362), so both forms infer
   `in_channels=1` and give identical outputs — verified on batches mixing
   7-packet and 20-packet flows. The released 1-D form is not a bug.
2. **Packet ranking for `Q_payload` uses `mean(|IG|)` per packet, not the
   literal Algorithm 3 lines 19-21.** Read literally, each packet's 1500-dim
   attribution vector is L2-normalised and *then* averaged; since every row
   then has unit norm, the mean is dominated by sign cancellation and collapses
   to a ~1e-3 band, whereas `mean(|IG|)` spans many orders of magnitude. The
   two rankings agree on the top packet in 2 of 3 sampled flows and diverge
   below that. The normalised vectors are still used to pick `top_byte_offsets`.
3. **Flow features are standardised and payload bytes divided by 255.**
   NFStream columns span microseconds to megabytes and the first GATConv sees
   them before any normalisation layer. `--paper-defaults` restores raw values.
   The scaler is fitted on train only and stored in `spec.json`.
4. **Packet lists are serialised as JSON.** The released CSVs rely on Python's
   `repr` and are parsed back with `.strip('][').split(', ')`, which breaks on
   any payload string containing a comma. The reader accepts both formats.
5. **Payloads are truncated to 1500 bytes at capture time.** The graph builder
   discards the rest anyway; keeping it inflates the intermediate CSV several-fold.
6. **Both rolling keys are emitted.** Table 1 defines the temporal features per
   destination; the released code also computes them per source-destination
   pair. Both are kept (`_Destination`, `_SourceDestination`) — 32 extra columns.
7. **`--window` accepts a time offset.** Sec. 3.1.2 describes a *time* window
   but the released code uses a 350-*row* window. The row window is the default,
   so results stay comparable; `--window 60s` gives the literal reading.

## Known gaps against the paper

### Eq. 6 is not what GATConv computes

Eq. 6 aggregates edge features as a separate summed message term,
`h_i = sigma( sum_j a_ij W h_j + sum_e a_ij W_e h_e )`. PyG's `GATConv` uses
`edge_attr` **only** inside the attention logits
(`alpha_edge = (lin_edge(edge_attr) * att_edge).sum(-1)`); its message is
`message(x_j, alpha) = x_j * alpha`, with no edge term. So neither this
implementation nor the released GNN4ID code realises Eq. 6 literally —
edge features reweight neighbours rather than contributing their own message.

A measurable consequence: the four `contain` edge features
(direction, ip_size, transport_size, payload_size) have **exactly zero**
influence in the `flow -> packet` direction. Each packet node has precisely one
incoming `contain` edge, so the attention softmax over a single logit is 1.0
regardless of the edge features. Multiplying those attributes by 100 changes
the output by 0.0 (verified), and Integrated Gradients reports
`edge:flow__contain__packet = 0.0` on every explained flow. The features do
reach the model through the reverse edge added by `ToUndirected`
(`edge:packet__rev_contain__flow` is non-zero), where the flow node has n
incoming edges to attend over. `link` edge attributes have small but non-zero
influence (each packet node has 1-2 incoming link edges).

### Not implemented

* **Component 6 has never been executed.** `step7_llm_explain.py` builds both
  prompts and contains the generation path, but `transformers`, `accelerate`
  and `bitsandbytes` are not installed in this environment and the Llama 3-8B
  weights are gated, so only the prompt-construction half is exercised. The
  generation code is untested.
* **Literature baselines in Tables 5 and 6.** Only the six generic models
  (Random Forest, Logistic Regression, AdaBoost, MLP, KNN, DNN) are
  implemented. Missing: Conv-AE (Khan and Kim, 2020), IIDS (Narayan et al.,
  2023), CNN-LSTM (Gueriani et al., 2024), AEIDS (Pratomo et al., 2018),
  Packet2Vec (Goodman et al., 2020), EsPADA (Vidal et al., 2020).
* **Sec. 4.1.2 / Fig. 3 dual-modality state of the art.** Premkumar et al.
  (2023), Kiflay et al. (2024) and TR-IDS (Min et al., 2018) are not
  reimplemented, so the comparison the paper's headline claim rests on is absent.
* **Sec. 4.2.1 / Fig. 5 explainability baseline.** The Shapley-value plus
  instruction-tuning-template comparison against Khediri et al. (2024) is not
  implemented, so the claim that rolling-window features explain flow attacks
  better than plain flow attributes is untested here.
* **Training hyperparameters.** The paper does not state epochs, learning rate,
  batch size or optimiser for the HGNN, so the values in `TrainConfig`
  (AdamW, lr 1e-3, 30 epochs, batch 64, hidden 64, 1 attention head) are this
  implementation's choices, not the paper's.

### Smaller discrepancies

* `Rolling_FIN_Sum` and `Rolling_fin_Sum` appear as two separate rows in
  Table 1 with identical descriptions. Only one FIN counter is implemented per
  grouping key.
* The paper reports 76 flow features into the flow node; after dropping
  identifiers/timestamps and adding the temporal set, this implementation puts
  85 numeric features there. The exact drop list of the released code is
  reproduced in `step3_graph.DROP_COLUMNS`.
* The LLM stage is zero-shot against whatever model id is passed. Outputs are
  not scored — the paper evaluates them qualitatively too.

## Demo run on two captures

`scripts/run_xgnid_demo.sh` runs the whole pipeline on
`data/cic_pcap/{DictionaryBruteForce,XSS}.pcap`.

| Stage | Result |
|---|---|
| Component 1 | 15,313 flows (11,043 + 4,270), ~4 s |
| Component 2 | +39 columns (31 rolling + `packet_size_variation` + 7 one-hot) |
| Preprocessing | 12,000 train / 1,281 test flows over 3 classes |
| Component 3 | 13,281 heterogeneous graphs, 925 MB |
| Component 4 | 0.43 M parameters, ~6 s/epoch on one GPU |
| Test | accuracy 0.987, macro-F1 0.947, weighted-F1 0.987 |

Per class: Benign F1 1.00 (800), BruteForce F1 0.99 (437), WebBased F1 0.86 (44).

Re-scoring the raw `XSS.pcap` through `infer` (no MAC filter, all 4,270 flows)
recovers the attacker flows at recall 1.00 / precision 0.88 against the Table 3
MAC ground truth.

**These numbers do not reproduce the paper's headline result, and are not meant
to.** Two captures give three classes instead of eight, both attacks are
payload-specific, and the Benign class here is background traffic harvested
from inside the same two captures — so the model never sees DDoS, DoS, Mirai,
Recon or Spoofing, and the baseline tables cannot show the packet-only
weakness on volumetric attacks that Table 6 is built to show. The
`infer` check also reuses flows that appear in training, so it measures the
pipeline wiring, not generalisation. Reproducing the reported 97% F1 needs the
full 33-capture CIC-IoT2023 download.
