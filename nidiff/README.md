# NI-Diff — reproduction on CIC-IoT2023

Re-implementation of **NI-Diff: Zero-Day and Adversarial Network Intrusion
Detection with Diffusion Models** (Zhang, De Lucia, Swami, Ashdown, Bastian,
Restuccia — MILCOM 2025), paper at [`paper/ni-diff.pdf`](../paper/ni-diff.pdf).

**Scope: CIC-IoT2023 only.** ACI-IoT-2023 is deliberately out of scope — it is
the smaller and easier of the paper's two datasets, and the interesting
disagreements with the paper all live on the CIC side. So the tables produced
here correspond to the paper's Table III, **V**, **VII** and the CIC columns of
**VIII**, plus the denoising timestep sweep of **Figure 2**.

## The idea in one paragraph

A frozen multi-class DNN classifies flows. In parallel, a VAE maps the flow into
a probabilistic latent space and a diffusion model resamples that latent with a
**single** diffusion-denoising step. The resampled latent is decoded and fed
back through the same classifier. For in-distribution flows both softmax
outputs agree; for zero-day or adversarial flows the round trip pulls the sample
back onto the training distribution and the two outputs diverge. The KL
divergence between them is the detection score (Algorithm 1).

Diffusing the *latent* rather than the raw flow is the point: adding Gaussian
noise directly to symbolic features (ACK flag counts, packet counts) destroys
their semantics, which is the limitation the paper raises against AdvPurRec.

## Layout

| file | contents |
|---|---|
| `config.py` | every hyper-parameter, sourced from the paper's Training Recipe and Tables I/II; `smoke()` gives a few-minute budget |
| `data.py` | 82 flow features, `FlowScaler`, the zero-day split, and the realism constraints used by the attacks |
| `models.py` | `NIDiffBase` / `NIDiffLarge` (Table I), `FlowVAE`, `DenoiseUNet1d` (Table II), `EMA` |
| `diffusion.py` | DDPM (Eqs. 3–5), one-step `denoise_once`, full `ddpm_sample_from` for the ablation |
| `train.py` | the three training loops, including the Eq. 2 VAE loss |
| `attacks.py` | constrained targeted PGD and the GAN-based attack |
| `detectors.py` | `NIDiffDetector` plus GEN, Energy, GradNorm, MANDA, DeepRP, NIDS-DA |
| `metrics.py` | TPR/FPR/precision/F1/AUROC and the table builders |
| `pipeline.py` | end-to-end run, writes CSVs + `report.md` |

## Running

```bash
python nidiff/scripts/run_nidiff.py --smoke     # sanity check, a few minutes
python nidiff/scripts/run_nidiff.py             # full paper recipe
python nidiff/scripts/run_nidiff.py --arch base --diff-epochs 300 --no-ablation
```

Checkpoints and results land in `nidiff/work/nidiff/` (`--out-dir` to change).
Re-running reuses existing checkpoints; pass `--force` to retrain.

## Two datasets

`--dataset cic_raw` (default) is the **official CIC-IoT2023 release** the paper
uses: 39 features, 34 fine classes, 46.7M rows, sampled at 45k per fine class to
1.17M — the paper's "1.2 million samples". Holding out Brute force / Spoofing /
Recon / Web-based leaves the data of 20 fine classes.

`--dataset xgnid` is the XG-NID nfstream re-extraction of the same PCAPs: 82
features, 8 super classes only, 160k rows. Kept because it carries packet
payload and because the first reproduction ran on it.

### Classifier head: `label_granularity`

The paper says "the classifier leverages the remaining 20 classes for training"
and reports 99.42% accuracy. Those two cannot both describe a 20-way head:

| head | accuracy (this repro) |
|---|---|
| 20-way fine (`fine`, default) | 83.7% |
| 4-way super (`super`) | 87.9% |
| 3-way, DDoS+DoS merged | **99.91%** |

`fine` is the default: it is the literal reading, and the paper's own wording
backs it. For ACI it says "the classifier **only uses 9 classes of data** for
training" — 12 classes with 3 held out, and no super-class grouping is defined
for that dataset, so that head has to be 9-way fine. The CIC sentence
("leverages the remaining **20 classes**") is the identical construction.

`super` was run too, because a coarser head lands nearer their accuracy figure.
But neither reading reaches 99.42%, so the accuracy number cannot arbitrate the
ambiguity and the wording decides. Both configurations fail the zero-day half
(AUROC 41.96 fine, 33.41 super), so nothing downstream hinges on the choice.

### Why neither head passes 88%

The entire gap is one cell of the confusion matrix:

```
rows=true, cols=pred (%)   Benign    DDos     Dos   Mirai
Benign                     100.00    0.00    0.00    0.00
DDos                         0.01   92.17    7.79    0.04
Dos                          0.01   35.87   64.09    0.04
Mirai                        0.00    0.34    0.03   99.63
```

Benign and Mirai are essentially perfect; DoS recall is 64%, with 36% of it
absorbed into DDoS. `DDoS-TCP_Flood` and `DoS-TCP_Flood` are the same behaviour
at a different number of attacking hosts, which a flow feature vector does not
carry. No rebalancing closes this — balancing at super-class level instead of
fine-class level still lands near 89%. **The paper's 99.42% is only reachable if
DDoS and DoS are not separated**, which the 8-super-class taxonomy it cites does
separate. Reported here as an unresolved discrepancy rather than papered over.

This does not undermine the detection results: NI-Diff reads the classifier's
softmax, and a classifier that confuses DDoS with DoS still produces sharp
softmax on in-distribution flows. It does mean the `Acc` column below reads 88%,
not 99.42%.

It also explains the attack success rates. Benign is perfectly separated *and*
only 5% of training rows, and both attacks target Benign — a target region that
is both tight and rare is hard to reach, which is why ASR lands at 77% (gradient)
and 69% (GAN) rather than the paper's 89% and 99.97%.

## Results on the official release (`nidiff/work/nidiff_raw/`)

Full recipe on 1.17M flows, 39 features, 4-way head. Accuracy 87.92%, ASR 73.81%
(gradient) / 50.18% (GAN). VAE MSE 0.00364, diffusion ε-MSE 0.1349, latent 1×9.
`comparison.md` in the run directory has every cell side by side with the paper.

AUROC, ours vs the paper:

| detector | zero-day | | gradient | | GAN | |
|---|---|---|---|---|---|---|
| | **ours** | paper | **ours** | paper | **ours** | paper |
| GEN | 24.34 | 82.05 | 63.11 | 99.89 | 97.06 | 99.33 |
| Energy | 21.49 | 84.01 | 51.79 | 99.90 | 89.58 | 99.42 |
| GradNorm | 24.88 | 76.50 | 39.44 | 99.71 | 63.57 | 99.39 |
| MANDA | 93.18 | 26.17 | 99.64 | 82.33 | 99.92 | 97.71 |
| DeepRP | 87.37 | 58.41 | 94.49 | 62.97 | 81.42 | 97.51 |
| NIDS-DA | **95.57** | 83.69 | 100.00 | 98.08 | 99.94 | 99.48 |
| NI-Diff | 33.41 | 95.70 | 98.21 | 99.46 | **99.94** | 99.94 |

**The adversarial half reproduces almost exactly.** NI-Diff GAN AUROC 99.94 vs
99.94 and TPR 99.98 vs 99.98; gradient AUROC 98.21 vs 99.46, TPR 91.90 vs 94.31.
The mechanism works as described.

**The zero-day half inverts.** NI-Diff drops to 33.41% AUROC — below random — and
so do all three softmax-based OOD baselines (GEN 24.34, Energy 21.49, GradNorm
24.88, against the paper's 76–84). Four independent detectors failing the same
way is not noise.

### Why zero-day detection inverts here

The KLD score distributions say it directly:

| group | n | median KLD | p90 |
|---|---|---|---|
| ID, correctly classified | 151,619 | 5.1e-07 | 4.6e-03 |
| ID, misclassified | 20,824 | 5.0e-04 | 4.2e-02 |
| zero-day | 62,031 | **7.0e-11** | 7.6e-01 |

Most zero-day flows round-trip through the VAE and diffusion **without moving the
classifier's prediction at all** — their median KLD is four orders of magnitude
*below* in-distribution flows. The distribution is bimodal: a minority do produce
a large KLD (p90 0.76), which is why the mean is high, but the median sits under
almost every ID flow. Flipping the sign of the score would give 66.59% AUROC, so
the signal exists and points the wrong way.

The mechanism assumes OOD data reconstructs poorly and therefore changes the
prediction. `nidiff/scripts/diagnose_zeroday.py` shows the first half holds and the
second does not:

| | median recon MSE | AUROC of recon MSE alone |
|---|---|---|
| ID test | 0.00053 | |
| zero-day | 0.02832 (**53× higher**) | **94.89%** |

The VAE reconstructs zero-day flows 53× worse than in-distribution ones, and
that reconstruction error *by itself* is a 94.89% AUROC detector — on par with
NIDS-DA's 95.57%. **The novelty signal is present and strong inside the model;
it is destroyed by reading it out through the softmax.** Despite a 53× worse
reconstruction, the classifier returns the same verdict for x and x′, so the KLD
collapses.

The cause is the **class partition**, not the VAE. 93.1% of zero-day flows are
classified as **Benign** with a mean max-softmax of 0.9931:

| | median MSP | share MSP > 0.99 | median entropy |
|---|---|---|---|
| ID test | 1.0000 | 61.5% | 2.57e-04 |
| zero-day | 1.0000 | **89.2%** | **9.51e-09** |

Zero-day flows are ~27,000× *more* confident than in-distribution ones by median
entropy. The known set is Benign plus three high-volume **flood** classes
(DDoS/DoS/Mirai); the four held-out super classes are all low-rate and stealthy
(port scans, DNS spoofing, SQL injection, dictionary brute force). With a 4-way
head whose only non-flood option is Benign, they are all absorbed into Benign —
confidently. `f(x)` is Benign@0.993 and the round trip also returns Benign@~0.99,
so the KLD collapses no matter how bad the reconstruction was.

Every softmax-confidence detector inverts together (MSP AUROC 22.04, entropy
19.72, GEN 24.34, Energy 21.49, GradNorm 24.88, NI-Diff 33.41), while every
feature-space detector still works (NIDS-DA 95.57, MANDA 93.18, DeepRP 87.37).

Comparing the two feature representations pins down what flips the sign, and it
is **not** the OOD side:

| | 82 features (XG-NID) | 39 features (official) |
|---|---|---|
| OOD classified Benign | 55.4% | 93.1% |
| OOD with MSP > 0.99 | 85.1% | 89.2% |
| **ID with MSP > 0.99** | **99.8%** | **61.5%** |
| MSP AUROC | 75.87 | 22.04 |

OOD confidence is nearly the same in both (85.1 vs 89.2). What changes is
**in-distribution** confidence: 99.8% against 61.5%. On 82 features the classifier
is more confident on ID than on OOD, so the ordering is right; on 39 features it
is *less* confident on ID, so every confidence-based score runs backwards.

The chain is therefore: 39 features cannot separate DDoS from DoS → accuracy falls
to 87.9% (against 99.96%) → in-distribution softmax becomes uncertain → ID ranks
below OOD → inversion. The representation breaks the *in-distribution* problem;
it does not make the novelty harder to see.

Three hypotheses were tested and refuted along the way, each cheaply:

* **Perceptual loss (λ₂) preserving the classifier verdict.** Dropping it
  entirely moves the perceptual quantity only ~9% (0.0445 vs 0.0407 at epoch 80),
  so it cannot account for a 33%→90% swing.
* **The VAE absorbing OOD onto the known manifold.** It does the opposite —
  reconstruction is 53× worse on OOD.
* **Reconstruction error hiding in features the classifier ignores.** The
  correlation is *positive* (Spearman 0.366, p=0.022): the error lands in
  features the classifier does use.

This predicts that the paper's literal "20 classes" reading may be *necessary*
for NI-Diff's zero-day half to work — a 20-way head gives an OOD flow somewhere
to be uncertain, instead of one confident Benign bucket. Accuracy drops to ~83%
but the softmax stops saturating. That run is in progress
(`--label-granularity fine`, `nidiff/work/nidiff_fine/`).

Two secondary contributors, both measured and both too small to be the cause:

* **`clip=10.0` erases part of the signal before the VAE.** Zero-day rows are
  clipped 17× more often than ID rows (11.78% vs 0.70%), concentrated in the
  protocol indicators (DNS 3.82% vs 0.18%, ARP/LLC/IPv 3.04% vs 0.11%). Real
  loss, but reconstruction error still reaches 94.89% despite it.
* **Misclassified ID flows** have 1000× the median KLD of correctly classified
  ones, yet excluding them moves AUROC only from 33.41% to 35.33%. That test was
  too narrow to clear the DDoS/DoS ambiguity, though: dropping the *wrong*
  predictions leaves the merely *uncertain* ones, and it is overall ID confidence
  (61.5% above MSP 0.99, against 99.8% on 82 features) that decides the sign.

The read-out formula is not the problem either: all six combinations of KLD
direction × {sample z, use μ} land in a 30.83–36.47% band, still inverted.

Consistent with all of this, the detectors that work on zero-day here are exactly
the ones reading **feature**-space distance or reconstruction error — NIDS-DA
(95.57), MANDA (93.18), DeepRP (87.37) — while every **softmax**-based detector
fails (GEN 24.34, Energy 21.49, GradNorm 24.88, NI-Diff 33.41).

The open question is therefore narrow: does the paper's perceptual network differ
from ours? Ours is the one choice that mechanistically produces the observed
failure, and the paper does not specify it. Retraining the VAE and diffusion with
λ₂ = 0 (≈7h, classifier reused) is the decisive test and has not been run.

## Results on the XG-NID re-extraction (`nidiff/work/nidiff/`)

Full recipe, `nidiff/work/nidiff/` (classifier 50 ep, VAE 100 ep, diffusion 1000 ep, PGD
100 steps, GAN 200 ep; VAE reconstruction MSE 0.00496, diffusion ε-MSE 0.1055,
latent 1×20). Threshold at 5% nominal FPR on the ID validation split.

| | Acc | ASR (Grad) | ASR (GAN) |
|---|---|---|---|
| ours | 99.96% | 84.17% | 92.88% |
| paper (CICIoT) | 99.42% | 89.29% | 99.97% |

TPR per scenario, ours vs the paper's Table V:

| detector | zero-day | | gradient | | GAN | |
|---|---|---|---|---|---|---|
| | **ours** | paper | **ours** | paper | **ours** | paper |
| GEN | 40.04% | 38.38% | 98.00% | 99.96% | 99.97% | 99.97% |
| Energy | 36.86% | 39.64% | 96.19% | 99.82% | 99.90% | 99.97% |
| GradNorm | 9.28% | 30.11% | 86.49% | 99.86% | 99.24% | 99.94% |
| MANDA | 69.02% | 11.77% | 93.35% | 56.77% | 99.98% | 99.95% |
| DeepRP | 51.79% | 16.96% | 1.05% | 3.96% | 0.52% | 94.97% |
| NIDS-DA | 59.37% | 13.38% | 85.59% | 32.73% | 71.76% | 94.74% |
| **NI-Diff** | **70.35%** | 51.18% | **99.96%** | 94.31% | 99.97% | 99.98% |

What reproduces cleanly:

* **The central claim holds.** Across the 12 comparable cells (3 scenarios ×
  TPR/precision/F1/AUROC) NI-Diff ranks 1st in 6, 2nd in 2 and 3rd in 4 — never
  below 3rd, and it is the only detector that is never weak on either threat.
  It takes best zero-day TPR (70.35%), precision (89.16%) and F1 (78.65%), and
  best gradient TPR (99.96%), F1 (96.09%) and AUROC (99.85%). In five of the six
  cells where it places 2nd or 3rd the gap to the leader is under 0.6 points, in
  cells where five detectors sit within a point of each other. The one real loss
  is zero-day AUROC, where MANDA leads by 3.6 points.
* **OOD detectors fail on nothing but OOD.** GEN/Energy/GradNorm land 96–100% TPR
  on adversarial flows but 9–40% on zero-day — the same lopsidedness the paper
  reports, with GEN and Energy landing within ~3 points of their published
  zero-day numbers.
* **GradNorm can be worse than random.** 30.59% zero-day AUROC here; the paper
  reports sub-50% AUROC cells for GradNorm on ACI for the same reason.
* **One-step denoising is the right operating point** (`fig2_timesteps.png`).
  With the threshold fixed at t=1, FPR climbs 5.1% → 100% and TPR converges to
  100% as the timestep grows: the detector degenerates into flagging everything,
  which is the effect Fig. 2 reports.

Where we differ from the paper, and why:

* **MANDA is a far stronger baseline here** (69.02% zero-day TPR, 94.39% AUROC vs
  the paper's 11.77% / 26.17%) and NIDS-DA likewise. With 4 known super classes
  instead of 20 fine classes, a held-out super class sits much further from the
  known manifolds, so kNN-distance and reconstruction-error detectors have an
  easy job. MANDA edges NI-Diff on zero-day AUROC (94.39% vs 90.84%) while
  losing on every thresholded zero-day metric and on both adversarial scenarios.
* **DeepRP collapses on adversarial** (1.05% / 0.52% TPR). Its gradient number
  actually matches the paper (3.96%), but the paper's DeepRP holds up on GAN
  flows (94.97%) and ours does not.
* **FPR is not a differentiator in our protocol.** Every detector is calibrated
  to the same 5% target, so the FPR column lands at 4.7–5.3% by construction.
  The paper's spread (2.02%–13.87%) implies a different threshold rule, which it
  does not state.
* **GAN ASR is 92.88%, not 99.97%.** The discriminator collapses (D loss → 0)
  while the generator keeps pushing, so the last few percent of flows never
  cross the boundary under the mutable-feature constraint.

## Figures

`python nidiff/scripts/plot_figures.py` writes into the run directory:

* `fig2_timesteps.png` — the Fig. 2 sweep.
* `fig_scores.png` — the KLD score distributions. This is the mechanism in one
  picture: 30% of in-distribution flows round-trip to *exactly* zero KLD and the
  rest peak at 1e-7, while both adversarial families pile up around 1e-1, five
  orders of magnitude right of the threshold. Zero-day flows spread across the
  whole range, which is exactly why they are the harder case at 70% TPR.

## Faithfulness notes

Followed exactly:

* Classifier architectures of Table I; `large` is the default because the paper
  uses it for CICIoT.
* VAE: 6 conv layers per side, 64 channels, pool/upsample after every second
  layer except the last pair, encoder ending in 2 channels and decoder in 1;
  loss `MSE + 1e-6·KLD + 1e-3·perceptual`.
* U-Net of Table II with GroupNorm + SiLU; DDPM with `β ∈ [1e-4, 0.02]` linear,
  `T = 1000`; Adam `1e-4 / 1e-4 / 1e-5` for classifier / VAE / diffusion with
  batch 1024 and EMA on the diffusion weights.
* Detection by KLD between the two softmax outputs with **one-step**
  diffusion-denoising, thresholded at `T`.
* Zero-day protocol: the four super classes `BruteForce, Spoofing, Recon,
  WebBased` are held out, the classifier only ever sees the other four.
* Adversarial attacks are targeted at the benign class and may only move
  statistical features; symbolic features stay frozen.

Choices the paper leaves open, decided here:

* **Perceptual network.** Eq. 2 cites Johnson et al. but not which network
  provides the features. We use the frozen DNN classifier's penultimate
  embedding — the natural analogue for flow data, and it is what "semantic
  properties of the reconstructed network traffic" has to mean in this pipeline.
* **Preprocessing.** `log1p` on non-negative columns, then z-score fitted on
  known-class training flows only, then clipping to `|z| ≤ 10`. Without the clip
  a few near-constant flag counters (`cwr`, `ece`, rare protocols) reach `|z| ≈ 280`
  and dominate the VAE reconstruction loss; 0.03% of values are affected.
* **Threshold.** `T` is the `1 − target_fpr` quantile of the scores on a
  held-out in-distribution validation split (10% of training flows), with
  `target_fpr = 0.05`. The paper reports the resulting test FPR but not how `T`
  was picked.
* **Which adversarial flows get scored.** Only the ones that actually fooled the
  classifier (`detect.adv_successful_only`), which is the usual convention for
  adversarial-detection evaluation.
* **MANDA / DeepRP / NIDS-DA** have no public code. They are re-implementations
  of the mechanism each paper describes — kNN manifold distance plus prediction
  instability, random-subspace Mahalanobis, and auto-encoder reconstruction
  error respectively. GEN, Energy and GradNorm are exact.

Where this deviates from the paper by necessity:

* **Dataset scale.** The paper resamples 1.2M flows across 34 classes (20 known).
  The available extraction is 160k training flows over 8 *super* classes, so the
  classifier sees 4 known classes rather than 20. Absolute numbers will differ
  from Table V/VII/VIII; the ranking between NI-Diff and the six baselines is
  what transfers.
* **No ACI-IoT-2023 by choice**, hence no Table IV/VI and no `NI-Diff-base`
  headline numbers. The architecture itself is implemented and selectable with
  `--arch base`, so adding a second dataset later needs only a loader: point
  `DataConfig` at the new CSVs and redefine `MUTABLE_FEATURES` for its schema.
  Everything downstream — models, diffusion, attacks, detectors, metrics — is
  schema-agnostic.
