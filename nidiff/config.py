"""Hyper-parameters of NI-Diff (Zhang et al., MILCOM'25).

Every default here is taken from the paper's "Training Recipe" and Tables I/II.
Values the paper leaves unspecified are marked with `# ours`.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os


@dataclass
class DataConfig:
    # 'cic_raw' = the official CIC-IoT2023 release (39 features, 34 fine
    # classes) the paper uses; 'xgnid' = the XG-NID nfstream re-extraction
    # (82 features, 8 super classes only).
    dataset: str = 'cic_raw'
    raw_cache_npz: str = '/home/congnc/data/ciciot2023_45k.npz'
    # What the DNN classifier predicts. 'fine' is the default because it is the
    # literal reading and the one the paper's own wording supports: for ACI it
    # says "the classifier only uses 9 classes of data for training" (12 classes,
    # 3 held out) where no super-class grouping exists, so that head must be
    # 9-way fine; the CIC sentence "leverages the remaining 20 classes" is the
    # same construction. 'super' (a 4-way head over the known super classes,
    # trained on the same data) was tried because it lands closer to the paper's
    # 99.42% accuracy -- but neither reaches it (83.7% fine, 87.9% super), so the
    # accuracy figure cannot settle the ambiguity and the wording wins.
    label_granularity: str = 'fine'
    train_csv: str = 'work/df_class_8_train_cleaned.csv'
    test_csv: str = 'work/df_class_8_test_cleaned.csv'
    cache_npz: str = 'work/baseline_tabular_cache.npz'
    label_col: str = 'Label'
    # CIC-IoT2023 super classes as encoded in df_class_8_*.csv
    class_names: List[str] = field(default_factory=lambda: [
        'Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce'])
    # Paper (Sec. IV, Attack Model): for CICIoT the 4 super classes Brute force,
    # Spoofing, Recon and Web-based are held out as zero-day.
    zero_day_classes: List[str] = field(default_factory=lambda: [
        'BruteForce', 'Spoofing', 'Recon', 'WebBased'])
    benign_class: str = 'Benign'
    log1p: bool = True          # ours: heavy-tailed flow counters
    clip: float = 10.0          # ours: rare flag counters blow up to |z| ~ 280
    val_frac: float = 0.1       # ours: held-out ID split used to pick threshold T
    seed: int = 42


@dataclass
class ClassifierConfig:
    arch: str = 'large'         # 'base' (Table I left) | 'large' (Table I right)
    lr: float = 1e-4
    batch_size: int = 1024
    epochs: int = 50


@dataclass
class VAEConfig:
    channels: int = 64          # Table: uniform channel dim of 64
    n_layers: int = 6           # 6 layers of 1x3 1D-CNN, encoder and decoder
    # Pooling stages. None = the paper's "after every two layers except the last
    # pair" (n_layers//2 - 1), which is a 1x9 latent on 39 features. Lower it to
    # widen the bottleneck without changing the layer count.
    n_pool: Optional[int] = None
    lambda_kld: float = 1e-6    # lambda_1 in Eq. 2
    lambda_perc: float = 1e-3   # lambda_2 in Eq. 2
    lr: float = 1e-4
    batch_size: int = 1024
    epochs: int = 100


@dataclass
class DiffusionConfig:
    timesteps: int = 1000       # maximum timestep
    beta_start: float = 1e-4
    beta_end: float = 0.02      # linear increasing schedule
    lr: float = 1e-5
    batch_size: int = 1024
    epochs: int = 1000
    ema_decay: float = 0.999    # exponential moving weight average
    denoise_steps: int = 1      # one-step diffusion-denoising (Sec. III)


@dataclass
class AttackConfig:
    # gradient-based, FENCE-style constrained targeted attack -> benign.
    # The paper states no budget, so these are ours, and they are *schema
    # dependent*: with only 11 of 39 features mutable (cic_raw) each one has to
    # move much further than with 29 of 82 (xgnid). Tuned on a held-out sweep:
    #   cic_raw : eps 10, alpha 1.0  -> ASR 77% (eps 1.0 gives 0.1%)
    #   xgnid   : eps 1.0, alpha 0.1 -> ASR 84%
    # Above eps 10 the return flattens; above ~30 the sign-steps overshoot and
    # ASR collapses.
    grad_eps: float = 10.0
    grad_alpha: float = 1.0
    grad_steps: int = 100
    # GAN-based (Alhajjar et al.)
    gan_noise_dim: int = 32
    gan_hidden: int = 256
    gan_epochs: int = 200
    gan_lr: float = 1e-4
    gan_batch_size: int = 512
    gan_lambda_adv: float = 1.0     # fool-the-classifier weight
    gan_lambda_real: float = 1.0    # realism (discriminator) weight
    # Max perturbation in standardised space -- the GAN's equivalent of grad_eps,
    # and equally schema dependent:
    #   cic_raw : 20  -> ASR 69%  (1.0 gives 0.03%; 5 -> 44%, 10 -> 49%)
    #   xgnid   : 1.0 -> ASR 93%
    # Rises monotonically, unlike PGD, because the generator *learns* the
    # perturbation instead of taking fixed sign steps. It saturates near 20:
    # DataConfig.clip bounds the standardised space to [-10, 10], so a scale of
    # 20 already spans the whole box and more budget buys nothing.
    gan_pert_scale: float = 20.0
    max_attack_samples: int = 20000  # ours: cap for tractable evaluation
    batch_size: int = 512       # PGD keeps an autograd graph -> keep this small


@dataclass
class DetectConfig:
    target_fpr: float = 0.05    # T is the (1-target_fpr) quantile of ID-val scores
    kld_direction: str = 'forward'   # D_KL(y || y')
    use_mu: bool = False        # sample z ~ q(z|x) as in Alg. 1
    adv_successful_only: bool = True  # score only the flows that fooled the DNN
    # denoising-timestep sweep of Fig. 2
    ablation_steps: List[int] = field(default_factory=lambda: [
        1, 5, 10, 25, 50, 100, 200, 400, 600, 800, 1000])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    clf: ClassifierConfig = field(default_factory=ClassifierConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    diff: DiffusionConfig = field(default_factory=DiffusionConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    root: str = '/storage/congnc/GNN4ID'
    out_dir: str = 'nidiff/work/nidiff'
    device: str = 'cuda'
    seed: int = 42
    eval_batch_size: int = 1024   # inference-time batching for every detector

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def out(self, *parts: str) -> str:
        p = os.path.join(self.root, self.out_dir, *parts)
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
        return p

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.out('config.json')
        with open(path, 'w') as fh:
            json.dump(asdict(self), fh, indent=2)
        return path


def smoke(cfg: Config) -> Config:
    """Tiny budget so the whole pipeline can be exercised in a few minutes."""
    cfg.clf.epochs = 3
    cfg.vae.epochs = 3
    cfg.diff.epochs = 5
    cfg.attack.grad_steps = 20
    cfg.attack.gan_epochs = 10
    cfg.attack.max_attack_samples = 2000
    cfg.out_dir = 'nidiff/work/nidiff_smoke'
    return cfg
