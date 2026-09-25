"""NI-Diff -- zero-day and adversarial network intrusion detection with
diffusion models (Zhang et al., MILCOM 2025), reproduced on CIC-IoT2023.

    from nidiff.config import Config, smoke
    from nidiff.pipeline import run
    run(Config())
"""

__all__ = ['config', 'data', 'models', 'diffusion', 'train', 'attacks',
           'detectors', 'metrics', 'pipeline']
__version__ = '0.1.0'
