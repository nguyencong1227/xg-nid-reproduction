"""A from-scratch reimplementation of the XG-NID framework.

Component map (paper section -> module):

1. Flow and Feature Generator   3.1.1 -> :mod:`xgnid.step1_flow_extract`
2. Explainable Feature Extractor 3.1.2 -> :mod:`xgnid.step2_expl_features`
3. Graph Generator              3.1.3 -> :mod:`xgnid.step3_graph`
4. HGNN Model                   3.1.4 -> :mod:`xgnid.step4_model`
5. Integrated Gradient Explainer 3.1.5 -> :mod:`xgnid.step6_ig_explain`
6. Generative Explainer         3.1.6 -> :mod:`xgnid.step7_llm_explain`
"""
__version__ = "0.1.0"
