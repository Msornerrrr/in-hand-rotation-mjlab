"""Learned world models of the LEAP hand + object for SuperMimic-style policy learning (arXiv 2610.28480).

Branch ``wm-inhand`` (2026-10-25). Step 1 is the fidelity test: record transitions of the in-hand tasks
(``record``), fit a deterministic multi-step world model (``model``, ``train``) and measure how well it predicts the
object over 1-32 control steps and whether its gradients agree with the simulator's.
"""
