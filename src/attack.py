"""Adversarial evasion.

(a) Mimicry (black-box): search over injection rate / timing / free bytes to
    minimize reconstruction error while still delivering the malicious payload.
(b) PGD (white-box): gradient descent on window features to minimize
    reconstruction error, payload byte held fixed, projected to valid bus
    constraints, then mapped back to a frame schedule.

TODO (Phase 2 / Claude Code): implement. Evasion must beat the naive attack and
stay feasible.
"""
