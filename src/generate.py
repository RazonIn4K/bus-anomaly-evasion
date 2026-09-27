"""Synthetic CAN-bus-style telemetry.

Emit labeled normal traffic and attack streams (injection, spoof, flood, drop,
replay). Normal = ~15 message IDs, each with a nominal period + jitter, carrying
bounded signals (random-walk physical values, counters, constants).

TODO (Phase 2 / Claude Code): implement. Fix seeds. Return labeled frames.
"""
